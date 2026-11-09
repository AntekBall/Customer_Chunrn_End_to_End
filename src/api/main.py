import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.campaign import register_campaign_routes
from src.data.dashboard import PortfolioFilters, open_portfolio, summarize_portfolio
from src.data.loader import PROJECT_ROOT
from src.models.explain import ChurnExplainer
from src.models.predict import ChurnPredictor
from src.models.serving_bundle import load_bundle


logger = logging.getLogger(__name__)
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Binary = Literal[0, 1] | bool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Customer(StrictModel):
    """All fields are required; five numeric fields may explicitly be null."""

    customer_id: Text
    age: Count
    annual_income: Nonnegative | None
    dependents: Count
    tenure: Count
    monthlycharges: Nonnegative
    totalcharges: Nonnegative
    num_services: Count
    customer_satisfaction: Nonnegative | None
    num_complaints: Nonnegative | None
    num_service_calls: Count
    late_payments: Count
    avg_monthly_gb: Nonnegative | None
    days_since_last_interaction: Count
    credit_score: Nonnegative | None
    gender: Text
    education: Text
    marital_status: Text
    contract: Text
    payment_method: Text
    paperless_billing: Binary
    has_phone_service: Binary
    has_internet_service: Binary
    has_online_security: Binary
    has_online_backup: Binary
    has_device_protection: Binary
    has_tech_support: Binary
    has_streaming_tv: Binary
    has_streaming_movies: Binary

    @model_validator(mode="before")
    @classmethod
    def reject_boolean_numbers(cls, values):
        if isinstance(values, dict):
            binaries = {
                "paperless_billing", "has_phone_service", "has_internet_service",
                "has_online_security", "has_online_backup", "has_device_protection",
                "has_tech_support", "has_streaming_tv", "has_streaming_movies",
            }
            for name in cls.model_fields:
                if name in binaries:
                    value = values.get(name)
                    if value is not None and type(value) not in (int, bool):
                        raise ValueError(f"{name} requires integer 0/1 or boolean.")
                elif isinstance(values.get(name), bool):
                    raise ValueError(f"{name} does not accept boolean values.")
        return values


class BatchRequest(StrictModel):
    customers: Annotated[list[Customer], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [customer.customer_id for customer in self.customers]
        if len(set(ids)) != len(ids):
            raise ValueError("Customer IDs must be unique within the batch.")
        return self


class Provenance(StrictModel):
    model_run_id: str
    freeze_run_id: str
    final_test_run_id: str


class Prediction(Provenance):
    customer_id: str
    churn_probability: Annotated[float, Field(ge=0, le=1)]


class BatchResponse(StrictModel):
    predictions: list[Prediction]


class Health(Provenance):
    status: Literal["ready"]
    portfolio_customers: int


class Contribution(StrictModel):
    feature: str
    feature_value: str | float | None
    shap_log_odds: float


class Explanation(Provenance):
    customer_id: str
    churn_probability: Annotated[float, Field(ge=0, le=1)]
    base_log_odds: float
    raw_log_odds: float
    output_scale: Literal["log_odds"]
    method: Literal["native_catboost_tree_shap"]
    interpretation: str
    contributions: list[Contribution]


class Metrics(StrictModel):
    customers: int
    monthly_charge_total: float
    mean_monthly_charge: float | None
    mean_predicted_churn: float | None
    risk_weighted_monthly_charges: float
    customer_share: float | None
    monthly_charge_share: float | None


class SegmentMetrics(Metrics):
    revenue_segment: str
    risk_segment: str


class SummaryResponse(Provenance):
    overview: Metrics
    segments: list[SegmentMetrics]
    filters: PortfolioFilters
    interpretation: str


class PortfolioQueryFilters(PortfolioFilters):
    # Query strings carry booleans as text; keep the library model strict.
    has_internet_service: Annotated[bool, Field(strict=False)] | None = None


@dataclass
class Services:
    predictor: ChurnPredictor
    explainer: ChurnExplainer
    portfolio: duckdb.DuckDBPyConnection
    portfolio_customers: int
    lock: Lock = field(default_factory=Lock)


def _records(frame: pd.DataFrame) -> list[dict]:
    # DataFrame NaN/NA means "undefined" in these outputs; JSON requires null.
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _provenance(predictor: ChurnPredictor) -> dict[str, str]:
    source = predictor.provenance
    return {
        "model_run_id": source.source_finalist_run_id,
        "freeze_run_id": source.freeze_run_id,
        "final_test_run_id": source.final_test_run_id,
    }


def _services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Service is not ready.")
    return services


def create_app(
    *,
    bundle_directory: Path | None = None,
    tracking_directory: Path | None = None,
    portfolio_directory: Path | None = None,
    final_test_run_id: str | None = None,
) -> FastAPI:
    bundle_setting = os.environ.get("CHURN_BUNDLE_DIRECTORY")
    if bundle_setting is not None and not bundle_setting.strip():
        raise ValueError("CHURN_BUNDLE_DIRECTORY must not be blank.")
    bundle = bundle_directory if bundle_directory is not None else (
        Path(bundle_setting) if bundle_setting else None
    )
    if bundle is not None and (
        any(value is not None for value in (
            tracking_directory, portfolio_directory, final_test_run_id,
        ))
        or any(name in os.environ for name in (
            "CHURN_TRACKING_DIRECTORY", "CHURN_PORTFOLIO_DIRECTORY",
            "CHURN_FINAL_TEST_RUN_ID",
        ))
    ):
        raise ValueError("Choose bundle configuration or MLflow configuration, not both.")
    tracking = tracking_directory if tracking_directory is not None else Path(
        os.environ.get("CHURN_TRACKING_DIRECTORY", str(PROJECT_ROOT / "mlruns"))
    )
    portfolio = portfolio_directory if portfolio_directory is not None else Path(
        os.environ.get(
            "CHURN_PORTFOLIO_DIRECTORY",
            str(PROJECT_ROOT / "data" / "processed" / "portfolio_v1"),
        )
    )
    selected_run = final_test_run_id if final_test_run_id is not None else os.environ.get(
        "CHURN_FINAL_TEST_RUN_ID", "5818be9683af42a485ffc2896a868137",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if bundle is not None:
            loaded = load_bundle(bundle)
            predictor = loaded.predictor
            snapshot_directory = loaded.portfolio_directory
        else:
            predictor = ChurnPredictor.from_final_test(
                tracking, final_test_run_id=selected_run,
            )
            snapshot_directory = portfolio
        explainer = ChurnExplainer(predictor)
        with open_portfolio(snapshot_directory) as connection:
            versions = connection.sql("""
                SELECT DISTINCT model_run_id, freeze_run_id, final_test_run_id
                FROM portfolio
            """).fetchall()
            provenance = _provenance(predictor)
            if versions != [tuple(provenance.values())]:
                raise ValueError("Portfolio snapshot and serving model provenance differ.")
            # Validate the SQL and output contract before reporting readiness.
            summary = summarize_portfolio(connection)
            Metrics.model_validate(_records(summary.overview)[0])
            rows = int(summary.overview.iloc[0]["customers"])
            app.state.services = Services(predictor, explainer, connection, rows)
            logger.info("Churn API ready; portfolio rows=%d", rows)
            try:
                yield
            finally:
                app.state.services = None

    app = FastAPI(
        title="Customer Churn & Revenue-at-Risk API",
        version="0.1.0", lifespan=lifespan,
        description=(
            "Synthetic snapshot analytics. Predictions have no established time horizon. "
            "Charge exposure is not verified MRR or realized loss; SHAP is not causal."
        ),
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Do not echo customer fields, nonfinite JSON values or exception objects.
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]})

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception):
        logger.error(
            "API operation failed: %s %s", request.method, request.url.path,
            exc_info=(type(error), error, error.__traceback__),
        )
        return JSONResponse(status_code=500, content={"detail": "Internal service error."})

    @app.get("/health", response_model=Health)
    def health(request: Request):
        services = _services(request)
        return {
            "status": "ready", "portfolio_customers": services.portfolio_customers,
            **_provenance(services.predictor),
        }

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer, request: Request):
        services = _services(request)
        with services.lock:
            return services.predictor.predict_one(customer.model_dump())

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request):
        services = _services(request)
        with services.lock:
            predictions = services.predictor.predict_batch(pd.DataFrame([
                customer.model_dump() for customer in batch.customers
            ]))
        return {"predictions": _records(predictions)}

    @app.post("/explain", response_model=Explanation)
    def explain(customer: Customer, request: Request):
        services = _services(request)
        with services.lock:
            result = services.explainer.explain_one(customer.model_dump())
        return {
            **_provenance(services.predictor),
            "customer_id": result.customer_id,
            "churn_probability": result.churn_probability,
            "base_log_odds": result.base_log_odds,
            "raw_log_odds": result.raw_log_odds,
            "output_scale": "log_odds",
            "method": "native_catboost_tree_shap",
            "interpretation": "Model contributions, not probability points or causal effects.",
            "contributions": _records(result.contributions),
        }

    @app.get("/portfolio/summary", response_model=SummaryResponse)
    def portfolio_summary(
        request: Request, filters: Annotated[PortfolioQueryFilters, Query()],
    ):
        services = _services(request)
        validated_filters = PortfolioFilters.model_validate(filters.model_dump())
        with services.lock:
            result = summarize_portfolio(services.portfolio, validated_filters)
        return {
            **_provenance(services.predictor),
            "overview": _records(result.overview)[0],
            "segments": _records(result.segments.reset_index()),
            "filters": validated_filters,
            "interpretation": (
                "Descriptive snapshot; training scores are in-sample. Shares use the "
                "filtered population. Charges are not verified MRR or realized losses."
            ),
        }

    register_campaign_routes(app, _services)
    return app


app = create_app()
