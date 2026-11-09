"""HTTP-only dashboard data access; importing this module performs no I/O."""

import math
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.dashboard import PortfolioFilters
from src.data.portfolio import REVENUE_SEGMENTS, RISK_SEGMENTS
from src.models.pipeline import FEATURE_COLUMNS


class APIError(RuntimeError):
    """An unavailable API or invalid response, safe to display to the user."""


Finite = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[Finite, Field(ge=0, le=1)]


class Metrics(BaseModel):
    model_config = ConfigDict(strict=True)

    customers: Annotated[int, Field(ge=0)]
    monthly_charge_total: Annotated[Finite, Field(ge=0)]
    mean_monthly_charge: Annotated[Finite, Field(ge=0)] | None
    mean_predicted_churn: Probability | None
    risk_weighted_monthly_charges: Annotated[Finite, Field(ge=0)]
    customer_share: Probability | None
    monthly_charge_share: Probability | None


class Segment(Metrics):
    revenue_segment: Literal["Lower half", "Middle 40%", "High revenue", "Very high revenue"]
    risk_segment: Literal["Lower risk", "Elevated risk", "Highest risk"]


class Summary(BaseModel):
    model_config = ConfigDict(strict=True)

    model_run_id: Annotated[str, Field(min_length=1)]
    freeze_run_id: Annotated[str, Field(min_length=1)]
    final_test_run_id: Annotated[str, Field(min_length=1)]
    overview: Metrics
    segments: list[Segment]
    filters: PortfolioFilters
    interpretation: str

    @model_validator(mode="after")
    def complete_matrix(self):
        labels = {(row.revenue_segment, row.risk_segment) for row in self.segments}
        expected = {(revenue, risk) for revenue in REVENUE_SEGMENTS for risk in RISK_SEGMENTS}
        if len(self.segments) != 12 or labels != expected:
            raise ValueError("Expected exactly the 12 revenue/risk cells.")
        if sum(row.customers for row in self.segments) != self.overview.customers:
            raise ValueError("Segment counts do not reconcile with the overview.")
        for column in ("monthly_charge_total", "risk_weighted_monthly_charges"):
            if not math.isclose(
                sum(getattr(row, column) for row in self.segments),
                getattr(self.overview, column), rel_tol=1e-8, abs_tol=1e-6,
            ):
                raise ValueError(f"Segment {column} does not reconcile with the overview.")
        return self


class Contribution(BaseModel):
    model_config = ConfigDict(strict=True)

    feature: str
    feature_value: str | Finite | None
    shap_log_odds: Finite


class Explanation(BaseModel):
    model_config = ConfigDict(strict=True)

    customer_id: str
    model_run_id: Annotated[str, Field(min_length=1)]
    freeze_run_id: Annotated[str, Field(min_length=1)]
    final_test_run_id: Annotated[str, Field(min_length=1)]
    churn_probability: Probability
    base_log_odds: Finite
    raw_log_odds: Finite
    output_scale: Literal["log_odds"]
    method: Literal["native_catboost_tree_shap"]
    interpretation: str
    contributions: list[Contribution]

    @model_validator(mode="after")
    def reconstruct_output(self):
        names = [row.feature for row in self.contributions]
        if len(names) != len(FEATURE_COLUMNS) or set(names) != set(FEATURE_COLUMNS):
            raise ValueError("Expected one SHAP contribution per predictor.")
        raw = self.base_log_odds + sum(row.shap_log_odds for row in self.contributions)
        if not math.isclose(raw, self.raw_log_odds, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("SHAP contributions do not reconstruct the raw output.")
        exp_value = math.exp(-abs(self.raw_log_odds))
        probability = (
            1 / (1 + exp_value) if self.raw_log_odds >= 0 else exp_value / (1 + exp_value)
        )
        if not math.isclose(probability, self.churn_probability, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("SHAP output does not reconstruct the predicted probability.")
        return self


class PortfolioClient:
    def __init__(
        self, base_url: str, *, transport: httpx.BaseTransport | None = None,
    ) -> None:
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host or url.query or url.fragment:
            raise ValueError("CHURN_API_URL must be an HTTP(S) base URL without query/fragment.")
        self._base_url = base_url.rstrip("/") + "/"
        self._transport = transport

    def summary(self, filters: PortfolioFilters) -> Summary:
        params = {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in filters.model_dump(exclude_none=True).items()
        }
        response = self._request("GET", "portfolio/summary", params=params)
        try:
            summary = Summary.model_validate(response.json())
        except ValueError as error:
            raise APIError("The portfolio API returned an invalid summary response.") from error
        if summary.filters != filters:
            raise APIError("The portfolio API returned results for different filters.")
        return summary

    def explain(self, customer: dict[str, object]) -> Explanation:
        response = self._request("POST", "explain", json=customer)
        try:
            explanation = Explanation.model_validate(response.json())
        except ValueError as error:
            raise APIError("The API returned an invalid explanation response.") from error
        if explanation.customer_id != customer["customer_id"] or any(
            row.feature_value != customer[row.feature] for row in explanation.contributions
        ):
            raise APIError("The API returned an explanation for different customer inputs.")
        return explanation

    def _request(
        self, method: str, endpoint: str, *,
        params: dict[str, str] | None = None, json: dict[str, object] | None = None,
    ) -> httpx.Response:
        try:
            with httpx.Client(
                base_url=self._base_url, timeout=httpx.Timeout(30.0, connect=5.0),
                transport=self._transport,
            ) as client:
                response = client.request(method, endpoint, params=params, json=json)
                response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise APIError(
                f"Churn API returned HTTP {error.response.status_code}. "
                "Check the API logs, then retry."
            ) from error
        except httpx.RequestError as error:
            raise APIError(
                "Cannot reach the churn API. Start the backend and check CHURN_API_URL."
            ) from error
        return response
