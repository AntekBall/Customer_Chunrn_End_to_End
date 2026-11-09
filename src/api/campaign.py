"""Scenario-only retention estimates, not measured campaign or causal returns.

Capacity and positive-value selection always use the full loaded snapshot.
Display filters never reallocate that selection. No labels or new scoring used.
"""

from collections.abc import Callable
from threading import Lock
from typing import Annotated, Protocol

import duckdb
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.dashboard import PortfolioFilters
from src.data.portfolio import apply_campaign_overlay, summarize_campaign_segments
from src.models.campaign import CampaignScenario
from src.models.predict import ChurnPredictor


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ScenarioInput(APIModel):
    contact_cost: Annotated[float, Field(ge=0)] = 2.0
    gross_margin: Annotated[float, Field(ge=0, le=1)] = .2
    retained_months: Annotated[float, Field(gt=0)] = 6.0
    prevention_probability: Annotated[float, Field(ge=0, le=1)] = .2
    capacity_fraction: Annotated[float, Field(ge=0, le=1)] = .1

    @model_validator(mode="before")
    @classmethod
    def reject_booleans(cls, values):
        if isinstance(values, dict) and any(
            isinstance(value, bool) for value in values.values()
        ):
            raise ValueError("Scenario numbers must not be booleans.")
        return values


class SimulationRequest(APIModel):
    scenario: ScenarioInput = Field(default_factory=ScenarioInput)
    display_filters: PortfolioFilters = Field(default_factory=PortfolioFilters)


class CampaignTotals(APIModel):
    customers: int
    eligible_customers: int
    selected_customers: int
    eligible_fraction: float | None
    selected_fraction: float | None
    eligible_expected_net_value: float
    expected_churners_contacted: float
    expected_prevented_churn: float
    expected_retained_contribution: float
    campaign_cost: float
    expected_net_value: float
    scenario_roi: float | None


class CampaignCell(APIModel):
    revenue_segment: str
    risk_segment: str
    customers: int
    eligible_customers: int
    selected_customers: int
    mean_expected_net_value: float | None
    eligible_expected_net_value: float
    selected_expected_net_value: float
    eligible_fraction: float | None
    selected_fraction: float | None


class SimulationResponse(APIModel):
    model_run_id: str
    freeze_run_id: str
    final_test_run_id: str
    scenario: ScenarioInput
    policy: str
    selection_scope: str
    capacity_customers: int
    unused_capacity: int
    global_summary: CampaignTotals
    displayed_summary: CampaignTotals
    displayed_segments: list[CampaignCell]
    display_filters: PortfolioFilters
    interpretation: str


class SimulationServices(Protocol):
    predictor: ChurnPredictor
    portfolio: duckdb.DuckDBPyConnection
    lock: Lock


def _totals(frame: pd.DataFrame, scenario: CampaignScenario) -> dict:
    selected = frame.loc[frame["selected"]]
    count = len(frame)
    cost = float(len(selected) * scenario.contact_cost)
    contribution = float(selected["expected_retained_contribution"].sum())
    return {
        "customers": count,
        "eligible_customers": int(frame["eligible"].sum()),
        "selected_customers": len(selected),
        "eligible_fraction": float(frame["eligible"].mean()) if count else None,
        "selected_fraction": len(selected) / count if count else None,
        "eligible_expected_net_value": float(frame["eligible_expected_net_value"].sum()),
        "expected_churners_contacted": float(selected["churn_probability"].sum()),
        "expected_prevented_churn": float(selected["expected_prevented_churn"].sum()),
        "expected_retained_contribution": contribution,
        "campaign_cost": cost,
        "expected_net_value": contribution - cost,
        "scenario_roi": (contribution - cost) / cost if cost > 0 else None,
    }


def simulate_campaign(
    portfolio: pd.DataFrame, request: SimulationRequest,
) -> dict:
    """Keep global selection intact through all display filters."""
    scenario = CampaignScenario(**request.scenario.model_dump())
    overlay = apply_campaign_overlay(portfolio, scenario)
    global_summary = _totals(overlay, scenario)
    displayed = overlay
    for column, value in request.display_filters.model_dump(exclude_none=True).items():
        displayed = displayed.loc[displayed[column].eq(value)]
    cells = summarize_campaign_segments(displayed).reset_index()
    columns = list(CampaignCell.model_fields)
    cells = cells.loc[:, columns]
    capacity = int(scenario.capacity_fraction * len(overlay))
    return {
        "scenario": request.scenario,
        "policy": "expected_profit_positive_only",
        "selection_scope": "full_snapshot_before_display_filters",
        "capacity_customers": capacity,
        "unused_capacity": capacity - global_summary["selected_customers"],
        "global_summary": global_summary,
        "displayed_summary": _totals(displayed, scenario),
        "displayed_segments": cells.astype(object).where(cells.notna(), None).to_dict(
            orient="records",
        ),
        "display_filters": request.display_filters,
        "interpretation": (
            "Assumption-based estimates, not measured causal returns. Includes source "
            "training customers with in-sample probabilities. Snapshot membership does "
            "not establish active status or operational contact consent. Filters only "
            "change display; no capacity is reallocated. Eligibility requires net > 0; "
            "ranking is net descending, customer ID ascending. ROI is null at zero cost."
        ),
    }


def register_campaign_routes(
    app: FastAPI, get_services: Callable[[Request], SimulationServices],
) -> None:
    @app.post("/campaign/simulate", response_model=SimulationResponse)
    def campaign_simulation(body: SimulationRequest, request: Request):
        services = get_services(request)
        # Copy only the needed columns. Keep the shared connection and simulation
        # under the existing lock to bound concurrent large-frame allocations.
        with services.lock:
            frame = services.portfolio.sql("""
                SELECT customer_id, monthlycharges, churn_probability,
                       revenue_segment, risk_segment, partition, contract,
                       payment_method, has_internet_service
                FROM portfolio
            """).df()
            result = simulate_campaign(frame, body)
        provenance = services.predictor.provenance
        return {
            "model_run_id": provenance.source_finalist_run_id,
            "freeze_run_id": provenance.freeze_run_id,
            "final_test_run_id": provenance.final_test_run_id,
            **result,
        }
