from contextlib import contextmanager
from threading import Lock
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.campaign import (
    ScenarioInput, SimulationRequest, register_campaign_routes,
    simulate_campaign,
)
from src.data.portfolio import apply_portfolio_segments, fit_portfolio_thresholds
from src.models.campaign import CampaignScenario, compare_policies
from src.models.predict import ModelProvenance


@pytest.fixture
def portfolio():
    frame = pd.DataFrame({
        "customer_id": ["d", "c", "b", "a"],
        "monthlycharges": [100., 100., 100., 100.],
        "churn_probability": [.1, .2, .5, .5],
        "partition": ["train"] * 4,
        "contract": ["A", "A", "B", "B"],
        "payment_method": ["card"] * 4,
        "has_internet_service": [1, 0, 1, 0],
    })
    cuts = fit_portfolio_thresholds(frame, model_run_id="source")
    return apply_portfolio_segments(frame, cuts, model_run_id="source")


@contextmanager
def client_for(frame):
    app = FastAPI()
    with duckdb.connect() as connection:
        connection.register("fixture", frame)
        connection.sql("CREATE TABLE portfolio AS SELECT * FROM fixture")
        services = SimpleNamespace(
            portfolio=connection, lock=Lock(), predictor=SimpleNamespace(
                provenance=ModelProvenance("model", "source", "freeze", "final"),
            ),
        )
        register_campaign_routes(app, lambda request: services)
        with TestClient(app) as client:
            yield client


@pytest.mark.parametrize("capacity", [.1, .5, 1.])
def test_route_default_matches_existing_policy(portfolio, capacity):
    _, policies = compare_policies(
        portfolio, portfolio["churn_probability"], CampaignScenario(capacity_fraction=capacity),
    )
    expected = policies.set_index("policy").loc["expected_profit_positive_only"]
    with client_for(portfolio) as client:
        response = client.post("/campaign/simulate", json={
            "scenario": {"capacity_fraction": capacity},
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["model_run_id"] == "source"
    assert result["capacity_customers"] == int(capacity * 4)
    assert len(result["displayed_segments"]) == 12
    for column in (
        "selected_customers", "expected_net_value", "campaign_cost",
        "expected_churners_contacted", "expected_prevented_churn",
        "expected_retained_contribution",
    ):
        assert result["global_summary"][column] == pytest.approx(expected[column])
    if result["global_summary"]["campaign_cost"] == 0:
        assert result["global_summary"]["scenario_roi"] is None
    else:
        assert result["global_summary"]["scenario_roi"] == pytest.approx(expected["scenario_roi"])


def test_filter_does_not_reallocate_global_capacity(portfolio):
    body = {
        "scenario": {"capacity_fraction": .5},
        "display_filters": {"contract": "A"},
    }
    with client_for(portfolio) as client:
        result = client.post("/campaign/simulate", json=body).json()
        full = client.post("/campaign/simulate", json={
            "scenario": {"capacity_fraction": .5},
        }).json()
    assert result["global_summary"] == full["global_summary"]
    assert result["global_summary"]["selected_customers"] == 2
    assert result["displayed_summary"]["customers"] == 2
    assert result["displayed_summary"]["eligible_customers"] == 2
    assert result["displayed_summary"]["selected_customers"] == 0
    assert result["displayed_summary"]["scenario_roi"] is None
    assert sum(row["selected_customers"] for row in result["displayed_segments"]) == 0


def test_tie_break_and_floor(portfolio):
    from src.data.dashboard import PortfolioFilters

    one = SimulationRequest(
        scenario=ScenarioInput(capacity_fraction=.49),
        display_filters=PortfolioFilters(has_internet_service=False),
    )
    result = simulate_campaign(portfolio, one)
    assert result["capacity_customers"] == 1
    assert result["global_summary"]["selected_customers"] == 1
    # a wins the tie with b; a has no internet.
    assert result["displayed_summary"]["selected_customers"] == 1
    shuffled = simulate_campaign(portfolio.iloc[::-1], one)
    assert result["global_summary"] == shuffled["global_summary"]
    assert result["displayed_summary"] == shuffled["displayed_summary"]


@pytest.mark.parametrize("scenario", [
    {"capacity_fraction": 0.}, {"gross_margin": 0.},
    {"prevention_probability": 0.}, {"contact_cost": 1000.},
])
def test_no_selection(portfolio, scenario):
    with client_for(portfolio) as client:
        response = client.post("/campaign/simulate", json={"scenario": scenario})
    assert response.status_code == 200
    summary = response.json()["global_summary"]
    assert summary["selected_customers"] == 0
    assert summary["expected_net_value"] == 0
    assert summary["scenario_roi"] is None


def test_free_contact_roi_is_undefined(portfolio):
    with client_for(portfolio) as client:
        result = client.post("/campaign/simulate", json={
            "scenario": {"contact_cost": 0., "capacity_fraction": 1.},
        }).json()
    assert result["global_summary"]["selected_customers"] == 4
    assert result["global_summary"]["expected_net_value"] > 0
    assert result["global_summary"]["campaign_cost"] == 0
    assert result["global_summary"]["scenario_roi"] is None


def test_empty_display_and_reconciliation(portfolio):
    with client_for(portfolio) as client:
        empty = client.post("/campaign/simulate", json={
            "display_filters": {"contract": "absent"},
        }).json()
        full = client.post("/campaign/simulate", json={
            "scenario": {"capacity_fraction": .5},
        }).json()
    assert empty["displayed_summary"]["customers"] == 0
    assert empty["displayed_summary"]["eligible_fraction"] is None
    assert len(empty["displayed_segments"]) == 12
    assert all(row["mean_expected_net_value"] is None for row in empty["displayed_segments"])
    cells = full["displayed_segments"]
    for column in ("customers", "eligible_customers", "selected_customers",
                   "eligible_expected_net_value"):
        assert sum(row[column] for row in cells) == pytest.approx(
            full["global_summary"][column],
        )
    assert sum(row["selected_expected_net_value"] for row in cells) == pytest.approx(
        full["global_summary"]["expected_net_value"],
    )


@pytest.mark.parametrize("body", [
    {"scenario": {"contact_cost": -1}}, {"scenario": {"gross_margin": 1.1}},
    {"scenario": {"retained_months": 0}}, {"scenario": {"prevention_probability": -1}},
    {"scenario": {"capacity_fraction": 2}}, {"scenario": {"contact_cost": True}},
    {"scenario": {"gross_margin": "0.2"}}, {"scenario": {"unknown": 1}},
    {"display_filters": {"partition": "other"}},
    {"display_filters": {"has_internet_service": "false"}},
    {"unknown": 1},
])
def test_invalid_body(portfolio, body):
    with client_for(portfolio) as client:
        assert client.post("/campaign/simulate", json=body).status_code == 422


def test_net_zero_is_not_eligible(portfolio):
    # With margin=1, months=1 and prevention=1, p=.5 * 100 - 50 = 0.
    with client_for(portfolio) as client:
        result = client.post("/campaign/simulate", json={
            "scenario": {
                "contact_cost": 50., "gross_margin": 1., "retained_months": 1.,
                "prevention_probability": 1., "capacity_fraction": 1.,
            },
        }).json()
    assert result["global_summary"]["eligible_customers"] == 0
    assert result["global_summary"]["selected_customers"] == 0


def test_main_app_registers_route_and_requires_readiness():
    from src.api.main import create_app

    app = create_app()
    client = TestClient(app)
    assert "/campaign/simulate" in client.get("/openapi.json").json()["paths"]
    assert client.post("/campaign/simulate", json={}).status_code == 503
    client.close()
