import copy
import json

import httpx
import pandas as pd
import pytest

from src.api.campaign import SimulationRequest, simulate_campaign
from src.data.portfolio import apply_portfolio_segments, fit_portfolio_thresholds
from src.frontend.app import create_app
from src.frontend.client import APIError, PortfolioClient
from src.frontend.financial import simulation_request
from tests.test_frontend import callback


DEFAULT_VALUES = [2, 20, 6, 20, 50, *[None] * 6]


@pytest.fixture
def portfolio():
    frame = pd.DataFrame({
        "customer_id": ["d", "c", "b", "a"],
        "monthlycharges": [100., 100., 100., 100.],
        "churn_probability": [.1, .2, .5, .5],
        "partition": ["train"] * 4,
        "contract": ["month_to_month", "month_to_month", "two_year", "two_year"],
        "payment_method": ["credit_card"] * 4,
        "has_internet_service": [1, 0, 1, 0],
    })
    cuts = fit_portfolio_thresholds(frame, model_run_id="model")
    return apply_portfolio_segments(frame, cuts, model_run_id="model")


def response_payload(portfolio, request):
    result = simulate_campaign(portfolio, request)
    return {
        **result,
        "scenario": result["scenario"].model_dump(),
        "display_filters": result["display_filters"].model_dump(),
        "model_run_id": "model", "freeze_run_id": "freeze", "final_test_run_id": "final",
    }


def api_for(portfolio, calls):
    def handler(request):
        calls.append(request)
        body = SimulationRequest.model_validate_json(request.content)
        return httpx.Response(200, json=response_payload(portfolio, body))
    return PortfolioClient("http://test", transport=httpx.MockTransport(handler))


def test_percentages_convert_once():
    request = simulation_request(DEFAULT_VALUES)
    assert request.scenario.model_dump() == {
        "contact_cost": 2., "gross_margin": .2, "retained_months": 6.,
        "prevention_probability": .2, "capacity_fraction": .5,
    }
    values = DEFAULT_VALUES.copy()
    values[-1] = "no"
    assert simulation_request(values).display_filters.has_internet_service is False


@pytest.mark.parametrize("index,value", [
    (0, -1), (0, None), (0, True), (1, 101), (1, -1), (1, "20"),
    (1, True), (2, 0), (2, None), (3, 101), (4, 101),
    (4, None), (0, float("inf")), (5, "invalid"), (10, "invalid"),
])
def test_invalid_controls(index, value):
    values = DEFAULT_VALUES.copy()
    values[index] = value
    with pytest.raises(ValueError):
        simulation_request(values)


def test_http_request_and_full_response(portfolio):
    calls = []
    api = api_for(portfolio, calls)
    request = simulation_request(DEFAULT_VALUES)
    result = api.simulate(request)
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/campaign/simulate"
    assert json.loads(calls[0].content) == request.model_dump(mode="json")
    assert result.capacity_customers == 2
    assert result.global_summary.selected_customers == 2
    assert result.global_summary.expected_net_value == pytest.approx(20)


@pytest.mark.parametrize("mutation", [
    "cells", "labels", "counts", "scenario", "filters", "scope",
    "policy", "capacity", "unused", "cost", "net", "roi", "cell_net", "nonfinite",
])
def test_invalid_campaign_response(portfolio, mutation):
    request = simulation_request(DEFAULT_VALUES)
    payload = copy.deepcopy(response_payload(portfolio, request))
    if mutation == "cells":
        payload["displayed_segments"].pop()
    elif mutation == "labels":
        payload["displayed_segments"][1] = payload["displayed_segments"][0]
    elif mutation == "counts":
        payload["displayed_summary"]["customers"] = 100
    elif mutation == "scenario":
        payload["scenario"]["contact_cost"] = 3.
    elif mutation == "filters":
        payload["display_filters"]["partition"] = "test"
    elif mutation == "scope":
        payload["selection_scope"] = "filtered"
    elif mutation == "policy":
        payload["policy"] = "risk"
    elif mutation == "capacity":
        payload["capacity_customers"] = 99
    elif mutation == "unused":
        payload["unused_capacity"] = 10
    elif mutation == "cost":
        payload["global_summary"]["campaign_cost"] = 10.
    elif mutation == "net":
        payload["global_summary"]["expected_net_value"] = 99.
    elif mutation == "roi":
        payload["global_summary"]["scenario_roi"] = 999.
    elif mutation == "cell_net":
        payload["displayed_segments"][0]["selected_expected_net_value"] = 100.
    else:
        payload["global_summary"]["campaign_cost"] = "NaN"
    api = PortfolioClient(
        "http://test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload),
        ),
    )
    with pytest.raises(APIError, match="invalid campaign"):
        api.simulate(request)


def test_simulation_callback_and_global_filter_semantics(portfolio):
    calls = []
    app = create_app(api_for(portfolio, calls))
    client = app.server.test_client()
    result = callback(app, client, "finance-global.children", [1, *DEFAULT_VALUES])
    assert result["finance-error"]["children"] == ""
    assert len(result["finance-table"]["data"]) == 12
    assert sum(
        sum(row) for row in result["finance-matrix"]["figure"]["data"][0]["z"]
    ) == pytest.approx(20)
    figure = result["finance-matrix"]["figure"]
    assert figure["data"][0]["colorbar"]["orientation"] == "h"
    assert "<br>" in figure["layout"]["title"]["text"]
    assert "model" in result["finance-provenance"]["children"]
    filtered_values = DEFAULT_VALUES.copy()
    filtered_values[8] = "month_to_month"
    filtered = callback(
        app, client, "finance-global.children", [2, *filtered_values],
    )
    assert filtered["finance-global"] == result["finance-global"]
    assert sum(
        sum(row) for row in filtered["finance-matrix"]["figure"]["data"][0]["z"]
    ) == 0
    assert "did not reallocate" in filtered["finance-status"]["children"]
    assert len(calls) == 2


def test_edits_invalidate_without_network_request(portfolio):
    calls = []
    app = create_app(api_for(portfolio, calls))
    client = app.server.test_client()
    for field in ("contact_cost", "contract"):
        result = callback(
            app, client, "finance-global.children", [1, *DEFAULT_VALUES],
            triggered=f"finance-{field}.value",
        )
        assert result["finance-global"]["children"] == []
        assert result["finance-displayed"]["children"] == []
        assert result["finance-table"]["data"] == []
        assert result["finance-provenance"]["children"] == ""
        assert "Controls changed" in result["finance-status"]["children"]
    assert calls == []


def test_invalid_form_never_calls_api(portfolio):
    calls = []
    app = create_app(api_for(portfolio, calls))
    values = DEFAULT_VALUES.copy()
    values[1] = 101
    result = callback(
        app, app.server.test_client(), "finance-global.children", [1, *values],
    )
    assert result["finance-error"]["children"]
    assert result["finance-global"]["children"] == []
    assert calls == []


def test_error_and_retry(portfolio, caplog):
    fail = [True]
    def handler(request):
        if fail[0]:
            return httpx.Response(503)
        body = SimulationRequest.model_validate_json(request.content)
        return httpx.Response(200, json=response_payload(portfolio, body))
    api = PortfolioClient("http://test", transport=httpx.MockTransport(handler))
    app = create_app(api)
    client = app.server.test_client()
    error = callback(app, client, "finance-global.children", [1, *DEFAULT_VALUES])
    assert error["finance-global"]["children"] == []
    assert "HTTP 503" in str(error["finance-error"]["children"])
    assert "Campaign simulation request failed" in caplog.text
    fail[0] = False
    recovered = callback(app, client, "finance-global.children", [2, *DEFAULT_VALUES])
    assert recovered["finance-error"]["children"] == ""
    assert len(recovered["finance-table"]["data"]) == 12


@pytest.mark.parametrize("case", ["empty", "no_capacity", "free"])
def test_empty_zero_capacity_and_free_contact(portfolio, case):
    calls = []
    app = create_app(api_for(portfolio, calls))
    values = DEFAULT_VALUES.copy()
    if case == "empty":
        values[5] = "test"
    elif case == "no_capacity":
        values[4] = 0
    else:
        values[0] = 0
    result = callback(
        app, app.server.test_client(), "finance-global.children", [1, *values],
    )
    assert result["finance-error"]["children"] == ""
    if case == "empty":
        assert "No customers match" in result["finance-status"]["children"]
        assert all(
            row["mean_expected_net_value"] == "Not defined"
            for row in result["finance-table"]["data"]
        )
    else:
        assert "Not defined" in str(result["finance-global"]["children"])


def test_financial_tab_does_not_run_on_open(portfolio):
    calls = []
    app = create_app(api_for(portfolio, calls))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_data(as_text=True)
    assert "Financial Simulator" in layout
    assert "Run simulation" in layout
    switched = callback(app, client, "portfolio-panel.style", ["financial"])
    assert switched["financial-panel"]["style"] == {"display": "block"}
    assert switched["scoring-panel"]["style"] == {"display": "none"}
    assert switched["portfolio-panel"]["style"] == {"display": "none"}
    assert calls == []
