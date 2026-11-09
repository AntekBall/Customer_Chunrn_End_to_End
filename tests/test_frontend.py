import copy

import httpx
import pandas as pd
import pytest

from src.data.dashboard import PortfolioFilters
from src.data.portfolio import (
    apply_portfolio_segments, fit_portfolio_thresholds, summarize_portfolio_segments,
)
from src.frontend.app import create_app
from src.frontend.client import APIError, PortfolioClient


@pytest.fixture
def payload():
    frame = pd.DataFrame({
        "customer_id": ["a", "b", "c"],
        "monthlycharges": [0., 20., 100.],
        "churn_probability": [.1, .2, .5],
        "partition": ["train"] * 3,
    })
    cuts = fit_portfolio_thresholds(frame, model_run_id="model")
    frame = apply_portfolio_segments(frame, cuts, model_run_id="model")
    segments = summarize_portfolio_segments(frame).reset_index()
    return {
        "model_run_id": "model", "freeze_run_id": "freeze", "final_test_run_id": "final",
        "filters": {}, "interpretation": "Descriptive estimates, not measured losses.",
        "overview": {
            "customers": 3, "monthly_charge_total": 120., "mean_monthly_charge": 40.,
            "mean_predicted_churn": .8 / 3, "risk_weighted_monthly_charges": 54.,
            "customer_share": 1., "monthly_charge_share": 1.,
        },
        "segments": segments.astype(object).where(segments.notna(), None).to_dict("records"),
    }


def api_client(handler):
    return PortfolioClient("http://test", transport=httpx.MockTransport(handler))


def callback(app, client, key_fragment, inputs, states=None):
    keys = [key for key in app.callback_map if key_fragment in key]
    assert len(keys) == 1
    key = keys[0]
    metadata = app.callback_map[key]
    outputs = metadata["output"]
    response = client.post("/_dash-update-component", json={
        "output": key,
        "outputs": [{"id": output.component_id, "property": output.component_property}
                    for output in outputs],
        "inputs": [
            {"id": item["id"], "property": item["property"], "value": value}
            for item, value in zip(metadata["inputs"], inputs, strict=True)
        ],
        "state": [
            {"id": item["id"], "property": item["property"], "value": value}
            for item, value in zip(metadata["state"], states or [], strict=True)
        ],
        "changedPropIds": [metadata["inputs"][0]["id"] + "." + metadata["inputs"][0]["property"]],
    })
    assert response.status_code == 200, response.text
    return response.get_json()["response"]


def test_http_client_query_encoding_and_validation(payload):
    requests = []
    filters = PortfolioFilters(contract="' OR 1=1 --", has_internet_service=False)
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload | {"filters": filters.model_dump()})
    summary = api_client(handler).summary(filters)
    assert summary.overview.customers == 3
    assert requests[0].url.params["contract"] == "' OR 1=1 --"
    assert requests[0].url.params["has_internet_service"] == "false"
    assert requests[0].url.path == "/portfolio/summary"


@pytest.mark.parametrize("kind", ["status", "connect", "timeout", "json", "shape", "filters"])
def test_api_failures_are_explicit(payload, kind):
    def handler(request):
        if kind == "status":
            return httpx.Response(503, text="internal details should not be displayed")
        if kind == "connect":
            raise httpx.ConnectError("unavailable", request=request)
        if kind == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if kind == "json":
            return httpx.Response(200, text="not JSON")
        if kind == "shape":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json=payload | {"filters": {"partition": "test"}})
    with pytest.raises(APIError) as error:
        api_client(handler).summary(PortfolioFilters())
    assert "internal details" not in str(error.value)


@pytest.mark.parametrize("mutation", ["cells", "duplicates", "count", "charge", "nonfinite"])
def test_inconsistent_responses_rejected(payload, mutation):
    bad = copy.deepcopy(payload)
    if mutation == "cells":
        bad["segments"].pop()
    elif mutation == "duplicates":
        bad["segments"][1] = bad["segments"][0]
    elif mutation == "count":
        bad["overview"]["customers"] = 100
    elif mutation == "charge":
        bad["overview"]["monthly_charge_total"] = 9999.
    else:
        bad["overview"]["monthly_charge_total"] = "NaN"
    with pytest.raises(APIError, match="invalid summary"):
        api_client(lambda request: httpx.Response(200, json=bad)).summary(PortfolioFilters())


@pytest.mark.parametrize("url", ["file:///tmp", "ftp://test", "http://", "http://test?q=1"])
def test_invalid_api_url(url):
    with pytest.raises(ValueError):
        PortfolioClient(url)


def test_layout_assets_and_initial_load(payload):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)
    app = create_app(api_client(handler))
    client = app.server.test_client()
    assert calls == []
    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout").get_data(as_text=True)
    assert "Portfolio overview" in layout
    assert "Synthetic data" in layout
    assert client.get("/assets/portfolio.css").status_code == 200
    loaded = callback(app, client, "summary-data.data", [0], [None] * 6)
    assert loaded["summary-data"]["data"]["overview"]["customers"] == 3
    rendered = callback(
        app, client, "kpis.children", [loaded["summary-data"]["data"], "customers"],
    )
    assert len(rendered["kpis"]["children"]) == 4
    assert len(rendered["segment-table"]["data"]) == 12
    assert "Model: model" in rendered["provenance"]["children"]
    figure = rendered["segment-heatmap"]["figure"]["data"][0]
    assert len(figure["z"]) == 4
    assert all(len(row) == 3 for row in figure["z"])
    assert sum(sum(row) for row in figure["z"]) == 3
    assert len(calls) == 1


def test_apply_filters_and_metric_changes_do_not_refetch(payload):
    calls = []
    filters = PortfolioFilters(
        partition="train", revenue_segment="Lower half", risk_segment="Lower risk",
        has_internet_service=False, contract="two_year", payment_method="card",
    )
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload | {"filters": filters.model_dump()})
    app = create_app(api_client(handler))
    client = app.server.test_client()
    loaded = callback(
        app, client, "summary-data.data", [1],
        ["train", "Lower half", "Lower risk", "no", "  two_year ", "card "],
    )["summary-data"]["data"]
    assert calls[0].url.params["contract"] == "two_year"
    assert calls[0].url.params["has_internet_service"] == "false"
    for metric in [
        "customers", "monthly_charge_total", "mean_predicted_churn",
        "risk_weighted_monthly_charges",
    ]:
        callback(app, client, "kpis.children", [loaded, metric])
    assert len(calls) == 1
    reset = callback(app, client, "partition.value", [1])
    assert reset["partition"]["value"] is None
    assert reset["contract"]["value"] == ""


def test_error_clears_old_results_and_retry_succeeds(payload, caplog):
    fail = [True]
    def handler(request):
        return httpx.Response(503) if fail[0] else httpx.Response(200, json=payload)
    app = create_app(api_client(handler))
    client = app.server.test_client()
    result = callback(app, client, "summary-data.data", [1], [None] * 6)
    assert result["summary-data"]["data"] is None
    assert "HTTP 503" in str(result["error-message"]["children"])
    assert "Portfolio request failed" in caplog.text
    rendered = callback(app, client, "kpis.children", [None, "customers"])
    assert rendered["kpis"]["children"] == []
    assert rendered["segment-table"]["data"] == []
    assert rendered["provenance"]["children"] == ""
    fail[0] = False
    result = callback(app, client, "summary-data.data", [2], [None] * 6)
    assert result["summary-data"]["data"]["overview"]["customers"] == 3
    assert result["error-message"]["children"] == ""


def test_empty_results_render_as_undefined(payload):
    payload["overview"] = {
        "customers": 0, "monthly_charge_total": 0., "mean_monthly_charge": None,
        "mean_predicted_churn": None, "risk_weighted_monthly_charges": 0.,
        "customer_share": None, "monthly_charge_share": None,
    }
    for row in payload["segments"]:
        row.update(payload["overview"])
    app = create_app(api_client(lambda request: httpx.Response(200, json=payload)))
    client = app.server.test_client()
    loaded = callback(app, client, "summary-data.data", [0], [None] * 6)
    rendered = callback(
        app, client, "kpis.children",
        [loaded["summary-data"]["data"], "mean_predicted_churn"],
    )
    assert "No customers match" in rendered["snapshot-message"]["children"]
    assert rendered["segment-table"]["data"][0]["mean_predicted_churn"] == "Not defined"
    assert all(
        value is None
        for row in rendered["segment-heatmap"]["figure"]["data"][0]["z"] for value in row
    )


def test_invalid_filter_does_not_call_api(payload):
    calls = []
    app = create_app(api_client(
        lambda request: calls.append(request) or httpx.Response(200, json=payload),
    ))
    result = callback(
        app, app.server.test_client(), "summary-data.data", [1],
        ["invalid", None, None, None, None, None],
    )
    assert result["summary-data"]["data"] is None
    assert "Invalid filters" in str(result["error-message"]["children"])
    assert calls == []
