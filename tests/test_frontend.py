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


def callback(app, client, key_fragment, inputs, states=None, *, triggered=None):
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
        "changedPropIds": [
            triggered or metadata["inputs"][0]["id"] + "." + metadata["inputs"][0]["property"]
        ],
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
    assert figure["colorbar"]["orientation"] == "h"
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
    assert reset["contract"]["value"] is None


def test_categorical_dropdowns_share_exact_source_values():
    from src.frontend.categories import CATEGORY_OPTIONS
    from src.frontend.scoring import DEMO_CUSTOMER
    from src.models.pipeline import CATEGORICAL_COLUMNS

    app = create_app(api_client(lambda request: httpx.Response(500)))
    layout = app.server.test_client().get("/_dash-layout").get_json()
    components = {}

    def collect(node):
        if isinstance(node, dict):
            props = node.get("props", {})
            if "id" in props:
                components[props["id"]] = node
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for child in node:
                collect(child)

    collect(layout)
    assert set(CATEGORY_OPTIONS) == set(CATEGORICAL_COLUMNS)
    for column in CATEGORICAL_COLUMNS:
        node = components[f"score-{column}"]
        assert node["type"] == "Dropdown"
        assert node["props"]["options"] == CATEGORY_OPTIONS[column]
        values = [option["value"] for option in CATEGORY_OPTIONS[column]]
        assert len(values) == len(set(values))
        assert DEMO_CUSTOMER[column] in values
        assert node["props"]["clearable"] is True
    for column in ("contract", "payment_method"):
        component_id = "payment" if column == "payment_method" else column
        assert components[component_id]["type"] == "Dropdown"
        assert components[component_id]["props"]["options"] == CATEGORY_OPTIONS[column]
        assert components[component_id]["props"]["placeholder"] == "All"
    assert CATEGORY_OPTIONS["contract"][0] == {
        "label": "Month-to-month", "value": "month_to_month",
    }


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


def explanation_payload(customer):
    import math

    from src.models.pipeline import FEATURE_COLUMNS

    return {
        "customer_id": customer["customer_id"],
        "model_run_id": "model", "freeze_run_id": "freeze", "final_test_run_id": "final",
        "churn_probability": 1 / (1 + math.exp(1.2)),
        "base_log_odds": -1.5, "raw_log_odds": -1.2,
        "output_scale": "log_odds", "method": "native_catboost_tree_shap",
        "interpretation": "Model contributions, not causal effects.",
        "contributions": [
            {"feature": feature, "feature_value": customer[feature],
             "shap_log_odds": .3 if feature == "age" else 0.}
            for feature in FEATURE_COLUMNS
        ],
    }


def test_scoring_payload_and_api_schema_compatibility():
    from src.api.main import Customer
    from src.frontend.scoring import DEMO_CUSTOMER, FORM_COLUMNS, customer_payload

    customer = customer_payload([DEMO_CUSTOMER[column] for column in FORM_COLUMNS])
    assert Customer.model_validate(customer).model_dump() == DEMO_CUSTOMER
    assert len(customer) == 29
    assert customer["credit_score"] is None
    assert type(customer["age"]) is int
    assert type(customer["monthlycharges"]) is float
    integral = DEMO_CUSTOMER | {"age": 43.0}
    assert customer_payload([integral[column] for column in FORM_COLUMNS])["age"] == 43


@pytest.mark.parametrize("change", [
    {"age": 2.5}, {"age": None}, {"age": "43"}, {"age": True},
    {"contract": " "}, {"has_phone_service": 2}, {"has_phone_service": None},
    {"monthlycharges": -1}, {"credit_score": float("inf")}, {"customer_id": ""},
])
def test_invalid_scoring_form(change):
    from src.frontend.scoring import DEMO_CUSTOMER, FORM_COLUMNS, customer_payload

    customer = DEMO_CUSTOMER | change
    with pytest.raises(ValueError):
        customer_payload([customer[column] for column in FORM_COLUMNS])


def test_client_explanation_posts_exact_inputs():
    import json

    from src.frontend.scoring import DEMO_CUSTOMER

    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=explanation_payload(DEMO_CUSTOMER))
    result = api_client(handler).explain(DEMO_CUSTOMER)
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/explain"
    assert json.loads(seen[0].content) == DEMO_CUSTOMER
    assert len(result.contributions) == 28


@pytest.mark.parametrize("mutation", [
    "customer", "input_value", "raw", "probability", "nan", "scale",
    "missing_feature", "duplicate_feature", "json",
])
def test_invalid_explanation_response(mutation):
    from src.frontend.scoring import DEMO_CUSTOMER

    bad = explanation_payload(DEMO_CUSTOMER)
    if mutation == "customer":
        bad["customer_id"] = "someone-else"
    elif mutation == "input_value":
        bad["contributions"][0]["feature_value"] = 9999.
    elif mutation == "raw":
        bad["raw_log_odds"] = 0.
    elif mutation == "probability":
        bad["churn_probability"] = .9
    elif mutation == "nan":
        bad["raw_log_odds"] = "NaN"
    elif mutation == "scale":
        bad["output_scale"] = "probability"
    elif mutation == "missing_feature":
        bad["contributions"].pop()
    elif mutation == "duplicate_feature":
        bad["contributions"][-1] = bad["contributions"][-2]
    with pytest.raises(APIError):
        api_client(lambda request: (
            httpx.Response(200, text="invalid json") if mutation == "json"
            else httpx.Response(200, json=bad)
        )).explain(DEMO_CUSTOMER)


def test_customer_scoring_callback_and_edit_invalidation():
    from src.frontend.scoring import DEMO_CUSTOMER, FORM_COLUMNS

    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=explanation_payload(DEMO_CUSTOMER))
    app = create_app(api_client(handler))
    client = app.server.test_client()
    demo = callback(app, client, "score-customer_id.value", [1])
    assert demo["score-credit_score"]["value"] is None
    values = [demo[f"score-{column}"]["value"] for column in FORM_COLUMNS]
    result = callback(app, client, "score-card.children", [1, *values])
    assert result["score-status"]["children"] == "Profile scored."
    assert len(result["score-contributions"]["data"]) == 28
    assert "model" in result["score-provenance"]["children"]
    trace = result["score-waterfall"]["figure"]["data"][0]
    assert trace["orientation"] == "h"
    assert sum(trace["x"][:-1]) == pytest.approx(-1.2)
    assert any("Other 18" in label for label in trace["y"])
    assert result["score-contributions"]["data"][-1]["shap_log_odds"] == "+0.000000"
    changed = values.copy()
    changed[FORM_COLUMNS.index("age")] = 50
    invalidated = callback(
        app, client, "score-card.children", [1, *changed], triggered="score-age.value",
    )
    assert invalidated["score-card"]["children"] == []
    assert invalidated["score-contributions"]["data"] == []
    assert invalidated["score-provenance"]["children"] == ""
    assert "Profile changed" in invalidated["score-status"]["children"]
    assert len(calls) == 1


def test_scoring_failure_clears_previous_result_and_recovers(caplog):
    from src.frontend.scoring import DEMO_CUSTOMER, FORM_COLUMNS

    fail = [True]
    def handler(request):
        return httpx.Response(503) if fail[0] else httpx.Response(
            200, json=explanation_payload(DEMO_CUSTOMER),
        )
    app = create_app(api_client(handler))
    client = app.server.test_client()
    values = [DEMO_CUSTOMER[column] for column in FORM_COLUMNS]
    result = callback(app, client, "score-card.children", [1, *values])
    assert result["score-card"]["children"] == []
    assert result["score-contributions"]["data"] == []
    assert "HTTP 503" in str(result["score-error"]["children"])
    assert "Customer explanation request failed" in caplog.text
    fail[0] = False
    result = callback(app, client, "score-card.children", [2, *values])
    assert result["score-error"]["children"] == ""
    assert result["score-status"]["children"] == "Profile scored."


def test_empty_customer_form_never_calls_api():
    from src.frontend.scoring import FORM_COLUMNS

    calls = []
    app = create_app(api_client(lambda request: calls.append(request) or httpx.Response(500)))
    result = callback(
        app, app.server.test_client(), "score-card.children",
        [1, *[None for column in FORM_COLUMNS]],
    )
    assert result["score-card"]["children"] == []
    assert result["score-error"]["children"]
    assert calls == []


def test_tabs_keep_both_panels_mounted_without_api_requests():
    calls = []
    app = create_app(api_client(lambda request: calls.append(request) or httpx.Response(500)))
    client = app.server.test_client()
    layout = client.get("/_dash-layout").get_data(as_text=True)
    assert "Customer Scoring" in layout
    assert "Score & explain" in layout
    assert "Portfolio overview" in layout
    result = callback(app, client, "portfolio-panel.style", ["scoring"])
    assert result["portfolio-panel"]["style"] == {"display": "none"}
    assert result["scoring-panel"]["style"] == {"display": "block"}
    assert calls == []


def test_scoring_client_integrates_with_real_api_and_synthetic_model():
    import duckdb
    import numpy as np
    from fastapi.testclient import TestClient

    from src.api.main import Services, create_app as create_api
    from tests.helpers import model_config
    from src.frontend.scoring import DEMO_CUSTOMER, FORM_COLUMNS, customer_payload
    from src.models.explain import ChurnExplainer
    from src.models.model_builder import build_model_pipeline
    from src.models.pipeline import FEATURE_COLUMNS
    from src.models.predict import ChurnPredictor, ModelProvenance

    customer = customer_payload([DEMO_CUSTOMER[column] for column in FORM_COLUMNS])
    frame = pd.DataFrame([customer] * 8)
    frame["age"] = np.arange(20, 28)
    model = build_model_pipeline("catboost", model_config())
    model.set_params(
        classifier__iterations=2, classifier__depth=2, classifier__thread_count=1,
    )
    model.fit(frame[list(FEATURE_COLUMNS)], np.tile([0, 1], 4))
    predictor = ChurnPredictor(model, ModelProvenance("model", "model", "freeze", "final"))
    api = create_api()
    with duckdb.connect() as connection:
        api.state.services = Services(predictor, ChurnExplainer(predictor), connection, 0)
        backend = TestClient(api)
        try:
            def handler(request):
                response = backend.post("/explain", content=request.content, headers={
                    "content-type": "application/json",
                })
                return httpx.Response(response.status_code, content=response.content)
            result = api_client(handler).explain(customer)
            assert result.churn_probability == pytest.approx(
                predictor.predict_one(customer)["churn_probability"],
            )
            assert len(result.contributions) == 28
            assert next(
                row for row in result.contributions if row.feature == "credit_score"
            ).feature_value is None
        finally:
            backend.close()
