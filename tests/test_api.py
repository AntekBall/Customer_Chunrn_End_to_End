import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.config import load_training_config
from src.data import score_portfolio
from src.models.model_builder import build_model_pipeline
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
)
from src.models.predict import ChurnPredictor, ModelProvenance


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        column: rng.integers(1, 10, 40) for column in NUMERIC_COLUMNS
    })
    for column in CATEGORICAL_COLUMNS:
        frame[column] = rng.choice(["A", "B"], len(frame))
    for column in BINARY_COLUMNS:
        frame[column] = rng.integers(0, 2, len(frame))
    frame["paperless_billing"] = frame["paperless_billing"].astype(bool)
    for column in ("annual_income", "customer_satisfaction", "num_complaints",
                   "avg_monthly_gb", "credit_score"):
        frame[column] = None
    frame["customer_id"] = [f"customer-{i}" for i in range(len(frame))]
    model = build_model_pipeline("catboost", load_training_config())
    model.set_params(
        classifier__iterations=4, classifier__depth=2, classifier__thread_count=1,
    )
    model.fit(frame[list(FEATURE_COLUMNS)], np.tile([0, 1], 20))
    return frame, model


@pytest.fixture
def setup(tmp_path, monkeypatch, fitted):
    frame, model = fitted
    predictor = ChurnPredictor(
        model, ModelProvenance("model", "source", "freeze", "final"),
    )
    inputs = {"synthetic": {"path": "fixture", "sha256": "abc"}}
    monkeypatch.setattr(score_portfolio, "current_provenance", lambda config: inputs)
    output = score_portfolio.save_snapshot(
        frame.assign(partition=["train"] * 30 + ["validation"] * 5 + ["test"] * 5),
        predictor, tmp_path / "snapshot", input_provenance=inputs,
        config=load_training_config(),
    )
    loads = []
    def load(cls, directory, *, final_test_run_id):
        loads.append((directory, final_test_run_id))
        return predictor
    monkeypatch.setattr(ChurnPredictor, "from_final_test", classmethod(load))
    app = main.create_app(
        tracking_directory=tmp_path / "tracking",
        portfolio_directory=output, final_test_run_id="final",
    )
    return app, frame.iloc[0].to_dict(), predictor, loads, output


def test_health_lifecycle_load_once_and_prediction(setup):
    app, customer, predictor, loads, _ = setup
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ready", "portfolio_customers": 40, "model_run_id": "source",
            "freeze_run_id": "freeze", "final_test_run_id": "final",
        }
        for _ in range(2):
            response = client.post("/predict", json=customer)
            assert response.status_code == 200
            assert response.json() == predictor.predict_one(customer)
        assert len(loads) == 1
        assert client.get("/openapi.json").status_code == 200
    assert app.state.services is None


def test_batch_order_and_nullable_values(setup):
    app, customer, _, _, _ = setup
    second = customer | {"customer_id": "another", "contract": "unseen"}
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"customers": [second, customer]})
        assert response.status_code == 200
        rows = response.json()["predictions"]
        assert [row["customer_id"] for row in rows] == ["another", customer["customer_id"]]
        assert all(0 <= row["churn_probability"] <= 1 for row in rows)


@pytest.mark.parametrize("change", [
    {"age": -1}, {"age": 1.5}, {"age": "43"}, {"age": True},
    {"monthlycharges": True}, {"monthlycharges": "10"},
    {"contract": " "}, {"contract": None}, {"churn": 0},
    {"has_phone_service": 2}, {"has_phone_service": "1"},
    {"has_phone_service": 1.0}, {"customer_id": ""},
])
def test_request_validation(setup, change):
    app, customer, _, _, _ = setup
    with TestClient(app) as client:
        for endpoint in ["/predict", "/explain"]:
            response = client.post(endpoint, json=customer | change)
            assert response.status_code == 422, response.text
            assert "input" not in response.json()["detail"][0]


def test_required_nullable_field_and_nonfinite(setup):
    app, customer, _, _, _ = setup
    with TestClient(app) as client:
        missing = customer.copy()
        del missing["credit_score"]
        assert client.post("/predict", json=missing).status_code == 422
        # Send raw non-standard JSON to exercise validation error serialization.
        bad = json.dumps(customer | {"monthlycharges": float("nan")})
        response = client.post("/predict", content=bad, headers={"content-type": "application/json"})
        assert response.status_code == 422
        assert "input" not in response.json()["detail"][0]


@pytest.mark.parametrize("case", ["empty", "large", "duplicate", "extra"])
def test_batch_limits(setup, case):
    app, customer, _, _, _ = setup
    if case == "empty":
        payload = {"customers": []}
    elif case == "large":
        payload = {"customers": [customer | {"customer_id": str(i)} for i in range(1001)]}
    elif case == "duplicate":
        payload = {"customers": [customer, customer]}
    else:
        payload = {"customers": [customer], "extra": True}
    with TestClient(app) as client:
        assert client.post("/predict/batch", json=payload).status_code == 422


def test_batch_accepts_exact_limit(setup, monkeypatch):
    app, customer, predictor, _, _ = setup
    sizes = []
    def batch(frame):
        sizes.append(len(frame))
        return frame[["customer_id"]].assign(
            churn_probability=.1, **main._provenance(predictor),
        )
    monkeypatch.setattr(predictor, "predict_batch", batch)
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={
            "customers": [customer | {"customer_id": str(i)} for i in range(1000)]
        })
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1000
    assert sizes == [1000]


def test_explanation_json_and_reconstruction(setup):
    app, customer, _, _, _ = setup
    with TestClient(app) as client:
        response = client.post("/explain", json=customer)
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result["contributions"]) == 28
        contributions = {row["feature"]: row for row in result["contributions"]}
        assert contributions["credit_score"]["feature_value"] is None
        assert result["base_log_odds"] + sum(
            row["shap_log_odds"] for row in result["contributions"]
        ) == pytest.approx(result["raw_log_odds"])
        assert 1 / (1 + np.exp(-result["raw_log_odds"])) == pytest.approx(
            result["churn_probability"]
        )
        assert result["output_scale"] == "log_odds"
        assert result["model_run_id"] == "source"


@pytest.mark.parametrize("query,expected", [
    ("", 40), ("?partition=train", 30), ("?partition=test", 5),
    ("?contract=absent", 0), ("?contract=%27%20OR%201%3D1%20--", 0),
])
def test_summary_filters_and_empty_json(setup, query, expected):
    app, _, _, _, _ = setup
    with TestClient(app) as client:
        response = client.get("/portfolio/summary" + query)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["overview"]["customers"] == expected
        assert len(result["segments"]) == 12
        assert sum(row["customers"] for row in result["segments"]) == expected
        if expected == 0:
            assert result["overview"]["mean_predicted_churn"] is None
            assert result["overview"]["customer_share"] is None
            assert all(row["mean_monthly_charge"] is None for row in result["segments"])


@pytest.mark.parametrize("value", ["true", "false"])
def test_boolean_query_filter(setup, value):
    app, _, _, _, _ = setup
    with TestClient(app) as client:
        response = client.get("/portfolio/summary", params={"has_internet_service": value})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["filters"]["has_internet_service"] is (value == "true")
        assert 0 < result["overview"]["customers"] < 40


@pytest.mark.parametrize("query", [
    "?partition=other", "?has_internet_service=invalid", "?contract=",
    "?risk_segment=invalid", "?unrecognized=true",
])
def test_invalid_query_filters(setup, query):
    app, _, _, _, _ = setup
    with TestClient(app) as client:
        assert client.get("/portfolio/summary" + query).status_code == 422


def test_model_failures_are_logged_and_return_500(setup, monkeypatch, caplog):
    app, customer, predictor, _, _ = setup
    def fail(customer):
        raise ValueError("Synthetic internal model failure")
    monkeypatch.setattr(predictor, "predict_one", fail)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=customer)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal service error."}
    assert "API operation failed" in caplog.text
    assert "Synthetic internal model failure" not in response.text


def test_startup_rejects_mismatched_model_and_closes_portfolio(setup, monkeypatch):
    app, _, _, _, _ = setup
    original = main.open_portfolio
    closed = []
    @contextmanager
    def changed(path):
        with original(path) as connection:
            connection.sql("UPDATE portfolio SET freeze_run_id = 'different'")
            try:
                yield connection
            finally:
                closed.append(True)
    monkeypatch.setattr(main, "open_portfolio", changed)
    with pytest.raises(ValueError, match="provenance differ"):
        with TestClient(app):
            pytest.fail("Startup must fail.")
    assert closed == [True]


def test_startup_rejects_corrupt_snapshot(setup):
    app, _, _, _, output = setup
    with (output / "thresholds.json").open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        with TestClient(app):
            pytest.fail("Startup must fail.")


def test_health_without_startup_is_unavailable(setup):
    app, _, _, loads, _ = setup
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    assert loads == []


def test_concurrent_summary_and_prediction(setup):
    app, customer, _, loads, _ = setup
    with TestClient(app) as client:
        def request(index):
            if index % 2:
                return client.post("/predict", json=customer)
            return client.get("/portfolio/summary?partition=train")
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(request, range(12)))
        assert all(response.status_code == 200 for response in responses)
    assert len(loads) == 1
