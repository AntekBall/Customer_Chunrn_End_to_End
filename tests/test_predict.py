from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from threadpoolctl import threadpool_limits

from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.data.loader import NULLABLE_COLUMNS
from src.models.model_builder import build_model_pipeline
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
)
from src.models.predict import ChurnPredictor, ModelProvenance, validate_customers


@pytest.fixture
def customers():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        column: rng.integers(1, 10, 30).astype(float)
        for column in NUMERIC_COLUMNS
    })
    for column in CATEGORICAL_COLUMNS:
        frame[column] = rng.choice(["A", "B"], 30)
    for column in BINARY_COLUMNS:
        frame[column] = rng.integers(0, 2, 30)
    frame["paperless_billing"] = frame["paperless_billing"].astype(bool)
    frame["customer_id"] = [f"customer-{i}" for i in range(30)]
    return frame


@pytest.fixture
def provenance():
    return ModelProvenance("synthetic-model", "source", "freeze", "test")


@pytest.fixture
def model(customers):
    config = TrainingConfig.model_validate(
        yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    )
    model = build_model_pipeline("catboost", config)
    model.set_params(classifier__iterations=2, classifier__depth=2)
    with threadpool_limits(limits=1):
        model.fit(
            customers[list(FEATURE_COLUMNS)], np.tile([0, 1], 15)
        )
    return model


def test_batch_single_missing_and_nonmutation(customers, model, provenance):
    data = customers.iloc[:3].copy()
    data.index = [10, 10, 20]
    for column in NULLABLE_COLUMNS:
        data[column] = None
    data.loc[:, "contract"] = "previously-unseen"
    original = data.copy(deep=True)
    predictor = ChurnPredictor(model, provenance)
    with threadpool_limits(limits=1):
        batch = predictor.predict_batch(data)
        one = predictor.predict_one(data.iloc[0].to_dict())
    assert batch.index.tolist() == [10, 10, 20]
    assert batch["customer_id"].tolist() == data["customer_id"].tolist()
    assert batch["churn_probability"].between(0, 1).all()
    assert one["churn_probability"] == pytest.approx(batch.iloc[0]["churn_probability"])
    assert one["model_run_id"] == "source"
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("case", [
    "empty", "missing", "extra", "duplicate_column", "duplicate_id", "blank_id",
    "null_category", "blank_category", "numeric_category", "null_required",
    "negative", "infinity", "string_number", "bool_number", "fractional_count",
    "nonbinary", "null_binary",
])
def test_invalid_input(customers, case):
    data = customers.copy()
    if case == "empty":
        data = data.iloc[:0]
    elif case == "missing":
        data = data.drop(columns="age")
    elif case == "extra":
        data["churn"] = 0
    elif case == "duplicate_column":
        data = pd.concat([data, data[["age"]]], axis=1)
    elif case == "duplicate_id":
        data["customer_id"] = "same"
    elif case == "blank_id":
        data["customer_id"] = " "
    elif case == "null_category":
        data["contract"] = None
    elif case == "blank_category":
        data["contract"] = " "
    elif case == "numeric_category":
        data["contract"] = 1
    elif case == "null_required":
        data["age"] = np.nan
    elif case == "negative":
        data["monthlycharges"] = -1
    elif case == "infinity":
        data["annual_income"] = np.inf
    elif case == "string_number":
        data["age"] = "35"
    elif case == "bool_number":
        data["age"] = True
    elif case == "fractional_count":
        data["tenure"] = 1.5
    elif case == "nonbinary":
        data["has_phone_service"] = 2
    else:
        data["paperless_billing"] = None
    with pytest.raises(ValueError):
        validate_customers(data)


@pytest.mark.parametrize("kind", ["nan", "range", "sum", "shape"])
def test_invalid_model_output(customers, provenance, kind):
    class BrokenModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            if kind == "nan":
                return np.full((len(frame), 2), np.nan)
            if kind == "range":
                return np.tile([-0.1, 1.1], (len(frame), 1))
            if kind == "sum":
                return np.full((len(frame), 2), 0.1)
            return np.zeros(len(frame))

    with pytest.raises(ValueError, match="invalid class probabilities"):
        ChurnPredictor(BrokenModel(), provenance).predict_batch(customers)


def test_positive_class_order(customers, provenance):
    class ReversedModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, frame):
            return np.tile([0.3, 0.7], (len(frame), 1))

    result = ChurnPredictor(ReversedModel(), provenance).predict_batch(customers)
    assert result["churn_probability"].eq(0.3).all()


def test_frozen_loader_roundtrip(tmp_path, customers, model):
    if mlflow.active_run() is not None:
        pytest.fail("Run without an active MLflow run.")
    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    try:
        experiment_id = mlflow.create_experiment(
            "inference-tests", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        with mlflow.start_run(experiment_id=experiment_id) as source:
            mlflow.set_tags({
                "model": "catboost", "evaluation_partition": "validation",
            })
            mlflow.log_param("positive_class_weight", 1.0)
            logged = mlflow.sklearn.log_model(
                model, name="pipeline",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            source_id = source.info.run_id
        manifest = {
            "model_uri": logged.model_uri,
            "source_finalist_run_id": source_id,
            "calibration": "none",
        }
        with mlflow.start_run(experiment_id=experiment_id) as frozen:
            mlflow.set_tag("mode", "frozen_model_policy")
            mlflow.log_dict(manifest, "frozen_manifest.json")
            freeze_id = frozen.info.run_id
        with mlflow.start_run(experiment_id=experiment_id) as final:
            mlflow.set_tags({
                "mode": "final_test", "evaluation_partition": "test",
                "freeze_run_id": freeze_id, "source_finalist_run_id": source_id,
            })
            mlflow.log_dict(manifest, "frozen_manifest.json")
            final_id = final.info.run_id

        mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'different.db').as_posix()}")
        expected_uri = mlflow.get_tracking_uri()
        predictor = ChurnPredictor.from_final_test(
            tmp_path, final_test_run_id=final_id
        )
        assert mlflow.get_tracking_uri() == expected_uri
        assert mlflow.active_run() is None
        with threadpool_limits(limits=1):
            actual = predictor.predict_batch(customers)
            expected = model.predict_proba(customers[list(FEATURE_COLUMNS)])[:, 1]
        np.testing.assert_allclose(actual["churn_probability"], expected)
        assert actual["model_run_id"].eq(source_id).all()
        assert predictor.provenance.model_uri == logged.model_uri
        with pytest.raises(ValueError, match="completed frozen final-test"):
            ChurnPredictor.from_final_test(tmp_path, final_test_run_id=source_id)
        client = MlflowClient(tracking_uri=uri)
        client.log_dict(final_id, manifest | {"calibration": "changed"}, "frozen_manifest.json")
        with pytest.raises(ValueError, match="differs"):
            ChurnPredictor.from_final_test(tmp_path, final_test_run_id=final_id)
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_missing_tracking_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChurnPredictor.from_final_test(tmp_path / "missing", final_test_run_id="none")
