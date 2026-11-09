import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import experiments.finalist_validation as module
from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.models.evaluate import score_predictions, validate_partitions
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
)


@pytest.fixture
def config(tmp_path):
    contents = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    contents["mlflow"]["tracking_directory"] = str(tmp_path)
    contents["mlflow"]["log_finalist_models"] = True
    return TrainingConfig.model_validate(contents)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        name: rng.uniform(1, 10, 90) for name in NUMERIC_COLUMNS
    })
    for name in CATEGORICAL_COLUMNS:
        frame[name] = rng.choice(["A", "B"], 90)
    for name in BINARY_COLUMNS:
        frame[name] = rng.integers(0, 2, 90)
    frame["customer_id"] = [f"id-{i:03d}" for i in range(90)]
    frame["churn"] = np.tile([0, 1], 45)
    frame["partition"] = "train"
    training = frame.iloc[:60].copy()
    validation = frame.iloc[60:].copy()
    validation["partition"] = "validation"
    return training, validation


@pytest.mark.parametrize("case", ["test", "overlap", "schema", "labels"])
def test_partition_guards(data, case):
    training, validation = data
    if case == "test":
        validation["partition"] = "test"
    elif case == "overlap":
        validation.loc[60, "customer_id"] = training.loc[0, "customer_id"]
    elif case == "schema":
        training["unexpected"] = 1
    else:
        validation["churn"] = 0
    with pytest.raises(ValueError):
        validate_partitions(training, validation)


@pytest.mark.parametrize("probabilities", [
    np.full(30, np.nan), np.full(30, 1.1), np.zeros((30, 2)),
])
def test_probability_guards(config, data, probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        score_predictions(data[1], probabilities, config)


def test_metrics(config, data):
    validation = data[1]
    result = score_predictions(validation, validation["churn"].to_numpy(), config)
    assert result["average_precision"] == 1
    assert result["roc_auc"] == 1
    assert result["brier_score"] == 0
    assert result["precision_top_10pct"] == 1
    assert result["capture_top_10pct"] == pytest.approx(3 / 15)


def test_refit_logging_and_model_roundtrip(config, data, monkeypatch, tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    if mlflow.active_run() is not None:
        pytest.fail("Run without an active MLflow run.")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        experiment_id = mlflow.create_experiment(
            config.mlflow.experiment_name,
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
        provenance = {"raw_data": {"path": "synthetic", "sha256": "fixture"}}
        sources = {}
        for name, parameters in {
            "catboost": {"iterations": 2, "depth": 2},
            "xgboost": {"n_estimators": 2, "max_depth": 2},
        }.items():
            with mlflow.start_run(experiment_id=experiment_id) as parent:
                mlflow.log_dict(provenance, "input_provenance.json")
                mlflow.log_dict(config.model_dump(mode="json"), "resolved_config.json")
                with mlflow.start_run(experiment_id=experiment_id, nested=True) as child:
                    mlflow.set_tag("model", name)
                    sources[name] = child.info.run_id
                mlflow.log_dict({
                    "best_trial_run_id": sources[name],
                    "best_parameters": parameters,
                }, "best_trial.json")
        monkeypatch.setattr(module, "SOURCE_RUNS", sources)
        monkeypatch.setattr(module, "current_provenance", lambda config: provenance)
        monkeypatch.setattr(module, "load_data", lambda config: data)
        original_fit = Pipeline.fit
        fitted_ids = []
        models = []

        def fit(self, features, target, **kwargs):
            fitted_ids.append(features.index.tolist())
            models.append(self)
            assert kwargs == {}
            return original_fit(self, features, target, **kwargs)

        monkeypatch.setattr(Pipeline, "fit", fit)
        expected_uri = mlflow.get_tracking_uri()
        with threadpool_limits(limits=1):
            summary = module.run_validation(config)
        assert mlflow.get_tracking_uri() == expected_uri
        assert mlflow.active_run() is None
        assert set(summary["model"]) == {"catboost", "xgboost", "histogram_baseline"}
        assert fitted_ids == [data[0].index.tolist()] * 3
        client = MlflowClient()
        for row in summary.itertuples():
            run = client.get_run(row.run_id)
            assert run.info.status == "FINISHED"
            assert run.data.tags["mlflow.parentRunId"] == row.parent_run_id
            assert run.data.metrics["validation_average_precision"] == row.average_precision
            assert row.model_uri
            reloaded = mlflow.sklearn.load_model(row.model_uri)
            original = models[["catboost", "xgboost", "histogram_baseline"].index(row.model)]
            features = data[1].loc[:, list(module.FEATURE_COLUMNS)]
            with threadpool_limits(limits=1):
                np.testing.assert_allclose(
                    reloaded.predict_proba(features), original.predict_proba(features)
                )
        baseline = models[-1].named_steps["classifier"]
        assert baseline.max_iter == 150
        assert baseline.random_state == 42
    finally:
        mlflow.set_tracking_uri(previous_uri)
