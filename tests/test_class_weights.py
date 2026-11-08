import mlflow
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import src.models.objective as objective_module
from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.models.model_builder import MODEL_NAMES
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
)


@pytest.fixture
def config():
    contents = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for model in contents["models"].values():
        model["enabled"] = True
    contents["mlflow"]["log_trial_models"] = False
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
    frame["churn"] = [1] * 20 + [0] * 70
    frame["partition"] = "train"
    return frame


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True])
def test_invalid_weights(config, data, weight):
    with pytest.raises(ValueError, match="positive_class_weight"):
        objective_module.make_objective(
            "catboost", config, data, "unused",
            positive_class_weight=weight,
        )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("weight", [1.0, 3.0])
def test_weights_reach_classifier(
    config, data, tmp_path, monkeypatch, model_name, weight
):
    if mlflow.active_run() is not None:
        pytest.fail("Run tests without an active MLflow run.")
    parameters = {
        "hist_gradient_boosting": {"max_iter": 2, "min_samples_leaf": 5},
        "lightgbm": {"n_estimators": 2, "min_child_samples": 5},
        "xgboost": {"n_estimators": 2, "max_depth": 2},
        "random_forest": {"n_estimators": 2, "min_samples_leaf": 5},
        "catboost": {"iterations": 2, "depth": 2},
    }
    monkeypatch.setattr(
        objective_module, "suggest_parameters",
        lambda trial, name: parameters[name],
    )
    original_fit = Pipeline.fit
    calls = []

    def fit(self, features, target, **kwargs):
        calls.append(features.index.to_numpy())
        if weight == 1.0:
            assert "classifier__sample_weight" not in kwargs
        else:
            actual = kwargs["classifier__sample_weight"]
            expected = np.where(target.to_numpy() == 1, weight, 1.0)
            expected /= expected.mean()
            np.testing.assert_allclose(actual, expected)
            assert actual.mean() == pytest.approx(1.0)
            assert actual[target.to_numpy() == 1][0] / actual[
                target.to_numpy() == 0
            ][0] == pytest.approx(weight)
        return original_fit(self, features, target, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    try:
        experiment_id = mlflow.create_experiment(
            "weights", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        trial = optuna.trial.FixedTrial({})
        with threadpool_limits(limits=1):
            score = objective_module.make_objective(
                model_name, config, data, experiment_id,
                positive_class_weight=weight,
            )(trial)
        assert len(calls) == config.cross_validation.n_splits
        assert np.isfinite(score)
        run = mlflow.tracking.MlflowClient().get_run(
            trial.user_attrs["mlflow_run_id"]
        )
        assert run.info.status == "FINISHED"
        assert float(run.data.params["positive_class_weight"]) == weight
    finally:
        mlflow.set_tracking_uri(previous_uri)
