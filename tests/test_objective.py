import json

import mlflow
import numpy as np
import optuna
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import src.models.objective as objective_module
from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.models.pipeline import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
)


PARAMETERS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}


@pytest.fixture
def config():
    contents = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    contents["models"]["hist_gradient_boosting"]["enabled"] = True
    contents["cross_validation"] = {"n_splits": 3, "shuffle": True}
    contents["evaluation"]["metrics"] = [
        "average_precision", "roc_auc", "log_loss", "brier_score"
    ]
    contents["evaluation"]["targeting_fractions"] = [0.1, 0.2, 1.0]
    contents["mlflow"]["log_trial_models"] = False
    return TrainingConfig.model_validate(contents)


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    data = pd.DataFrame({
        column: rng.uniform(1, 10, 90) for column in NUMERIC_COLUMNS
    })
    for column in CATEGORICAL_COLUMNS:
        data[column] = rng.choice(["A", "B"], 90)
    for column in BINARY_COLUMNS:
        data[column] = rng.integers(0, 2, 90)
    data["customer_id"] = [f"customer-{i:03d}" for i in range(90)]
    data["churn"] = np.tile([0, 1], 45)
    data["partition"] = "train"
    return data


@pytest.fixture(scope="module")
def tracking(tmp_path_factory):
    if mlflow.active_run() is not None:
        pytest.fail("Run objective tests without an active MLflow run.")
    directory = tmp_path_factory.mktemp("objective-mlflow")
    previous_uri = mlflow.get_tracking_uri()
    database = (directory / "tracking.db").as_posix()
    uri = f"sqlite:///{database}"
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        "objective-tests",
        artifact_location=(directory / "artifacts").as_uri(),
    )
    try:
        yield experiment_id, MlflowClient(tracking_uri=uri)
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def cheap_trials(monkeypatch):
    monkeypatch.setattr(
        objective_module, "suggest_parameters",
        lambda trial, model_name: PARAMETERS.copy(),
    )
    with threadpool_limits(limits=1):
        yield


def run_trial(config, training_data, tracking):
    experiment_id, client = tracking
    objective = objective_module.make_objective(
        "hist_gradient_boosting", config, training_data, experiment_id
    )
    study = optuna.create_study(direction=config.optimization.direction)
    study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    return trial, client.get_run(trial.user_attrs["mlflow_run_id"])


@pytest.mark.parametrize("metric", [
    "average_precision", "roc_auc", "log_loss", "brier_score"
])
def test_metrics_artifacts_and_no_leakage(
    config, training_data, tracking, cheap_trials, monkeypatch, metric
):
    config.optimization.metric = metric
    config.optimization.direction = (
        "maximize" if metric in {"average_precision", "roc_auc"} else "minimize"
    )
    fit_indices = []
    score_indices = []
    models = []
    original_fit = Pipeline.fit
    original_predict = Pipeline.predict_proba

    def record_fit(self, features, target, **kwargs):
        models.append(self)
        fit_indices.append(features.index.to_numpy())
        assert tuple(features.columns) == FEATURE_COLUMNS
        return original_fit(self, features, target, **kwargs)

    def record_predict(self, features, **kwargs):
        score_indices.append(features.index.to_numpy())
        return original_predict(self, features, **kwargs)

    def forbid_model_logging(*args, **kwargs):
        pytest.fail("Trial models must not be logged when disabled.")

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    monkeypatch.setattr(Pipeline, "predict_proba", record_predict)
    monkeypatch.setattr(mlflow.sklearn, "log_model", forbid_model_logging)
    experiment_id, client = tracking
    with mlflow.start_run(experiment_id=experiment_id) as parent:
        trial, run = run_trial(config, training_data, tracking)
        assert mlflow.active_run().info.run_id == parent.info.run_id
        assert run.data.tags["mlflow.parentRunId"] == parent.info.run_id

    assert trial.state == optuna.trial.TrialState.COMPLETE
    assert run.info.status == "FINISHED"
    assert trial.value == pytest.approx(run.data.metrics[f"mean_{metric}"])
    assert trial.value > 0
    assert run.data.tags["trial_state"] == "COMPLETE"
    assert run.data.params["training_rows"] == "90"
    assert run.data.params["max_iter"] == "100"
    assert len(models) == len({id(model) for model in models}) == 3

    expected_splits = list(StratifiedKFold(
        n_splits=3, shuffle=True, random_state=config.random_seed
    ).split(training_data, training_data["churn"]))
    np.testing.assert_array_equal(
        np.sort(np.concatenate(score_indices)), np.arange(90)
    )

    expected_rows = []
    for (fit, score), actual_fit, actual_score in zip(
        expected_splits, fit_indices, score_indices, strict=True
    ):
        np.testing.assert_array_equal(fit, actual_fit)
        np.testing.assert_array_equal(score, actual_score)
        assert set(fit).isdisjoint(score)
        target = training_data.iloc[score]["churn"]
        # Each fit fold has 60 rows: min_samples_leaf=50 prevents any split.
        probabilities = np.full(len(score), training_data.iloc[fit]["churn"].mean())
        row = {
            "average_precision": average_precision_score(target, probabilities),
            "roc_auc": roc_auc_score(target, probabilities),
            "log_loss": log_loss(target, probabilities),
            "brier_score": brier_score_loss(target, probabilities),
        }
        ranked = training_data.iloc[score].sort_values("customer_id")
        for fraction in config.evaluation.targeting_fractions:
            count = int(np.ceil(fraction * len(score)))
            captured = ranked.head(count)["churn"].sum()
            suffix = f"top_{100 * fraction:g}pct"
            row[f"precision_{suffix}"] = captured / count
            row[f"capture_{suffix}"] = captured / target.sum()
        expected_rows.append(row)

    expected = pd.DataFrame(expected_rows)
    for name in expected.columns:
        assert run.data.metrics[f"mean_{name}"] == pytest.approx(expected[name].mean())
        assert run.data.metrics[f"std_{name}"] == pytest.approx(expected[name].std(ddof=1))
        history = client.get_metric_history(run.info.run_id, f"fold_{name}")
        assert [entry.step for entry in history] == [0, 1, 2]
        assert [entry.value for entry in history] == pytest.approx(expected[name].tolist())

    for name in ("fit_seconds", "predict_seconds"):
        assert run.data.metrics[f"mean_{name}"] >= 0

    for filename in ("resolved_config.json", "feature_schema.json", "fold_metrics.json"):
        path = client.download_artifacts(run.info.run_id, filename)
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        if filename == "feature_schema.json":
            assert payload["predictors"] == list(FEATURE_COLUMNS)
        elif filename == "fold_metrics.json":
            assert len(payload) == 3
        else:
            assert payload == config.model_dump(mode="json")


@pytest.mark.parametrize("case,message", [
    ("validation", "training-only"),
    ("test", "training-only"),
    ("empty", "training-only"),
    ("missing", "feature schema"),
    ("extra", "feature schema"),
    ("duplicate_column", "feature schema"),
    ("duplicate_id", "non-null and unique"),
    ("null_id", "non-null and unique"),
    ("one_class", "both binary classes"),
    ("invalid_label", "both binary classes"),
    ("null_label", "both binary classes"),
    ("rare_class", "one customer per fold"),
])
def test_invalid_training_data(config, training_data, case, message):
    data = training_data.copy()
    if case in {"validation", "test"}:
        data.loc[0, "partition"] = case
    elif case == "empty":
        data = data.iloc[:0]
    elif case == "missing":
        data = data.drop(columns=[FEATURE_COLUMNS[0]])
    elif case == "extra":
        data["unexpected"] = 0
    elif case == "duplicate_column":
        data = pd.concat([data, data[["churn"]]], axis=1)
    elif case == "duplicate_id":
        data.loc[0, "customer_id"] = data.loc[1, "customer_id"]
    elif case == "null_id":
        data.loc[0, "customer_id"] = None
    elif case == "one_class":
        data["churn"] = 0
    elif case == "invalid_label":
        data.loc[0, "churn"] = 2
    elif case == "null_label":
        data["churn"] = data["churn"].astype(float)
        data.loc[0, "churn"] = np.nan
    elif case == "rare_class":
        data["churn"] = 0
        data.loc[0, "churn"] = 1
    with pytest.raises(ValueError, match=message):
        objective_module.make_objective(
            "hist_gradient_boosting", config, data, "unused"
        )


def test_unknown_and_disabled_models(config, training_data):
    with pytest.raises(ValueError, match="Unknown model"):
        objective_module.make_objective("unknown", config, training_data, "unused")
    config.models.hist_gradient_boosting.enabled = False
    with pytest.raises(ValueError, match="disabled"):
        objective_module.make_objective(
            "hist_gradient_boosting", config, training_data, "unused"
        )


@pytest.mark.parametrize("failure", ["fit", "probabilities"])
def test_failed_trials_are_logged_and_raised(
    config, training_data, tracking, cheap_trials, monkeypatch, failure
):
    if failure == "fit":
        def fail_fit(*args, **kwargs):
            raise RuntimeError("Synthetic fit failure")
        monkeypatch.setattr(Pipeline, "fit", fail_fit)
        error, message = RuntimeError, "Synthetic fit failure"
    else:
        monkeypatch.setattr(
            Pipeline, "predict_proba",
            lambda self, features: np.full((len(features), 2), np.nan),
        )
        error, message = ValueError, "invalid probabilities"
    experiment_id, client = tracking
    objective = objective_module.make_objective(
        "hist_gradient_boosting", config, training_data, experiment_id
    )
    study = optuna.create_study()
    with pytest.raises(error, match=message):
        study.optimize(objective, n_trials=1)
    trial = study.trials[0]
    run = client.get_run(trial.user_attrs["mlflow_run_id"])
    assert trial.state == optuna.trial.TrialState.FAIL
    assert run.info.status == "FAILED"
    assert run.data.tags["trial_state"] == "FAIL"
    path = client.download_artifacts(run.info.run_id, "error.txt")
    with open(path, encoding="utf-8") as file:
        assert message in file.read()
    assert mlflow.active_run() is None
