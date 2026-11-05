import hashlib
import json
import sys
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

import src.models.train as train
from src.config import DEFAULT_CONFIG_PATH, TrainingConfig


@pytest.fixture
def config(tmp_path, monkeypatch):
    contents = yaml.safe_load(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    )
    for settings in contents["models"].values():
        settings["enabled"] = True
    contents["optimization"]["trials_per_model"] = 2
    contents["optimization"]["timeout_seconds_per_model"] = 60
    contents["mlflow"]["tracking_directory"] = str(tmp_path / "tracking")
    contents["mlflow"]["log_trial_models"] = True
    for name in ("split_manifest", "feature_sql"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        contents["data"][name] = str(path)
    raw = tmp_path / "raw.csv"
    raw.write_text("synthetic raw data", encoding="utf-8")
    monkeypatch.setattr(train, "DATA_PATH", raw)
    return TrainingConfig.model_validate(contents)


@pytest.fixture(autouse=True)
def isolate_mlflow(tmp_path):
    if mlflow.active_run() is not None:
        pytest.fail("Run these tests without an active MLflow run.")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(
        f"sqlite:///{(tmp_path / 'outside.db').as_posix()}"
    )
    try:
        yield
    finally:
        while mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        mlflow.set_tracking_uri(previous_uri)


@pytest.fixture
def lightweight_objectives(monkeypatch):
    data = pd.DataFrame({
        "customer_id": ["a", "b"],
        "churn": [0, 1],
        "partition": ["train", "train"],
    })
    calls = []
    loads = []

    def load(config):
        loads.append(config)
        return data

    def make(model_name, config, training_data, experiment_id):
        assert training_data is data
        calls.append((model_name, config.model_copy(deep=True)))

        def objective(trial):
            trial.suggest_int("example_parameter", 1, 3)
            score = (
                0.1 * (train.MODEL_NAMES.index(model_name) + 1)
                + trial.number * 0.01
            )
            with mlflow.start_run(
                experiment_id=experiment_id, nested=True
            ) as child:
                trial.set_user_attr("mlflow_run_id", child.info.run_id)
                mlflow.log_metric("synthetic_score", score)
            return score

        return objective

    monkeypatch.setattr(train, "load_training_data", load)
    monkeypatch.setattr(train, "make_objective", make)
    return calls, loads


def tracking_client(config):
    directory = Path(config.mlflow.tracking_directory)
    directory.mkdir(parents=True, exist_ok=True)
    return MlflowClient(
        tracking_uri=f"sqlite:///{(directory / 'mlflow.db').as_posix()}"
    )


def read_artifact(client, run_id, filename):
    path = client.download_artifacts(run_id, filename)
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("smoke", [True, False])
@pytest.mark.parametrize("metric,direction", [
    ("average_precision", "maximize"),
    ("log_loss", "minimize"),
])
def test_studies_logging_budgets_and_sorting(
    config, lightweight_objectives, smoke, metric, direction
):
    config.optimization.metric = metric
    config.optimization.direction = direction
    original_config = config.model_dump()
    previous_uri = mlflow.get_tracking_uri()

    summary = train.run_training(config, smoke=smoke)

    assert config.model_dump() == original_config
    assert mlflow.get_tracking_uri() == previous_uri
    assert mlflow.active_run() is None
    assert set(summary["model"]) == set(train.MODEL_NAMES)
    assert summary["study_name"].nunique() == len(train.MODEL_NAMES)
    assert summary["metric"].eq(metric).all()
    assert summary["completed_trials"].eq(1 if smoke else 2).all()
    assert summary["best_cv_score"].tolist() == sorted(
        summary["best_cv_score"], reverse=direction == "maximize"
    )

    calls, loads = lightweight_objectives
    assert len(loads) == 1
    assert [name for name, _ in calls] == list(train.MODEL_NAMES)
    for _, effective_config in calls:
        assert effective_config.mlflow.log_trial_models is (not smoke)
        assert effective_config.optimization.trials_per_model == (
            1 if smoke else 2
        )

    client = tracking_client(config)
    directory = Path(config.mlflow.tracking_directory)
    storage = f"sqlite:///{(directory / 'optuna.db').as_posix()}"
    for row in summary.itertuples():
        study = optuna.load_study(
            study_name=row.study_name, storage=storage
        )
        assert study.direction.name.lower() == direction
        assert len(study.trials) == (1 if smoke else 2)
        assert study.best_value == pytest.approx(row.best_cv_score)
        assert study.best_trial.number == row.best_trial_number
        assert study.best_params == row.best_parameters
        assert study.user_attrs["mlflow_parent_run_id"] == row.parent_run_id
        assert row.best_trial_number == (
            0 if smoke or direction == "minimize" else 1
        )

        parent = client.get_run(row.parent_run_id)
        child = client.get_run(row.best_trial_run_id)
        assert parent.info.status == child.info.status == "FINISHED"
        assert child.data.tags["mlflow.parentRunId"] == row.parent_run_id
        assert parent.data.tags["mode"] == ("smoke" if smoke else "tuning")
        assert parent.data.metrics["best_cv_score"] == row.best_cv_score
        assert read_artifact(
            client, row.parent_run_id, "best_trial.json"
        )["best_parameters"] == row.best_parameters

        resolved = read_artifact(
            client, row.parent_run_id, "resolved_config.json"
        )
        assert resolved["optimization"]["trials_per_model"] == (
            1 if smoke else 2
        )
        provenance = read_artifact(
            client, row.parent_run_id, "input_provenance.json"
        )
        assert set(provenance) == {
            "raw_data", "split_manifest", "feature_sql"
        }
        for item in provenance.values():
            path = Path(item["path"])
            assert path.is_absolute()
            assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_disabled_models_and_fresh_studies(
    config, lightweight_objectives
):
    for name in train.MODEL_NAMES:
        getattr(config.models, name).enabled = name == "hist_gradient_boosting"
    first = train.run_training(config, smoke=True)
    second = train.run_training(config, smoke=True)
    assert first["model"].tolist() == ["hist_gradient_boosting"]
    assert second["model"].tolist() == ["hist_gradient_boosting"]
    assert first.iloc[0]["study_name"] != second.iloc[0]["study_name"]
    client = tracking_client(config)
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    for summary in (first, second):
        run = client.get_run(summary.iloc[0]["parent_run_id"])
        assert run.info.experiment_id == experiment.experiment_id


def test_failure_stops_runner_and_restores_tracking(
    config, lightweight_objectives, monkeypatch
):
    def fail(model_name, config, data, experiment_id):
        def objective(trial):
            raise RuntimeError("Synthetic objective failure")
        return objective

    monkeypatch.setattr(train, "make_objective", fail)
    previous_uri = mlflow.get_tracking_uri()
    with pytest.raises(RuntimeError, match="Synthetic objective failure"):
        train.run_training(config, smoke=True)
    assert mlflow.get_tracking_uri() == previous_uri
    assert mlflow.active_run() is None
    client = tracking_client(config)
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
    directory = Path(config.mlflow.tracking_directory)
    studies = optuna.get_all_study_summaries(
        storage=f"sqlite:///{(directory / 'optuna.db').as_posix()}"
    )
    assert len(studies) == 1
    study = optuna.load_study(
        study_name=studies[0].study_name,
        storage=f"sqlite:///{(directory / 'optuna.db').as_posix()}",
    )
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_active_run_is_rejected(config, lightweight_objectives):
    mlflow.set_experiment("outside")
    with mlflow.start_run() as parent:
        with pytest.raises(RuntimeError, match="active MLflow run"):
            train.run_training(config, smoke=True)
        assert mlflow.active_run().info.run_id == parent.info.run_id
    assert lightweight_objectives[1] == []


@pytest.mark.parametrize("deleted", [False, True])
def test_incompatible_experiment_is_rejected(
    config, lightweight_objectives, deleted
):
    client = tracking_client(config)
    experiment_id = client.create_experiment(
        config.mlflow.experiment_name,
        artifact_location=(
            Path(config.mlflow.tracking_directory) / "different"
        ).as_uri(),
    )
    if deleted:
        client.delete_experiment(experiment_id)
    previous_uri = mlflow.get_tracking_uri()
    with pytest.raises(
        RuntimeError,
        match="deleted" if deleted else "different artifact location",
    ):
        train.run_training(config, smoke=True)
    assert mlflow.get_tracking_uri() == previous_uri
    assert lightweight_objectives[1] == []


def test_data_loader_selects_only_sorted_training_rows(config, monkeypatch):
    calls = []

    def raw(connection):
        calls.append("raw")

    def splits(connection, path):
        assert path == Path(config.data.split_manifest)
        calls.append("splits")

    def features(connection, path):
        assert path == Path(config.data.feature_sql)
        calls.append("features")
        connection.execute("""
            CREATE TEMP TABLE baseline_features AS
            SELECT * FROM (VALUES
                ('z', 1, 'train'),
                ('a', 0, 'train'),
                ('v', 1, 'validation'),
                ('t', 0, 'test')
            ) AS customers(customer_id, churn, partition)
        """)

    monkeypatch.setattr(train, "load_customers", raw)
    monkeypatch.setattr(train, "load_split_views", splits)
    monkeypatch.setattr(train, "load_feature_view", features)
    data = train.load_training_data(config)
    assert calls == ["raw", "splits", "features"]
    assert data["customer_id"].tolist() == ["a", "z"]
    assert data["partition"].eq("train").all()


def test_file_hash(tmp_path):
    path = tmp_path / "input.bin"
    content = b"abc" * 400_000
    path.write_bytes(content)
    assert train.file_sha256(path) == hashlib.sha256(content).hexdigest()
    with pytest.raises(FileNotFoundError):
        train.file_sha256(tmp_path / "missing")


@pytest.mark.parametrize("smoke", [True, False])
def test_cli(config, monkeypatch, capsys, smoke):
    argv = ["train", "--config", "custom.yaml"]
    if smoke:
        argv.append("--smoke")
    monkeypatch.setattr(sys, "argv", argv)

    def load(path):
        assert path == Path("custom.yaml")
        return config

    def run(actual_config, *, smoke=False):
        assert actual_config is config
        assert smoke == ("--smoke" in argv)
        return pd.DataFrame([{
            "model": "hist_gradient_boosting",
            "completed_trials": 1,
            "metric": "average_precision",
            "best_cv_score": 0.2,
            "best_trial_number": 0,
            "best_trial_run_id": "example-run",
        }])

    monkeypatch.setattr(train, "load_training_config", load)
    monkeypatch.setattr(train, "run_training", run)
    train.main()
    assert "example-run" in capsys.readouterr().out
