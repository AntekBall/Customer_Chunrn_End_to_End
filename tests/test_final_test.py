from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

import experiments.final_test as module
from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS,
)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    if mlflow.active_run() is not None:
        pytest.fail("Run without an active MLflow run.")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        config = TrainingConfig.model_validate(
            yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        )
        config.mlflow.tracking_directory = str(tmp_path)
        provenance = {
            name: {"path": "synthetic", "sha256": value * 64}
            for name, value in (
                ("raw_data", "a"), ("split_manifest", "b"), ("feature_sql", "c")
            )
        }
        monkeypatch.setattr(module, "current_provenance", lambda config: provenance)
        frame = pd.DataFrame({
            name: np.full(20, 100.0) for name in NUMERIC_COLUMNS
        })
        for name in CATEGORICAL_COLUMNS:
            frame[name] = "A"
        for name in BINARY_COLUMNS:
            frame[name] = 0
        frame["customer_id"] = [f"id-{i:02d}" for i in range(20)]
        frame["churn"] = [0, 1] * 10
        frame["partition"] = "test"
        model = DummyClassifier(strategy="prior").fit(
            frame[list(module.FEATURE_COLUMNS)], frame["churn"]
        )
        experiment_id = mlflow.create_experiment(
            "final-test-fixture", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        with mlflow.start_run(experiment_id=experiment_id) as parent:
            mlflow.log_dict(config.model_dump(mode="json"), "evaluation_config.json")
            mlflow.log_dict(provenance, "input_provenance.json")
            with mlflow.start_run(experiment_id=experiment_id, nested=True) as source:
                mlflow.set_tags({
                    "model": "catboost", "evaluation_partition": "validation",
                })
                mlflow.log_param("positive_class_weight", 1.0)
                logged = mlflow.sklearn.log_model(
                    model, name="pipeline",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                source_id = source.info.run_id
            mlflow.log_dict([{
                "run_id": source_id, "model_uri": logged.model_uri,
            }], "validation_comparison.json")
        with mlflow.start_run(experiment_id=experiment_id) as scenario:
            mlflow.set_tags({
                "mode": "financial_scenario",
                "evaluation_partition": "validation",
                "source_finalist_run_id": source_id,
            })
            mlflow.log_dict({
                "contact_cost": 2.0, "gross_margin": 0.2, "retained_months": 6.0,
                "prevention_probability": 0.2, "capacity_fraction": 0.1,
            }, "campaign_scenario.json")
            mlflow.log_dict(module.POLICY, "policy_definition.json")
            mlflow.log_dict(provenance, "input_provenance.json")
            monkeypatch.setattr(module, "SCENARIO_RUN_ID", scenario.info.run_id)
        yield config, frame, provenance, MlflowClient(), experiment_id
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_freeze_evaluate_and_repeat_guard(setup, monkeypatch, tmp_path):
    config, frame, _, client, experiment_id = setup
    loads = []

    def load(config):
        loads.append("test loaded")
        return frame.copy()

    monkeypatch.setattr(module, "load_test", load)
    freeze_id = module.freeze(client)
    assert loads == []
    manifest = module.read_json(client, freeze_id, "frozen_manifest.json")
    assert manifest["scenario"]["contact_cost"] == 2.0
    assert manifest["calibration"] == "none"
    metrics, calibration, selected = module.evaluate(client, freeze_id, tmp_path)
    assert loads == ["test loaded"]
    assert metrics["average_precision"] == pytest.approx(0.5)
    assert calibration["predicted_minus_observed"] == 0
    assert selected.iloc[0]["selected_customers"] == 2
    assert selected.iloc[0]["expected_net_value"] == pytest.approx(20)
    with pytest.raises(RuntimeError, match="already reserved"):
        module.evaluate(client, freeze_id, tmp_path)
    assert loads == ["test loaded"]
    runs = client.search_runs(
        [experiment_id], filter_string="tags.mode = 'final_test'"
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert runs[0].data.tags["freeze_run_id"] == freeze_id
    assert client.download_artifacts(runs[0].info.run_id, "reliability.html")
    assert module.read_json(
        client, runs[0].info.run_id, "frozen_policy_scenario.json"
    )[0]["expected_net_value"] == pytest.approx(20)


def test_changed_inputs_rejected_before_loading_test(setup, monkeypatch, tmp_path):
    _, _, _, client, _ = setup
    freeze_id = module.freeze(client)
    monkeypatch.setattr(module, "current_provenance", lambda config: {})
    monkeypatch.setattr(module, "load_test", lambda config: pytest.fail("Test accessed"))
    with pytest.raises(ValueError, match="changed"):
        module.evaluate(client, freeze_id, tmp_path)
    assert not list(tmp_path.glob("final_test_*.lock"))


def test_failed_attempt_stays_reserved(setup, monkeypatch, tmp_path):
    _, _, _, client, experiment_id = setup
    freeze_id = module.freeze(client)

    def fail(config):
        raise RuntimeError("Synthetic test loading failure")

    monkeypatch.setattr(module, "load_test", fail)
    with pytest.raises(RuntimeError, match="Synthetic"):
        module.evaluate(client, freeze_id, tmp_path)
    with pytest.raises(RuntimeError, match="already reserved"):
        module.evaluate(client, freeze_id, tmp_path)
    runs = client.search_runs(
        [experiment_id], filter_string="tags.mode = 'final_test'"
    )
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"


@pytest.mark.parametrize("case", ["partition", "ids", "labels", "schema"])
def test_test_partition_validation(case):
    frame = pd.DataFrame({
        name: [1, 2] for name in module.FEATURE_COLUMNS
    })
    frame["customer_id"] = ["a", "b"]
    frame["churn"] = [0, 1]
    frame["partition"] = "test"
    if case == "partition":
        frame["partition"] = "validation"
    elif case == "ids":
        frame["customer_id"] = "a"
    elif case == "labels":
        frame["churn"] = 0
    else:
        frame["extra"] = 0
    with pytest.raises(ValueError):
        module.validate_test(frame)
