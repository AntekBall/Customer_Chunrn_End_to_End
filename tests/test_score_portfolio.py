import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.data import score_portfolio as scoring
from src.data.portfolio import (
    PortfolioThresholds, apply_portfolio_segments, summarize_portfolio_segments,
)
from src.models.artifacts import file_sha256
from src.models.pipeline import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.models.predict import ChurnPredictor, ModelProvenance


class SimpleModel:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.batch_lengths = []

    def predict_proba(self, frame):
        self.batch_lengths.append(len(frame))
        p = frame["age"].to_numpy(dtype=float) / 100
        return np.column_stack([1 - p, p])


@pytest.fixture
def customers():
    frame = pd.DataFrame({column: np.ones(6) for column in NUMERIC_COLUMNS})
    for column in CATEGORICAL_COLUMNS:
        frame[column] = "A"
    for column in BINARY_COLUMNS:
        frame[column] = 1
    frame["paperless_billing"] = True
    frame["customer_id"] = [f"customer-{i}" for i in range(6)]
    frame["partition"] = ["train"] * 4 + ["validation", "test"]
    frame["age"] = [1, 2, 3, 4, 80, 90]
    frame["monthlycharges"] = [10., 20., 30., 40., 1000., 2000.]
    frame.index = [9, 9, 7, 6, 5, 4]
    return frame


@pytest.fixture
def predictor():
    return ChurnPredictor(
        SimpleModel(), ModelProvenance("model", "source", "freeze", "final"),
    )


@pytest.fixture
def config():
    return TrainingConfig.model_validate(
        yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    )


@pytest.fixture
def fingerprints(monkeypatch):
    inputs = {"synthetic": {"path": "fixture", "sha256": "abc"}}
    monkeypatch.setattr(scoring, "current_provenance", lambda config: inputs.copy())
    return inputs


def test_training_only_thresholds_and_batch_contract(customers, predictor):
    before = customers.copy(deep=True)
    snapshot, thresholds = scoring.build_snapshot(customers, predictor, batch_size=2)
    assert predictor._model.batch_lengths == [2, 2, 2]
    assert thresholds.reference_rows == 4
    np.testing.assert_allclose(thresholds.revenue_thresholds, [25., 37., 39.7])
    np.testing.assert_allclose(thresholds.risk_thresholds, [.025, .037])
    np.testing.assert_allclose(snapshot["churn_probability"], [.01, .02, .03, .04, .8, .9])
    assert snapshot.index.tolist() == customers.index.tolist()
    assert snapshot["customer_id"].tolist() == customers["customer_id"].tolist()
    assert snapshot["model_run_id"].eq("source").all()
    assert snapshot["freeze_run_id"].eq("freeze").all()
    assert snapshot["final_test_run_id"].eq("final").all()
    assert not {"churn", "eligible", "selected"}.intersection(snapshot.columns)
    pd.testing.assert_frame_equal(customers, before)


@pytest.mark.parametrize("batch_size", [1, 3, 99])
def test_batch_size_invariance(customers, predictor, batch_size):
    expected, cuts = scoring.build_snapshot(customers, predictor, batch_size=2)
    actual, other = scoring.build_snapshot(customers, predictor, batch_size=batch_size)
    pd.testing.assert_frame_equal(actual, expected)
    assert cuts == other


@pytest.mark.parametrize("case", [
    "empty", "no_train", "unknown_partition", "missing_partition",
    "duplicate_id", "label", "missing_feature", "bad_feature",
    "duplicate_column", "zero_batch", "negative_batch", "bool_batch", "float_batch",
])
def test_invalid_snapshot_inputs(customers, predictor, case):
    batch_size = 2
    if case == "empty":
        customers = customers.iloc[:0]
    elif case == "no_train":
        customers["partition"] = "validation"
    elif case == "unknown_partition":
        customers["partition"] = "holdout"
    elif case == "missing_partition":
        customers["partition"] = None
    elif case == "duplicate_id":
        customers.iloc[-1, customers.columns.get_loc("customer_id")] = "customer-0"
    elif case == "label":
        customers["churn"] = 0
    elif case == "missing_feature":
        customers = customers.drop(columns="age")
    elif case == "bad_feature":
        customers["age"] = -1
    elif case == "duplicate_column":
        customers = pd.concat([customers, customers[["age"]]], axis=1)
    else:
        batch_size = {
            "zero_batch": 0, "negative_batch": -1, "bool_batch": True, "float_batch": 2.5,
        }[case]
    with pytest.raises(ValueError):
        scoring.build_snapshot(customers, predictor, batch_size=batch_size)


def test_reordered_predictor_output_is_rejected(customers, predictor, monkeypatch):
    original = predictor.predict_batch
    monkeypatch.setattr(predictor, "predict_batch", lambda frame: original(frame).iloc[::-1])
    with pytest.raises(ValueError, match="order or coverage"):
        scoring.build_snapshot(customers, predictor, batch_size=2)


def test_snapshot_roundtrip(tmp_path, customers, predictor, config, fingerprints):
    output = tmp_path / "snapshot"
    assert scoring.save_snapshot(
        customers, predictor, output, input_provenance=fingerprints, config=config,
        batch_size=2,
    ) == output
    with duckdb.connect() as connection:
        restored = connection.read_parquet(str(output / "customers.parquet")).df()
    thresholds = PortfolioThresholds.model_validate_json(
        (output / "thresholds.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["partition_rows"] == {"train": 4, "validation": 1, "test": 1}
    assert manifest["model"] == {
        "model_uri": "model", "source_finalist_run_id": "source",
        "freeze_run_id": "freeze", "final_test_run_id": "final", "calibration": "none",
    }
    assert manifest["input_provenance"] == fingerprints
    assert manifest["training_scores_are_in_sample"] is True
    assert manifest["contains_churn_labels"] is False
    assert manifest["contains_campaign_selection"] is False
    for name in ("customers.parquet", "thresholds.json"):
        assert manifest["files"][name]["sha256"] == file_sha256(output / name)
    assert restored["customer_id"].tolist() == customers["customer_id"].tolist()
    reapplied = apply_portfolio_segments(restored, thresholds, model_run_id="source")
    assert reapplied["revenue_segment"].astype(str).tolist() == restored["revenue_segment"].tolist()
    summary = summarize_portfolio_segments(reapplied)
    assert len(summary) == 12
    assert summary["customers"].sum() == 6
    assert summary["monthly_charge_total"].sum() == pytest.approx(3100)
    assert summary["risk_weighted_monthly_charges"].sum() == pytest.approx(2603)
    with pytest.raises(FileExistsError):
        scoring.save_snapshot(
            customers, predictor, output, input_provenance=fingerprints, config=config,
        )


@pytest.mark.parametrize("during", [False, True])
def test_source_drift_never_publishes(
    tmp_path, customers, predictor, config, fingerprints, monkeypatch, during,
):
    calls = []

    def fingerprint(config):
        calls.append(True)
        if during and len(calls) == 1:
            return fingerprints
        return {"changed": {}}

    monkeypatch.setattr(scoring, "current_provenance", fingerprint)
    with pytest.raises(ValueError, match="Source inputs changed"):
        scoring.save_snapshot(
            customers, predictor, tmp_path / "snapshot",
            input_provenance=fingerprints, config=config,
        )
    assert list(tmp_path.iterdir()) == []


def test_scoring_failure_never_publishes(
    tmp_path, customers, predictor, config, fingerprints, monkeypatch,
):
    def fail(frame):
        raise RuntimeError("model failure")

    monkeypatch.setattr(predictor, "predict_batch", fail)
    with pytest.raises(RuntimeError, match="model failure"):
        scoring.save_snapshot(
            customers, predictor, tmp_path / "snapshot",
            input_provenance=fingerprints, config=config,
        )
    assert list(tmp_path.iterdir()) == []


def test_load_inputs_excludes_label(customers, config, monkeypatch):
    source = customers.assign(churn=1)
    monkeypatch.setattr(scoring, "load_customers", lambda connection: None)
    monkeypatch.setattr(scoring, "load_split_views", lambda connection, path: None)

    def feature_view(connection, path):
        connection.register("fixture", source)
        connection.sql("CREATE VIEW baseline_features AS SELECT * FROM fixture")

    monkeypatch.setattr(scoring, "load_feature_view", feature_view)
    result = scoring.load_snapshot_inputs(config)
    assert set(result.columns) == set(customers.columns)
    assert result["customer_id"].tolist() == sorted(customers["customer_id"].tolist())
    assert "churn" not in result


def test_cli_uses_frozen_configuration_and_inputs(
    tmp_path, customers, predictor, config, fingerprints, monkeypatch,
):
    output = tmp_path / "cli-snapshot"
    monkeypatch.setattr(
        ChurnPredictor, "from_final_test",
        classmethod(lambda cls, directory, **kwargs: predictor),
    )
    monkeypatch.setattr(scoring, "MlflowClient", lambda **kwargs: object())
    monkeypatch.setattr(scoring, "read_json", lambda *args: {
        "config": config.model_dump(), "input_provenance": fingerprints,
    })
    monkeypatch.setattr(scoring, "load_snapshot_inputs", lambda frozen_config: customers)
    monkeypatch.setattr(sys, "argv", ["score_portfolio", "--output", str(output), "--batch-size", "2"])
    scoring.main()
    assert (output / "customers.parquet").is_file()
    assert predictor._model.batch_lengths == [2, 2, 2]
