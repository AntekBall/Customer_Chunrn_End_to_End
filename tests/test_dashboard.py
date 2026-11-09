import json
import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.data import dashboard
from src.data.portfolio import (
    apply_portfolio_segments, fit_portfolio_thresholds, summarize_portfolio_segments,
)
from src.models.artifacts import file_sha256
from src.models.pipeline import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


@pytest.fixture
def snapshot(tmp_path):
    frame = pd.DataFrame({name: np.ones(6) for name in NUMERIC_COLUMNS})
    for name in CATEGORICAL_COLUMNS:
        frame[name] = "A"
    for name in BINARY_COLUMNS:
        frame[name] = 1
    frame["customer_id"] = [f"c{i}" for i in range(6)]
    frame["partition"] = ["train"] * 4 + ["validation", "test"]
    frame["contract"] = ["A", "B", "A", "B", "A", "B"]
    frame["has_internet_service"] = [1, 0, 1, 0, 1, 0]
    frame["monthlycharges"] = [0., 0., 10., 20., 100., 200.]
    frame["churn_probability"] = [.01, .02, .03, .04, .8, .9]
    frame["model_run_id"] = "source"
    frame["freeze_run_id"] = "freeze"
    frame["final_test_run_id"] = "final"
    cuts = fit_portfolio_thresholds(frame.iloc[:4], model_run_id="source")
    frame = apply_portfolio_segments(frame, cuts, model_run_id="source")
    with duckdb.connect() as connection:
        connection.register("fixture", frame)
        connection.table("fixture").write_parquet(str(tmp_path / "customers.parquet"))
    (tmp_path / "thresholds.json").write_text(cuts.model_dump_json(), encoding="utf-8")
    manifest = {
        "schema_version": 1, "purpose": "descriptive_portfolio_not_model_evaluation",
        "training_scores_are_in_sample": True, "includes_source_test_customers": True,
        "contains_churn_labels": False, "contains_campaign_selection": False,
        "threshold_reference": "train", "rows": 6,
        "partition_rows": {"train": 4, "validation": 1, "test": 1},
        "model": {
            "source_finalist_run_id": "source", "freeze_run_id": "freeze",
            "final_test_run_id": "final", "calibration": "none",
        },
        "files": {
            name: {"sha256": file_sha256(tmp_path / name)}
            for name in ("customers.parquet", "thresholds.json")
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, frame


def summary(connection, filters=None):
    return dashboard.summarize_portfolio(connection, filters)


def test_totals_and_exact_pandas_equivalence(snapshot):
    directory, frame = snapshot
    with dashboard.open_portfolio(directory) as connection:
        result = summary(connection)
    overview = result.overview.iloc[0]
    assert overview["customers"] == 6
    assert overview["monthly_charge_total"] == pytest.approx(330)
    assert overview["risk_weighted_monthly_charges"] == pytest.approx(261.1)
    assert overview["mean_predicted_churn"] == pytest.approx(frame["churn_probability"].mean())
    assert overview["customer_share"] == 1
    expected = summarize_portfolio_segments(frame)
    pd.testing.assert_frame_equal(result.segments, expected, check_dtype=False)


@pytest.mark.parametrize("values", [
    {"partition": "train"}, {"partition": "test"}, {"contract": "A"},
    {"payment_method": "A"}, {"has_internet_service": True},
    {"has_internet_service": False},
    {"revenue_segment": "Lower half"}, {"risk_segment": "Highest risk"},
    {"contract": "A", "partition": "train", "has_internet_service": True},
    {"contract": "B", "risk_segment": "Lower risk"},
])
def test_filter_reconciliation(snapshot, values):
    directory, frame = snapshot
    expected = frame
    for column, value in values.items():
        expected = expected.loc[expected[column].eq(value)]
    with dashboard.open_portfolio(directory) as connection:
        result = summary(connection, dashboard.PortfolioFilters(**values))
        full = summary(connection)
    pd.testing.assert_frame_equal(
        result.segments, summarize_portfolio_segments(expected), check_dtype=False,
    )
    assert result.overview.iloc[0]["customers"] == len(expected)
    assert result.segments["customers"].sum() == len(expected)
    assert full.overview.iloc[0]["customers"] == 6


@pytest.mark.parametrize("contract", ["absent", "' OR 1=1 --"])
def test_empty_and_parameterized_filter(snapshot, contract):
    directory, _ = snapshot
    with dashboard.open_portfolio(directory) as connection:
        result = summary(connection, dashboard.PortfolioFilters(contract=contract))
    assert len(result.segments) == 12
    assert result.segments["customers"].eq(0).all()
    assert result.segments["monthly_charge_total"].eq(0).all()
    assert result.segments["customer_share"].isna().all()
    assert result.segments["mean_predicted_churn"].isna().all()
    assert result.overview.iloc[0]["customers"] == 0
    assert np.isnan(result.overview.iloc[0]["mean_monthly_charge"])


def test_zero_charge_denominator(snapshot):
    directory, _ = snapshot
    with dashboard.open_portfolio(directory) as connection:
        result = summary(
            connection, dashboard.PortfolioFilters(revenue_segment="Lower half"),
        )
    assert result.overview.iloc[0]["customers"] == 2
    assert result.overview.iloc[0]["monthly_charge_total"] == 0
    assert np.isnan(result.overview.iloc[0]["monthly_charge_share"])
    assert result.segments["monthly_charge_share"].isna().all()
    assert result.segments["customer_share"].sum() == 1


@pytest.mark.parametrize("values", [
    {"partition": "other"}, {"has_internet_service": 1},
    {"contract": ""}, {"payment_method": " "}, {"revenue_segment": "fake"},
    {"risk_segment": "fake"}, {"customer_id": "c0"},
])
def test_invalid_filters(values):
    with pytest.raises(ValidationError):
        dashboard.PortfolioFilters(**values)


@pytest.mark.parametrize("filename", ["customers.parquet", "thresholds.json"])
def test_checksum_tampering(snapshot, filename):
    directory, _ = snapshot
    with (directory / filename).open("ab") as file:
        file.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        with dashboard.open_portfolio(directory):
            pytest.fail("Must not open corrupt snapshot.")


@pytest.mark.parametrize("mutation", [
    "rows", "partition_rows", "model", "reference_rows", "segment", "score",
    "duplicate_id", "risk_weighted", "churn", "purpose",
])
def test_snapshot_inconsistency_with_updated_checksums(snapshot, mutation):
    directory, frame = snapshot
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "rows":
        manifest["rows"] = 7
    elif mutation == "partition_rows":
        manifest["partition_rows"]["train"] = 5
    elif mutation == "model":
        manifest["model"]["freeze_run_id"] = "other"
    elif mutation == "purpose":
        manifest["purpose"] = "other"
    elif mutation == "reference_rows":
        cuts_path = directory / "thresholds.json"
        cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
        cuts["reference_rows"] = 5
        cuts_path.write_text(json.dumps(cuts), encoding="utf-8")
    else:
        if mutation == "segment":
            frame["risk_segment"] = "Lower risk"
        elif mutation == "score":
            frame["churn_probability"] = np.inf
        elif mutation == "duplicate_id":
            frame["customer_id"] = "same"
        elif mutation == "risk_weighted":
            frame["risk_weighted_monthly_charges"] = 9999.
        else:
            frame["churn"] = 0
        with duckdb.connect() as connection:
            connection.register("fixture", frame)
            connection.table("fixture").write_parquet(str(directory / "customers.parquet"))
    for name in manifest["files"]:
        manifest["files"][name]["sha256"] = file_sha256(directory / name)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        with dashboard.open_portfolio(directory):
            pytest.fail("Must not open inconsistent snapshot.")


def test_scoring_writer_to_dashboard_reader(snapshot, monkeypatch):
    from src.config import load_training_config
    from src.data import score_portfolio
    from src.models.pipeline import FEATURE_COLUMNS
    from src.models.predict import ChurnPredictor, ModelProvenance

    class Model:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            return np.tile([.9, .1], (len(frame), 1))

    directory, frame = snapshot
    inputs = {"synthetic": {"sha256": "abc", "path": "fixture"}}
    monkeypatch.setattr(score_portfolio, "current_provenance", lambda config: inputs)
    predictor = ChurnPredictor(
        Model(), ModelProvenance("model", "source", "freeze", "final"),
    )
    output = score_portfolio.save_snapshot(
        frame.loc[:, ["customer_id", "partition", *FEATURE_COLUMNS]],
        predictor, directory / "writer-output", input_provenance=inputs,
        config=load_training_config(), batch_size=2,
    )
    with dashboard.open_portfolio(output) as connection:
        result = summary(connection)
    assert result.overview.iloc[0]["customers"] == 6
    assert result.overview.iloc[0]["risk_weighted_monthly_charges"] == pytest.approx(33)
    assert result.segments["customers"].sum() == 6
