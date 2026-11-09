import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import TrainingConfig
from src.data.features import load_feature_view
from src.data.loader import PROJECT_ROOT, load_customers
from src.data.portfolio import (
    PortfolioThresholds, apply_portfolio_segments, fit_portfolio_thresholds,
)
from src.data.splits import load_split_views
from src.models.artifacts import current_provenance, file_sha256, read_json
from src.models.pipeline import FEATURE_COLUMNS
from src.models.predict import ChurnPredictor


FINAL_TEST_RUN_ID = "5818be9683af42a485ffc2896a868137"


def build_snapshot(
    customers: pd.DataFrame,
    predictor: ChurnPredictor,
    *,
    batch_size: int = 25_000,
) -> tuple[pd.DataFrame, PortfolioThresholds]:
    """Score once per customer; fit quantile cuts only on training rows."""
    expected = {"customer_id", "partition", *FEATURE_COLUMNS}
    if not customers.columns.is_unique or set(customers.columns) != expected:
        raise ValueError("Expected only customer_id, partition and the 28 predictors.")
    if customers.empty:
        raise ValueError("Portfolio must be nonempty.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not customers["customer_id"].is_unique:
        raise ValueError("Customer IDs must be unique across the full portfolio.")
    if not customers["partition"].isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown or missing portfolio partition.")
    if not customers["partition"].eq("train").any():
        raise ValueError("Training rows are required to establish reference thresholds.")

    probabilities = np.empty(len(customers), dtype=float)
    for start in range(0, len(customers), batch_size):
        batch = customers.iloc[start:start + batch_size]
        scores = predictor.predict_batch(batch.loc[:, ["customer_id", *FEATURE_COLUMNS]])
        if scores["customer_id"].tolist() != batch["customer_id"].tolist():
            raise ValueError("Predictor changed batch customer order or coverage.")
        probabilities[start:start + len(batch)] = scores["churn_probability"].to_numpy()

    result = customers.copy()
    result["churn_probability"] = probabilities
    provenance = predictor.provenance
    result["model_run_id"] = provenance.source_finalist_run_id
    result["freeze_run_id"] = provenance.freeze_run_id
    result["final_test_run_id"] = provenance.final_test_run_id
    thresholds = fit_portfolio_thresholds(
        result.loc[result["partition"].eq("train")],
        model_run_id=provenance.source_finalist_run_id,
    )
    return apply_portfolio_segments(
        result, thresholds, model_run_id=provenance.source_finalist_run_id,
    ), thresholds


def load_snapshot_inputs(config: TrainingConfig) -> pd.DataFrame:
    """Reuse strict ingestion and persisted split validation; discard labels."""
    with duckdb.connect() as connection:
        load_customers(connection)
        load_split_views(connection, Path(config.data.split_manifest))
        load_feature_view(connection, Path(config.data.feature_sql))
        columns = ", ".join(
            f'"{name}"' for name in ("customer_id", "partition", *FEATURE_COLUMNS)
        )
        return connection.sql(
            f"SELECT {columns} FROM baseline_features ORDER BY customer_id"
        ).df()


def save_snapshot(
    customers: pd.DataFrame,
    predictor: ChurnPredictor,
    output: Path,
    *,
    input_provenance: dict,
    config: TrainingConfig,
    batch_size: int = 25_000,
) -> Path:
    """Publish a complete new directory, never overwrite a previous snapshot.

    Source fingerprints must match both before scoring and before publication.
    Parquet, threshold JSON and a manifest with output checksums travel together.
    """
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Snapshot output already exists: {output}")
    if current_provenance(config) != input_provenance:
        raise ValueError("Source inputs changed before snapshot generation.")
    snapshot, thresholds = build_snapshot(customers, predictor, batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        parquet = staging / "customers.parquet"
        threshold_file = staging / "thresholds.json"
        with duckdb.connect() as connection:
            connection.register("snapshot", snapshot)
            connection.table("snapshot").write_parquet(str(parquet), compression="zstd")
            saved_rows = connection.read_parquet(str(parquet)).count("*").fetchall()[0][0]
        if saved_rows != len(snapshot):
            raise RuntimeError("Persisted snapshot row count does not match scored rows.")
        threshold_file.write_text(thresholds.model_dump_json(indent=2), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "descriptive_portfolio_not_model_evaluation",
            "training_scores_are_in_sample": True,
            "includes_source_test_customers": bool(snapshot["partition"].eq("test").any()),
            "contains_churn_labels": False,
            "contains_campaign_selection": False,
            "model": asdict(predictor.provenance),
            "threshold_reference": "train",
            "rows": len(snapshot),
            "partition_rows": {
                name: int(count) for name, count
                in snapshot["partition"].value_counts().sort_index().items()
            },
            "input_provenance": input_provenance,
            "files": {
                path.name: {"sha256": file_sha256(path)}
                for path in (parquet, threshold_file)
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8",
        )
        if current_provenance(config) != input_provenance:
            raise ValueError("Source inputs changed during snapshot generation.")
        if output.exists():
            raise FileExistsError(f"Snapshot output already exists: {output}")
        os.rename(staging, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-directory", type=Path, default=PROJECT_ROOT / "mlruns")
    parser.add_argument("--final-test-run-id", default=FINAL_TEST_RUN_ID)
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Snapshot output already exists: {args.output}")
    predictor = ChurnPredictor.from_final_test(
        args.tracking_directory, final_test_run_id=args.final_test_run_id,
    )
    database = args.tracking_directory.resolve() / "mlflow.db"
    client = MlflowClient(tracking_uri=f"sqlite:///{database.as_posix()}")
    frozen = read_json(client, predictor.provenance.freeze_run_id, "frozen_manifest.json")
    config = TrainingConfig.model_validate(frozen["config"])
    inputs = frozen["input_provenance"]
    if current_provenance(config) != inputs:
        raise ValueError("Current inputs differ from the frozen model inputs.")
    customers = load_snapshot_inputs(config)
    destination = save_snapshot(
        customers, predictor, args.output, input_provenance=inputs,
        config=config, batch_size=args.batch_size,
    )
    print(f"Saved descriptive snapshot: {destination}")
    print(f"Customers: {len(customers):,}; training-only reference thresholds.")
    print("No model evaluation, retraining or campaign selection was performed.")


if __name__ == "__main__":
    main()
