import argparse
import re
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from experiments.check_calibration import calibration_report, reliability_figure
from experiments.finalist_validation import (
    current_provenance, read_json, score_validation,
)
from experiments.financial_policy import CampaignScenario, compare_policies
from src.config import TrainingConfig, load_training_config
from src.data.features import load_feature_view
from src.data.loader import load_customers
from src.data.splits import load_split_views
from src.models.pipeline import FEATURE_COLUMNS


SCENARIO_RUN_ID = "129ead19386044cb824002981e41fbd1"
POLICY = {
    "ranking": "expected_net_value_desc_customer_id_asc",
    "eligibility": "expected_net_value > 0",
    "capacity": "floor(capacity_fraction * population_size)",
}


def verify_inputs(config, expected):
    actual = current_provenance(config)
    if set(actual) != set(expected) or any(
        actual[name]["sha256"] != expected[name]["sha256"] for name in actual
    ):
        raise ValueError("Frozen input data, split, or feature SQL changed.")


def freeze(client):
    scenario_run = client.get_run(SCENARIO_RUN_ID)
    if (
        scenario_run.info.status != "FINISHED"
        or scenario_run.data.tags.get("mode") != "financial_scenario"
        or scenario_run.data.tags.get("evaluation_partition") != "validation"
    ):
        raise ValueError("Expected a finished validation scenario run.")
    source_id = scenario_run.data.tags["source_finalist_run_id"]
    source = client.get_run(source_id)
    if (
        source.info.status != "FINISHED"
        or source.data.tags.get("model") != "catboost"
        or source.data.tags.get("evaluation_partition") != "validation"
        or float(source.data.params["positive_class_weight"]) != 1
    ):
        raise ValueError("Expected a finished unweighted CatBoost finalist.")
    parent_id = source.data.tags["mlflow.parentRunId"]
    if client.get_run(parent_id).info.status != "FINISHED":
        raise ValueError("Source finalist comparison did not finish.")
    scenario = read_json(client, SCENARIO_RUN_ID, "campaign_scenario.json")
    CampaignScenario(**scenario)
    expected_scenario = {
        "contact_cost": 2.0, "gross_margin": 0.2, "retained_months": 6.0,
        "prevention_probability": 0.2, "capacity_fraction": 0.1,
    }
    if scenario != expected_scenario:
        raise ValueError("Scenario differs from the agreed base assumptions.")
    definition = read_json(client, SCENARIO_RUN_ID, "policy_definition.json")
    if any(definition[name] != value for name, value in POLICY.items()):
        raise ValueError("Scenario policy differs from the agreed rule.")
    config_data = read_json(client, parent_id, "evaluation_config.json")
    config = TrainingConfig.model_validate(config_data)
    provenance = read_json(client, parent_id, "input_provenance.json")
    verify_inputs(config, provenance)
    verify_inputs(
        config, read_json(client, SCENARIO_RUN_ID, "input_provenance.json")
    )
    comparisons = read_json(client, parent_id, "validation_comparison.json")
    selected = [row for row in comparisons if row["run_id"] == source_id]
    if len(selected) != 1 or not selected[0]["model_uri"]:
        raise ValueError("Expected exactly one saved finalist model.")
    manifest = {
        "model_uri": selected[0]["model_uri"],
        "source_finalist_run_id": source_id,
        "source_scenario_run_id": SCENARIO_RUN_ID,
        "config": config_data,
        "input_provenance": provenance,
        "scenario": scenario,
        "policy": POLICY,
        "calibration": "none",
        "calibration_bins": 10,
        "test_plan": "Single frozen model; no refit, tuning, or sensitivity search.",
    }
    with mlflow.start_run(
        experiment_id=source.info.experiment_id, run_name="frozen_catboost_policy"
    ) as run:
        mlflow.set_tag("mode", "frozen_model_policy")
        mlflow.log_dict(manifest, "frozen_manifest.json")
        print(f"Frozen setup run ID: {run.info.run_id}")
        return run.info.run_id


def load_test(config):
    with duckdb.connect() as connection:
        load_customers(connection)
        load_split_views(connection, Path(config.data.split_manifest))
        load_feature_view(connection, Path(config.data.feature_sql))
        return connection.sql("""
            SELECT * FROM baseline_features
            WHERE partition = 'test'
            ORDER BY customer_id
        """).df()


def validate_test(frame):
    required = set(FEATURE_COLUMNS) | {"customer_id", "churn", "partition"}
    if (
        frame.empty or not frame.columns.is_unique
        or set(frame.columns) != required
        or not frame["partition"].eq("test").all()
    ):
        raise ValueError("Expected only test rows with the frozen feature schema.")
    ids = frame["customer_id"]
    if not ids.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all() or not ids.is_unique:
        raise ValueError("Expected unique nonblank test customer IDs.")
    if frame["churn"].isna().any() or set(frame["churn"].unique()) != {0, 1}:
        raise ValueError("Expected both binary classes in test.")


def evaluate(client, freeze_id, directory):
    frozen = client.get_run(freeze_id)
    if (
        frozen.info.status != "FINISHED"
        or frozen.data.tags.get("mode") != "frozen_model_policy"
    ):
        raise ValueError("Expected a completed freeze run.")
    manifest = read_json(client, freeze_id, "frozen_manifest.json")
    if manifest["policy"] != POLICY or manifest["calibration"] != "none":
        raise ValueError("Unsupported frozen policy or calibration.")
    config = TrainingConfig.model_validate(manifest["config"])
    scenario = CampaignScenario(**manifest["scenario"])
    verify_inputs(config, manifest["input_provenance"])
    model = mlflow.sklearn.load_model(manifest["model_uri"])
    positive = np.flatnonzero(model.classes_ == 1)
    if len(positive) != 1:
        raise ValueError("Expected one positive churn class.")

    split_hash = manifest["input_provenance"]["split_manifest"]["sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", split_hash):
        raise ValueError("Invalid split fingerprint.")
    marker = directory / f"final_test_{split_hash}.lock"
    try:
        with marker.open("x", encoding="utf-8") as file:
            file.write(f"freeze_run_id={freeze_id}\n")
    except FileExistsError as error:
        raise RuntimeError(
            f"Test evaluation already reserved for this split: {marker}. "
            "Inspect the existing run; do not retry automatically."
        ) from error

    with mlflow.start_run(
        experiment_id=frozen.info.experiment_id, run_name="frozen_catboost_final_test"
    ) as run:
        with marker.open("a", encoding="utf-8") as file:
            file.write(f"test_run_id={run.info.run_id}\n")
        mlflow.set_tags({
            "mode": "final_test",
            "evaluation_partition": "test",
            "freeze_run_id": freeze_id,
            "source_finalist_run_id": manifest["source_finalist_run_id"],
            "not_measured_causal_roi": "true",
        })
        mlflow.log_dict(manifest, "frozen_manifest.json")
        test = load_test(config)
        validate_test(test)
        probabilities = model.predict_proba(
            test.loc[:, list(FEATURE_COLUMNS)]
        )[:, positive[0]]
        metrics = score_validation(test, probabilities, config)
        bins, calibration = calibration_report(
            test["churn"], probabilities, n_bins=manifest["calibration_bins"]
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_metrics({
            f"calibration_{key}": value for key, value in calibration.items()
        })
        _, policies = compare_policies(
            test[["customer_id", "monthlycharges"]], probabilities, scenario
        )
        selected = policies.loc[
            policies["policy"] == "expected_profit_positive_only"
        ].copy()
        records = selected.astype(object).where(selected.notna(), None).to_dict(
            orient="records"
        )
        mlflow.log_dict(records, "frozen_policy_scenario.json")
        mlflow.log_metrics({
            f"scenario_{key}": value
            for key, value in records[0].items()
            if key != "policy" and value is not None
        })
        figure = reliability_figure(bins)
        figure.data[0].name = "Test bins"
        figure.update_layout(
            title="CatBoost test reliability (95% pointwise Wilson intervals)"
        )
        mlflow.log_text(bins.to_csv(index=False), "calibration_bins.csv")
        mlflow.log_text(
            figure.to_html(include_plotlyjs=True, full_html=True), "reliability.html"
        )
        mlflow.log_param("test_rows", len(test))
        print(f"Final test run: {run.info.run_id}")
        print(pd.Series(metrics).to_string())
        print(pd.Series(calibration).to_string())
        print("SCENARIO ESTIMATES, NOT MEASURED CAMPAIGN RETURNS")
        print(selected.to_string(index=False))
        return metrics, calibration, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--evaluate", metavar="FREEZE_RUN_ID")
    args = parser.parse_args()
    if mlflow.active_run() is not None:
        raise RuntimeError("Start without an active MLflow run.")
    config = load_training_config()
    directory = Path(config.mlflow.tracking_directory)
    database = directory / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError(f"MLflow database not found: {database}")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    try:
        client = MlflowClient()
        if args.freeze:
            freeze(client)
        else:
            evaluate(client, args.evaluate, directory)
    finally:
        mlflow.set_tracking_uri(previous_uri)


if __name__ == "__main__":
    main()
