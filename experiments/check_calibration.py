from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import brier_score_loss, log_loss

from src.config import TrainingConfig, load_training_config
from src.data.features import load_feature_view
from src.data.loader import load_customers
from src.data.splits import load_split_views
from src.models.artifacts import current_provenance, read_json
from src.models.calibration import calibration_report, reliability_figure
from src.models.pipeline import FEATURE_COLUMNS


SOURCE_RUN_ID = "4b8fa8456a984b8da19bb09e4a5d2334"


def main():
    if mlflow.active_run() is not None:
        raise RuntimeError("Start without an active MLflow run.")
    local_config = load_training_config()
    database = Path(local_config.mlflow.tracking_directory) / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError(f"MLflow database not found: {database}")
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    try:
        client = MlflowClient()
        source = client.get_run(SOURCE_RUN_ID)
        if (
            source.info.status != "FINISHED"
            or source.data.tags.get("model") != "catboost"
            or source.data.tags.get("evaluation_partition") != "validation"
            or float(source.data.params["positive_class_weight"]) != 1.0
        ):
            raise ValueError("Expected the completed unweighted CatBoost finalist.")
        parent_id = source.data.tags["mlflow.parentRunId"]
        if client.get_run(parent_id).info.status != "FINISHED":
            raise ValueError("Finalist comparison did not finish.")
        config = TrainingConfig.model_validate(
            read_json(client, parent_id, "evaluation_config.json")
        )
        provenance = current_provenance(config)
        saved = read_json(client, parent_id, "input_provenance.json")
        if any(
            provenance[name]["sha256"] != saved[name]["sha256"]
            for name in provenance
        ):
            raise ValueError("Finalist data or feature SQL changed.")
        comparisons = read_json(client, parent_id, "validation_comparison.json")
        matches = [row for row in comparisons if row["run_id"] == SOURCE_RUN_ID]
        if len(matches) != 1 or not matches[0]["model_uri"]:
            raise ValueError("Expected exactly one saved CatBoost model URI.")

        model = mlflow.sklearn.load_model(matches[0]["model_uri"])
        with duckdb.connect() as connection:
            load_customers(connection)
            load_split_views(connection, Path(config.data.split_manifest))
            load_feature_view(connection, Path(config.data.feature_sql))
            validation = connection.sql("""
                SELECT * FROM baseline_features
                WHERE partition = 'validation'
                ORDER BY customer_id
            """).df()
        required = set(FEATURE_COLUMNS) | {"customer_id", "churn", "partition"}
        if (
            validation.empty
            or not validation.columns.is_unique
            or set(validation.columns) != required
            or not validation["partition"].eq("validation").all()
            or validation["customer_id"].isna().any()
            or not validation["customer_id"].is_unique
        ):
            raise ValueError("Invalid validation data.")
        positive = np.flatnonzero(model.classes_ == 1)
        if len(positive) != 1:
            raise ValueError("Expected one positive class.")
        probabilities = model.predict_proba(
            validation.loc[:, list(FEATURE_COLUMNS)]
        )[:, positive[0]]
        table, metrics = calibration_report(validation["churn"], probabilities)
        for metric in ("log_loss", "brier_score"):
            if not np.isclose(
                metrics[metric], source.data.metrics[f"validation_{metric}"],
                rtol=1e-8, atol=1e-10,
            ):
                raise ValueError(f"{metric} differs from the saved finalist result.")
        figure = reliability_figure(table)
        with mlflow.start_run(
            experiment_id=source.info.experiment_id,
            run_name="catboost_validation_calibration_diagnostics",
        ) as run:
            mlflow.set_tags({
                "source_finalist_run_id": SOURCE_RUN_ID,
                "mode": "calibration_diagnostics",
                "evaluation_partition": "validation",
            })
            mlflow.log_params({
                "binning": "quantile_ties_not_split",
                "requested_bins": 10,
                "validation_rows": len(validation),
            })
            mlflow.log_metrics(metrics)
            mlflow.log_dict(provenance, "input_provenance.json")
            mlflow.log_dict(table.to_dict(orient="records"), "calibration_bins.json")
            mlflow.log_text(table.to_csv(index=False), "calibration_bins.csv")
            mlflow.log_text(
                figure.to_html(include_plotlyjs=True, full_html=True),
                "reliability.html",
            )
            print(f"MLflow diagnostics run: {run.info.run_id}")
        print(pd.Series(metrics).to_string())
        print(table.to_string(index=False))
    finally:
        mlflow.set_tracking_uri(previous_uri)


if __name__ == "__main__":
    main()
