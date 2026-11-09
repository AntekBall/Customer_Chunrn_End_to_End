import logging
import warnings
from pathlib import Path
from time import perf_counter

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.exceptions import ConvergenceWarning

from src.config import TrainingConfig, load_training_config
from src.data.features import load_feature_view
from src.data.loader import PROJECT_ROOT, load_customers
from src.data.splits import load_split_views
from src.models.artifacts import current_provenance, read_json
from src.models.evaluate import (
    score_predictions as score_validation,
    validate_partitions,
)
from src.models.pipeline import FEATURE_COLUMNS, build_boosting_pipeline
from src.models.search_spaces import build_tuned_pipeline


SOURCE_RUNS = {
    "catboost": "d61cbb819fa845a8ad25f8cb2775f065",
    "xgboost": "2885f7d8db824bb9b54cb53711153a08",
}


def load_data(config):
    with duckdb.connect() as connection:
        load_customers(connection)
        load_split_views(connection, Path(config.data.split_manifest))
        load_feature_view(connection, Path(config.data.feature_sql))
        return tuple(
            connection.sql(f"""
                SELECT *
                FROM baseline_features
                WHERE partition = '{partition}'
                ORDER BY customer_id
            """).df()
            for partition in ("train", "validation")
        )


def run_validation(config):
    if mlflow.active_run() is not None:
        raise RuntimeError("Start without an active MLflow run.")
    directory = Path(config.mlflow.tracking_directory)
    database = directory / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError(f"MLflow database not found: {database}")

    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    try:
        client = MlflowClient()
        provenance = current_provenance(config)
        candidates = []
        experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
        if experiment is None or experiment.lifecycle_stage != "active":
            raise ValueError("Configured source experiment is missing or deleted.")

        for model_name, run_id in SOURCE_RUNS.items():
            source = client.get_run(run_id)
            if (
                source.info.status != "FINISHED"
                or source.info.experiment_id != experiment.experiment_id
                or source.data.tags.get("model") != model_name
                or float(source.data.params.get("positive_class_weight", "1")) != 1
            ):
                raise ValueError(f"Invalid unweighted source run: {model_name}")
            parent_id = source.data.tags["mlflow.parentRunId"]
            parent = client.get_run(parent_id)
            best = read_json(client, parent_id, "best_trial.json")
            if parent.info.status != "FINISHED" or best["best_trial_run_id"] != run_id:
                raise ValueError("Source is not a completed best-trial selection.")
            saved = read_json(client, parent_id, "input_provenance.json")
            if any(
                provenance[name]["sha256"] != saved[name]["sha256"]
                for name in provenance
            ):
                raise ValueError(f"Source data or feature SQL changed: {model_name}")
            source_config = TrainingConfig.model_validate(
                read_json(client, parent_id, "resolved_config.json")
            )
            candidates.append((
                model_name,
                build_tuned_pipeline(model_name, source_config, best["best_parameters"]),
                run_id,
            ))

        candidates.append(("histogram_baseline", build_boosting_pipeline(), None))
        training, validation = load_data(config)
        validate_partitions(training, validation)
        train_features = training.loc[:, list(FEATURE_COLUMNS)]
        validation_features = validation.loc[:, list(FEATURE_COLUMNS)]
        results = []

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name="unweighted_finalist_validation",
        ) as parent:
            mlflow.set_tags({
                "mode": "finalist_validation",
                "evaluation_partition": "validation",
            })
            mlflow.log_dict(provenance, "input_provenance.json")
            mlflow.log_dict(config.model_dump(mode="json"), "evaluation_config.json")
            for name, model, source_id in candidates:
                logging.info("Fitting %s on %s training rows.", name, len(training))
                with mlflow.start_run(
                    experiment_id=experiment.experiment_id,
                    run_name=name,
                    nested=True,
                ) as child:
                    mlflow.set_tags({
                        "model": name,
                        "evaluation_partition": "validation",
                    })
                    if source_id is not None:
                        mlflow.set_tag("source_trial_run_id", source_id)
                    mlflow.log_params({
                        **model.named_steps["classifier"].get_params(),
                        "positive_class_weight": 1.0,
                        "training_rows": len(training),
                        "validation_rows": len(validation),
                    })
                    started = perf_counter()
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(train_features, training["churn"])
                    fit_seconds = perf_counter() - started
                    positive_class = np.flatnonzero(model.classes_ == 1)
                    if len(positive_class) != 1:
                        raise ValueError("Expected one positive churn class.")
                    started = perf_counter()
                    probabilities = model.predict_proba(
                        validation_features
                    )[:, positive_class[0]]
                    predict_seconds = perf_counter() - started
                    metrics = score_validation(validation, probabilities, config)
                    metrics.update(
                        fit_seconds=fit_seconds, predict_seconds=predict_seconds
                    )
                    mlflow.log_metrics({
                        f"validation_{key}": value for key, value in metrics.items()
                    })
                    model_uri = None
                    if config.mlflow.log_finalist_models:
                        logged = mlflow.sklearn.log_model(
                            model,
                            name="pipeline",
                            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                            signature=infer_signature(train_features),
                            code_paths=[str(PROJECT_ROOT / "src")],
                        )
                        model_uri = logged.model_uri
                    results.append({
                        "model": name, **metrics,
                        "run_id": child.info.run_id,
                        "model_uri": model_uri,
                        "parent_run_id": parent.info.run_id,
                    })
            mlflow.log_dict(results, "validation_comparison.json")
        return pd.DataFrame(results).sort_values(
            config.optimization.metric,
            ascending=config.optimization.direction == "minimize",
        ).reset_index(drop=True)
    finally:
        mlflow.set_tracking_uri(previous_uri)


def main():
    logging.basicConfig(level=logging.INFO)
    results = run_validation(load_training_config())
    print(results.drop(columns=["parent_run_id", "model_uri"]).to_string(index=False))


if __name__ == "__main__":
    main()
