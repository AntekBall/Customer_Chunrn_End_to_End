import json
import logging
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import TrainingConfig, load_training_config
from src.data.loader import DATA_PATH
from src.models.objective import make_objective
from src.models.train import file_sha256, load_training_data


SOURCE_RUNS = {
    "catboost": "d61cbb819fa845a8ad25f8cb2775f065",
    "xgboost": "2885f7d8db824bb9b54cb53711153a08",
}
POSITIVE_WEIGHTS = (1.0, 1.5, 2.0, 3.0)


def read_json(client, run_id, filename):
    path = client.download_artifacts(run_id, filename)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    logging.basicConfig(level=logging.INFO)
    if mlflow.active_run() is not None:
        raise RuntimeError("Start without an active MLflow run.")

    current = load_training_config()
    directory = Path(current.mlflow.tracking_directory)
    database = directory / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError(f"MLflow database not found: {database}")

    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
    try:
        client = MlflowClient()
        sources = {}
        for model_name, source_run_id in SOURCE_RUNS.items():
            source = client.get_run(source_run_id)
            if (
                source.info.status != "FINISHED"
                or source.data.tags.get("model") != model_name
            ):
                raise ValueError(f"Invalid source run for {model_name}.")

            parent_id = source.data.tags["mlflow.parentRunId"]
            parent = client.get_run(parent_id)
            if parent.info.status != "FINISHED":
                raise ValueError(f"Source parent failed for {model_name}.")
            best = read_json(client, parent_id, "best_trial.json")
            if best["best_trial_run_id"] != source_run_id:
                raise ValueError("Source run is not the recorded best trial.")

            config = TrainingConfig.model_validate(
                read_json(client, parent_id, "resolved_config.json")
            )
            provenance = read_json(client, parent_id, "input_provenance.json")
            for name, path in {
                "raw_data": DATA_PATH,
                "split_manifest": Path(config.data.split_manifest),
                "feature_sql": Path(config.data.feature_sql),
            }.items():
                if file_sha256(path) != provenance[name]["sha256"]:
                    raise ValueError(f"Source input changed: {name}")

            config.mlflow.log_trial_models = False
            sources[model_name] = (
                config, best["best_parameters"], source, provenance
            )

        results = []
        for model_name, (config, parameters, source, provenance) in sources.items():
            data = load_training_data(config)
            with mlflow.start_run(
                experiment_id=source.info.experiment_id,
                run_name=f"{model_name}_class_weight_comparison",
            ) as parent:
                mlflow.set_tags({
                    "model": model_name,
                    "mode": "class_weight_comparison",
                    "source_trial_run_id": source.info.run_id,
                    "evaluation_partition": "train_cv",
                })
                mlflow.log_dict(provenance, "input_provenance.json")
                mlflow.log_dict(parameters, "fixed_model_parameters.json")
                mlflow.log_dict(
                    {"positive_class_weights": list(POSITIVE_WEIGHTS)},
                    "weight_grid.json",
                )

                baseline = None
                model_results = []
                for number, weight in enumerate(POSITIVE_WEIGHTS):
                    logging.info("%s: positive class weight %s", model_name, weight)
                    trial = optuna.trial.FixedTrial(parameters, number=number)
                    objective = make_objective(
                        model_name, config, data, source.info.experiment_id,
                        positive_class_weight=weight,
                    )
                    objective(trial)
                    metrics = trial.user_attrs["cv_metrics"]
                    if baseline is None:
                        baseline = metrics
                    row = {
                        "model": model_name,
                        "positive_class_weight": weight,
                        "trial_run_id": trial.user_attrs["mlflow_run_id"],
                        "parent_run_id": parent.info.run_id,
                        **metrics,
                        **{
                            f"delta_{name}": value - baseline[name]
                            for name, value in metrics.items()
                            if name.startswith("mean_")
                        },
                    }
                    model_results.append(row)
                mlflow.log_dict(model_results, "weight_comparison.json")
                results.extend(model_results)
            del data

        summary = pd.DataFrame(results)
        print(summary[[
            "model", "positive_class_weight",
            "mean_average_precision", "delta_mean_average_precision",
            "mean_capture_top_10pct", "mean_capture_top_20pct",
            "mean_log_loss", "mean_brier_score",
        ]].to_string(index=False))
    finally:
        mlflow.set_tracking_uri(previous_uri)


if __name__ == "__main__":
    main()
