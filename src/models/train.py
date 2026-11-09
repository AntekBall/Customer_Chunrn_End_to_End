import argparse
import logging
from pathlib import Path
from uuid import uuid4

import duckdb
import mlflow
import optuna
import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, TrainingConfig, load_training_config
from src.data.features import load_feature_view
from src.data.loader import DATA_PATH, load_customers
from src.data.splits import load_split_views
from src.models.artifacts import file_sha256
from src.models.model_builder import MODEL_NAMES
from src.models.objective import make_objective


logger = logging.getLogger(__name__)


def load_training_data(config: TrainingConfig) -> pd.DataFrame:
    with duckdb.connect() as connection:
        load_customers(connection)
        load_split_views(connection, Path(config.data.split_manifest))
        load_feature_view(connection, Path(config.data.feature_sql))

        return connection.sql("""
            SELECT *
            FROM baseline_features
            WHERE partition = 'train'
            ORDER BY customer_id
        """).df()


def run_training(
    config: TrainingConfig,
    *,
    smoke: bool = False,
) -> pd.DataFrame:
    if mlflow.active_run() is not None:
        raise RuntimeError("Start the runner without an active MLflow run.")

    config = config.model_copy(deep=True)

    if smoke:
        config.optimization.trials_per_model = 1
        config.mlflow.log_trial_models = False

    directory = Path(config.mlflow.tracking_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    artifacts = directory / "artifacts"
    artifacts.mkdir(exist_ok=True)

    mlflow_uri = f"sqlite:///{(directory / 'mlflow.db').as_posix()}"
    optuna_uri = f"sqlite:///{(directory / 'optuna.db').as_posix()}"

    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    try:
        experiment = mlflow.get_experiment_by_name(
            config.mlflow.experiment_name
        )

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                config.mlflow.experiment_name,
                artifact_location=artifacts.as_uri(),
            )
        else:
            if experiment.lifecycle_stage != "active":
                raise RuntimeError(
                    "The configured MLflow experiment is deleted."
                )

            if experiment.artifact_location != artifacts.as_uri():
                raise RuntimeError(
                    "Existing experiment has a different artifact location. "
                    "Choose a new experiment name or use its original directory."
                )

            experiment_id = experiment.experiment_id

        logger.info("Loading and validating training data.")
        training_data = load_training_data(config)

        provenance = {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in {
                "raw_data": DATA_PATH,
                "split_manifest": Path(config.data.split_manifest),
                "feature_sql": Path(config.data.feature_sql),
            }.items()
        }

        batch_id = uuid4().hex
        results = []

        for model_name in MODEL_NAMES:
            if not getattr(config.models, model_name).enabled:
                continue

            study_name = f"{batch_id}_{model_name}"

            logger.info(
                "Starting %s with %s trials.",
                model_name,
                config.optimization.trials_per_model,
            )

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{model_name}_{batch_id}",
            ) as parent:
                study = optuna.create_study(
                    study_name=study_name,
                    storage=optuna_uri,
                    direction=config.optimization.direction,
                    sampler=optuna.samplers.TPESampler(
                        seed=config.random_seed
                    ),
                    pruner=optuna.pruners.NopPruner(),
                    load_if_exists=False,
                )

                study.set_user_attr(
                    "mlflow_parent_run_id",
                    parent.info.run_id,
                )
                study.set_user_attr("batch_id", batch_id)

                mlflow.set_tags({
                    "model": model_name,
                    "batch_id": batch_id,
                    "study_name": study_name,
                    "mode": "smoke" if smoke else "tuning",
                    "evaluation_partition": "train_cv",
                })

                mlflow.log_dict(
                    config.model_dump(mode="json"),
                    "resolved_config.json",
                )
                mlflow.log_dict(
                    provenance,
                    "input_provenance.json",
                )

                study.optimize(
                    make_objective(
                        model_name,
                        config,
                        training_data,
                        experiment_id,
                    ),
                    n_trials=config.optimization.trials_per_model,
                    timeout=config.optimization.timeout_seconds_per_model,
                    n_jobs=config.optimization.parallel_trials,
                    gc_after_trial=True,
                )

                best = study.best_trial

                result = {
                    "model": model_name,
                    "study_name": study_name,
                    "completed_trials": len(study.get_trials(
                        states=(optuna.trial.TrialState.COMPLETE,)
                    )),
                    "metric": config.optimization.metric,
                    "best_cv_score": best.value,
                    "best_trial_number": best.number,
                    "best_trial_run_id": best.user_attrs["mlflow_run_id"],
                    "parent_run_id": parent.info.run_id,
                    "best_parameters": best.params,
                }

                mlflow.log_metric("best_cv_score", best.value)
                mlflow.log_dict(result, "best_trial.json")
                results.append(result)

        summary = pd.DataFrame(results).sort_values(
            "best_cv_score",
            ascending=config.optimization.direction == "minimize",
        ).reset_index(drop=True)

        logger.info("MLflow database: %s", directory / "mlflow.db")
        logger.info("Optuna database: %s", directory / "optuna.db")

        return summary

    finally:
        mlflow.set_tracking_uri(previous_uri)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-only model tuning."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One trial per enabled model; no trial model artifacts.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results = run_training(
        load_training_config(args.config),
        smoke=args.smoke,
    )

    print(results[[
        "model",
        "completed_trials",
        "metric",
        "best_cv_score",
        "best_trial_number",
        "best_trial_run_id",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()