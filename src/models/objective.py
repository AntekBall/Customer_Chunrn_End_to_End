import logging
import traceback
import warnings
from collections.abc import Callable
from time import perf_counter

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import TrainingConfig
from src.models.model_builder import MODEL_NAMES, ModelName
from src.models.pipeline import FEATURE_COLUMNS
from src.models.search_spaces import (
    build_tuned_pipeline,
    suggest_parameters,
)


logger = logging.getLogger(__name__)

METRIC_FUNCTIONS = {
    "average_precision": average_precision_score,
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
    "brier_score": brier_score_loss,
}


def make_objective(
    model_name: ModelName,
    config: TrainingConfig,
    training_data: pd.DataFrame,
    experiment_id: str,
    *,
    positive_class_weight: float = 1.0,
) -> Callable[[optuna.trial.Trial | optuna.trial.FixedTrial], float]:
    if (
        isinstance(positive_class_weight, bool)
        or not isinstance(positive_class_weight, (int, float))
        or not np.isfinite(positive_class_weight)
        or positive_class_weight <= 0
    ):
        raise ValueError("positive_class_weight must be finite and positive.")

    config = config.model_copy(deep=True)

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name!r}")

    if not getattr(config.models, model_name).enabled:
        raise ValueError(f"Model is disabled: {model_name}")

    required = set(FEATURE_COLUMNS) | {
        "customer_id", "churn", "partition"
    }

    if (
        not training_data.columns.is_unique
        or set(training_data.columns) != required
    ):
        raise ValueError("Training data does not match the feature schema.")

    if training_data.empty or not training_data["partition"].eq("train").all():
        raise ValueError("Objective requires nonempty training-only data.")

    ids = training_data["customer_id"].copy()
    target = training_data["churn"].copy()
    features = training_data.loc[:, list(FEATURE_COLUMNS)].copy()

    if ids.isna().any() or not ids.is_unique:
        raise ValueError("Training IDs must be non-null and unique.")

    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("Training data must contain both binary classes.")

    fold_count = config.cross_validation.n_splits

    if target.value_counts().min() < fold_count:
        raise ValueError("Each class needs at least one customer per fold.")

    cv = StratifiedKFold(
        n_splits=fold_count,
        shuffle=config.cross_validation.shuffle,
        random_state=(
            config.random_seed
            if config.cross_validation.shuffle
            else None
        ),
    )

    splits = list(cv.split(features, target))

    def objective(
        trial: optuna.trial.Trial | optuna.trial.FixedTrial,
    ) -> float:
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"{model_name}_trial_{trial.number}",
            nested=mlflow.active_run() is not None,
        ) as run:
            trial.set_user_attr("mlflow_run_id", run.info.run_id)

            try:
                parameters = suggest_parameters(trial, model_name)

                mlflow.set_tags({
                    "model": model_name,
                    "optuna_trial": str(trial.number),
                    "evaluation_partition": "train_cv",
                })

                mlflow.log_dict(
                    config.model_dump(mode="json"),
                    "resolved_config.json",
                )

                mlflow.log_dict(
                    {
                        "predictors": list(FEATURE_COLUMNS),
                        "dtypes": features.dtypes.astype(str).to_dict(),
                    },
                    "feature_schema.json",
                )

                template = build_tuned_pipeline(
                    model_name, config, parameters
                )

                mlflow.log_params({
                    "random_seed": config.random_seed,
                    "cv_folds": fold_count,
                    "positive_class_weight": positive_class_weight,
                    "sample_weight_normalization": "fit_fold_mean_one",
                    "training_rows": len(features),
                    **template.named_steps["classifier"].get_params(),
                })

                fold_results = []

                for fold, (fit_indices, score_indices) in enumerate(splits):
                    model = build_tuned_pipeline(
                        model_name, config, parameters
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)

                        started = perf_counter()
                        fit_features = features.iloc[fit_indices]
                        fit_target = target.iloc[fit_indices]

                        if positive_class_weight == 1.0:
                            model.fit(fit_features, fit_target)
                        else:
                            weights = np.where(
                                fit_target.to_numpy() == 1,
                                positive_class_weight,
                                1.0,
                            )
                            weights /= weights.mean()
                            model.fit(
                                fit_features,
                                fit_target,
                                classifier__sample_weight=weights,
                            )
                        fit_seconds = perf_counter() - started

                    positive_class = np.flatnonzero(model.classes_ == 1)

                    if len(positive_class) != 1:
                        raise ValueError("Expected one positive churn class.")

                    score_features = features.iloc[score_indices]
                    score_target = target.iloc[score_indices]

                    started = perf_counter()
                    probabilities = model.predict_proba(
                        score_features
                    )[:, positive_class[0]]
                    predict_seconds = perf_counter() - started

                    if (
                        not np.isfinite(probabilities).all()
                        or (probabilities < 0).any()
                        or (probabilities > 1).any()
                    ):
                        raise ValueError("Model produced invalid probabilities.")

                    metrics = {
                        name: float(
                            METRIC_FUNCTIONS[name](
                                score_target, probabilities
                            )
                        )
                        for name in config.evaluation.metrics
                    }

                    metrics["fit_seconds"] = fit_seconds
                    metrics["predict_seconds"] = predict_seconds

                    ranked = pd.DataFrame({
                        "customer_id": ids.iloc[score_indices].to_numpy(),
                        "churn": score_target.to_numpy(),
                        "probability": probabilities,
                    }).sort_values(
                        ["probability", "customer_id"],
                        ascending=[False, True],
                    )

                    for fraction in config.evaluation.targeting_fractions:
                        selected_count = int(
                            np.ceil(fraction * len(ranked))
                        )
                        captured = int(
                            ranked.head(selected_count)["churn"].sum()
                        )
                        suffix = f"top_{100 * fraction:g}pct"

                        metrics[f"precision_{suffix}"] = (
                            captured / selected_count
                        )
                        metrics[f"capture_{suffix}"] = (
                            captured / int(score_target.sum())
                        )

                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise ValueError("Evaluation produced non-finite metrics.")

                    mlflow.log_metrics(
                        {
                            f"fold_{name}": value
                            for name, value in metrics.items()
                        },
                        step=fold,
                    )

                    if config.mlflow.log_trial_models:
                        mlflow.sklearn.log_model(
                            model,
                            name=f"fold_{fold + 1}_pipeline",
                        )

                    fold_results.append(metrics)

                results = pd.DataFrame(fold_results)

                summary = {
                    **{
                        f"mean_{name}": float(value)
                        for name, value in results.mean().items()
                    },
                    **{
                        f"std_{name}": float(value)
                        for name, value in results.std(ddof=1).items()
                    },
                }

                mlflow.log_metrics(summary)
                mlflow.log_dict(
                    results.to_dict(orient="records"),
                    "fold_metrics.json",
                )

                score = summary[f"mean_{config.optimization.metric}"]
                trial.set_user_attr("cv_metrics", summary)
                mlflow.set_tag("trial_state", "COMPLETE")

                return score

            except Exception:
                logger.exception(
                    "Trial %s failed for %s", trial.number, model_name
                )
                mlflow.set_tag("trial_state", "FAIL")
                mlflow.log_text(traceback.format_exc(), "error.txt")
                raise

    return objective