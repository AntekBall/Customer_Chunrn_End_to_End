from collections.abc import Mapping

import optuna
from sklearn.pipeline import Pipeline

from src.config import TrainingConfig
from src.models.model_builder import (
    MODEL_NAMES,
    ModelName,
    build_model_pipeline,
)


ParameterValue = int | float | str | bool | None

TUNABLE_PARAMETERS = {
    "hist_gradient_boosting": {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    },
    "lightgbm": {
        "learning_rate",
        "n_estimators",
        "num_leaves",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "colsample_bytree",
    },
    "xgboost": {
        "learning_rate",
        "n_estimators",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    },
    "random_forest": {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "max_samples",
    },
    "catboost": {
        "learning_rate",
        "iterations",
        "depth",
        "l2_leaf_reg",
        "random_strength",
    },
}


def suggest_parameters(
    trial: optuna.trial.Trial,
    model_name: ModelName,
) -> dict[str, ParameterValue]:
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "max_iter": trial.suggest_int(
                "max_iter", 100, 400, step=50
            ),
            "max_leaf_nodes": trial.suggest_categorical(
                "max_leaf_nodes", [7, 15, 31]
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 50, 300, step=50
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 0.001, 10.0, log=True
            ),
        }

    if model_name == "lightgbm":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 400, step=50
            ),
            "num_leaves": trial.suggest_categorical(
                "num_leaves", [7, 15, 31]
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 50, 300, step=50
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 0.0001, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.001, 10.0, log=True
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
        }

    if model_name == "xgboost":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 400, step=50
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 50.0, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 0.0001, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.001, 10.0, log=True
            ),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", [100, 200, 300]
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", [8, 12, 16]
            ),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", [10, 25, 50, 100]
            ),
            "max_features": trial.suggest_float(
                "max_features", 0.3, 0.8
            ),
            "max_samples": trial.suggest_float(
                "max_samples", 0.5, 0.9
            ),
        }

    if model_name == "catboost":
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "iterations": trial.suggest_int(
                "iterations", 100, 400, step=50
            ),
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 20.0, log=True
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.0, 2.0
            ),
        }

    raise ValueError(f"Unknown model: {model_name!r}")


def build_tuned_pipeline(
    model_name: ModelName,
    config: TrainingConfig,
    parameters: Mapping[str, ParameterValue],
) -> Pipeline:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name!r}")

    fixed_parameters = getattr(
        config.models, model_name
    ).fixed_parameters.model_dump()

    conflicts = set(parameters) & set(fixed_parameters)

    if conflicts:
        raise ValueError(
            f"Cannot override fixed parameters: {sorted(conflicts)}"
        )

    unsupported = set(parameters) - TUNABLE_PARAMETERS[model_name]

    if unsupported:
        raise ValueError(
            f"Unsupported tuning parameters: {sorted(unsupported)}"
        )

    pipeline = build_model_pipeline(model_name, config)

    pipeline.set_params(
        **{
            f"classifier__{name}": value
            for name, value in parameters.items()
        }
    )

    return pipeline