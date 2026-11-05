from typing import Literal

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import TrainingConfig
from src.models.pipeline import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_boosting_pipeline,
)


ModelName = Literal[
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "random_forest",
    "catboost",
]

MODEL_NAMES: tuple[ModelName, ...] = (
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "random_forest",
    "catboost",
)


def prepare_catboost_features(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[:, list(FEATURE_COLUMNS)].copy()

    for column in CATEGORICAL_COLUMNS:
        if selected[column].isna().any():
            raise ValueError(
                f"Unexpected missing categorical values: {column}"
            )

        selected[column] = selected[column].astype(str)

    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        selected[column] = selected[column].astype("float64")

    return selected


def build_model_pipeline(
    model_name: ModelName,
    config: TrainingConfig,
) -> Pipeline:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name!r}")

    model_config = getattr(config.models, model_name)

    if not model_config.enabled:
        raise ValueError(f"Model is disabled: {model_name}")

    fixed = model_config.fixed_parameters.model_dump()
    seed = config.random_seed

    baseline = build_boosting_pipeline()

    if model_name == "hist_gradient_boosting":
        baseline.named_steps["classifier"].set_params(
            **fixed,
            random_state=seed,
        )
        return baseline

    preprocessor = clone(baseline.named_steps["preprocessor"])

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            **fixed,
            random_state=seed,
            n_estimators=150,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=100,
            reg_lambda=1.0,
        )

    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            **fixed,
            random_state=seed,
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            reg_lambda=1.0,
        )

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            **fixed,
            random_state=seed,
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=10,
            max_features="sqrt",
        )

    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        preprocessor = FunctionTransformer(
            prepare_catboost_features,
        )

        classifier = CatBoostClassifier(
            **fixed,
            random_seed=seed,
            iterations=150,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3.0,
            cat_features=list(CATEGORICAL_COLUMNS),
            verbose=False,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name!r}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )