from pathlib import Path
from typing import Annotated, Generic, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.loader import PROJECT_ROOT


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training.yaml"

PositiveInt = Annotated[int, Field(gt=0)]
NonemptyString = Annotated[str, Field(min_length=1)]
Metric = Literal[
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier_score",
]


class StrictConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class DataConfig(StrictConfig):
    split_manifest: NonemptyString
    feature_sql: NonemptyString


class CrossValidationConfig(StrictConfig):
    n_splits: Annotated[int, Field(ge=2)]
    shuffle: bool


class OptimizationConfig(StrictConfig):
    metric: Metric
    direction: Literal["maximize", "minimize"]
    trials_per_model: PositiveInt
    timeout_seconds_per_model: PositiveInt
    parallel_trials: Literal[1]

    @model_validator(mode="after")
    def validate_direction(self):
        expected = (
            "maximize"
            if self.metric in {"average_precision", "roc_auc"}
            else "minimize"
        )

        if self.direction != expected:
            raise ValueError(
                f"{self.metric} requires direction={expected!r}."
            )

        return self


class EvaluationConfig(StrictConfig):
    metrics: Annotated[list[Metric], Field(min_length=1)]
    targeting_fractions: Annotated[
        list[Annotated[float, Field(gt=0, le=1)]],
        Field(min_length=1),
    ]

    @model_validator(mode="after")
    def validate_unique_values(self):
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("Evaluation metrics must not contain duplicates.")

        fractions = self.targeting_fractions

        if len(fractions) != len(set(fractions)):
            raise ValueError("Targeting fractions must not contain duplicates.")

        return self


class MLflowConfig(StrictConfig):
    experiment_name: NonemptyString
    tracking_directory: NonemptyString
    log_trial_models: bool
    log_finalist_models: bool


class HistogramParameters(StrictConfig):
    early_stopping: Literal[False]


class LightGBMParameters(StrictConfig):
    objective: Literal["binary"]
    n_jobs: PositiveInt


class XGBoostParameters(StrictConfig):
    objective: Literal["binary:logistic"]
    tree_method: Literal["hist"]
    eval_metric: Literal["logloss"]
    n_jobs: PositiveInt


class RandomForestParameters(StrictConfig):
    n_jobs: PositiveInt


class CatBoostParameters(StrictConfig):
    loss_function: Literal["Logloss"]
    thread_count: PositiveInt
    allow_writing_files: Literal[False]


Parameters = TypeVar("Parameters", bound=StrictConfig)


class ModelConfig(StrictConfig, Generic[Parameters]):
    enabled: bool
    fixed_parameters: Parameters


class ModelsConfig(StrictConfig):
    hist_gradient_boosting: ModelConfig[HistogramParameters]
    lightgbm: ModelConfig[LightGBMParameters]
    xgboost: ModelConfig[XGBoostParameters]
    random_forest: ModelConfig[RandomForestParameters]
    catboost: ModelConfig[CatBoostParameters]

    @model_validator(mode="after")
    def validate_enabled_models(self):
        if not any(
            getattr(self, name).enabled
            for name in type(self).model_fields
        ):
            raise ValueError("At least one model must be enabled.")

        return self


class TrainingConfig(StrictConfig):
    random_seed: Annotated[int, Field(ge=0, le=4294967295)]
    data: DataConfig
    cross_validation: CrossValidationConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig
    models: ModelsConfig

    @model_validator(mode="after")
    def validate_optimization_metric(self):
        if self.optimization.metric not in self.evaluation.metrics:
            raise ValueError(
                "The optimization metric must be included "
                "in evaluation.metrics."
            )

        return self


class UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}

        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)

            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings.")

            if key in mapping:
                raise ValueError(f"Duplicate YAML key: {key!r}")

            mapping[key] = self.construct_object(value_node, deep=deep)

        return mapping


def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_training_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> TrainingConfig:
    path = path if path.is_absolute() else PROJECT_ROOT / path

    with path.open(encoding="utf-8") as file:
        contents = yaml.load(file, Loader=UniqueKeySafeLoader)

    if not isinstance(contents, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    config = TrainingConfig.model_validate(contents)

    config.data.split_manifest = str(
        resolve_project_path(config.data.split_manifest)
    )
    config.data.feature_sql = str(
        resolve_project_path(config.data.feature_sql)
    )
    config.mlflow.tracking_directory = str(
        resolve_project_path(config.mlflow.tracking_directory)
    )

    for input_path in (
        config.data.split_manifest,
        config.data.feature_sql,
    ):
        if not Path(input_path).is_file():
            raise FileNotFoundError(
                f"Configured input file not found: {input_path}"
            )

    return config


def main() -> None:
    config = load_training_config()

    print("Training configuration validated.")
    print(config.model_dump_json(indent=2))


if __name__ == "__main__":
    main()