from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Protocol

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.data.loader import NULLABLE_COLUMNS, SCHEMA
from src.models.artifacts import read_json
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
)


class ProbabilityModel(Protocol):
    classes_: np.ndarray

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelProvenance:
    model_uri: str
    source_finalist_run_id: str
    freeze_run_id: str
    final_test_run_id: str
    calibration: str = "none"


def validate_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Require IDs and exactly 28 predictors; preserve caller data and row order.

    Numeric missing values are allowed only where ingestion permits them.
    New nonblank categorical values are accepted, as in the fitted pipeline.
    """
    expected = {"customer_id", *FEATURE_COLUMNS}
    if not customers.columns.is_unique or set(customers.columns) != expected:
        missing = sorted(expected - set(customers.columns))
        extra = sorted(set(customers.columns) - expected, key=str)
        raise ValueError(f"Inference schema mismatch. Missing: {missing}; extra: {extra}")
    if customers.empty:
        raise ValueError("Inference batch must be nonempty.")
    ids = customers["customer_id"]
    if not ids.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all() or not ids.is_unique:
        raise ValueError("Customer IDs must be unique nonblank strings.")

    result = customers.loc[:, ["customer_id", *FEATURE_COLUMNS]].copy()
    for column in CATEGORICAL_COLUMNS:
        if not result[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"{column} requires nonblank strings without missing values.")

    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        series = result[column]
        missing = series.isna()
        if missing.any() and column not in NULLABLE_COLUMNS:
            raise ValueError(f"{column} does not allow missing values.")
        observed = series.loc[~missing]
        is_binary = column in BINARY_COLUMNS
        if not observed.map(
            lambda value: (
                isinstance(value, Real)
                and not isinstance(value, (bool, np.bool_))
            ) or (is_binary and isinstance(value, (bool, np.bool_)))
        ).all():
            raise ValueError(f"{column} requires numeric values, not numeric strings.")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        present = values[~missing.to_numpy()]
        if not np.isfinite(present).all() or (present < 0).any():
            raise ValueError(f"{column} requires finite nonnegative values.")
        if is_binary and not np.isin(present, [0, 1]).all():
            raise ValueError(f"{column} requires binary values.")
        if SCHEMA[column] == "BIGINT" and not np.equal(present, np.floor(present)).all():
            raise ValueError(f"{column} requires integer-valued counts.")
        result[column] = values
    return result


class ChurnPredictor:
    def __init__(
        self, model: ProbabilityModel, provenance: ModelProvenance,
    ) -> None:
        classes = np.asarray(model.classes_)
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("Expected a binary classifier with classes 0 and 1.")
        self._model = model
        self._positive_index = int(np.flatnonzero(classes == 1)[0])
        self.provenance = provenance

    def predict_batch(self, customers: pd.DataFrame) -> pd.DataFrame:
        validated = validate_customers(customers)
        probabilities = np.asarray(self._model.predict_proba(
            validated.loc[:, list(FEATURE_COLUMNS)]
        ))
        if (
            probabilities.shape != (len(validated), 2)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
        ):
            raise ValueError("Model produced invalid class probabilities.")
        result = validated[["customer_id"]].copy()
        result["churn_probability"] = probabilities[:, self._positive_index]
        result["model_run_id"] = self.provenance.source_finalist_run_id
        result["freeze_run_id"] = self.provenance.freeze_run_id
        result["final_test_run_id"] = self.provenance.final_test_run_id
        return result

    def predict_one(self, customer: Mapping[str, object]) -> dict[str, object]:
        return self.predict_batch(pd.DataFrame([dict(customer)])).iloc[0].to_dict()

    @classmethod
    def from_final_test(
        cls, tracking_directory: Path, *, final_test_run_id: str,
    ) -> "ChurnPredictor":
        """Load the selected artifact, not the latest/best run discovered dynamically.

        Pickle-based models can execute code. Use only a trusted local tracking
        database and model artifacts you generated. The input directory is the
        trust boundary; do not expose this loader as a public request parameter.
        """
        database = tracking_directory.resolve() / "mlflow.db"
        if not database.is_file():
            raise FileNotFoundError(f"MLflow database not found: {database}")
        if mlflow.active_run() is not None:
            raise RuntimeError("Load the model without an active MLflow run.")
        uri = f"sqlite:///{database.as_posix()}"
        client = MlflowClient(tracking_uri=uri)
        final = client.get_run(final_test_run_id)
        if (
            final.info.status != "FINISHED"
            or final.data.tags.get("mode") != "final_test"
            or final.data.tags.get("evaluation_partition") != "test"
        ):
            raise ValueError("Expected a completed frozen final-test run.")
        freeze_id = final.data.tags["freeze_run_id"]
        frozen = client.get_run(freeze_id)
        if (
            frozen.info.status != "FINISHED"
            or frozen.data.tags.get("mode") != "frozen_model_policy"
        ):
            raise ValueError("Expected a completed freeze run.")
        manifest = read_json(client, final_test_run_id, "frozen_manifest.json")
        original = read_json(client, freeze_id, "frozen_manifest.json")
        if manifest != original:
            raise ValueError("Final-test manifest differs from the frozen setup.")
        if (
            manifest["calibration"] != "none"
            or manifest["source_finalist_run_id"]
            != final.data.tags["source_finalist_run_id"]
        ):
            raise ValueError("Unsupported or inconsistent frozen model selection.")
        source_id = manifest["source_finalist_run_id"]
        source = client.get_run(source_id)
        if (
            source.info.status != "FINISHED"
            or source.data.tags.get("model") != "catboost"
            or source.data.tags.get("evaluation_partition") != "validation"
            or float(source.data.params["positive_class_weight"]) != 1.0
        ):
            raise ValueError("Expected the selected unweighted CatBoost finalist.")
        provenance = ModelProvenance(
            model_uri=manifest["model_uri"],
            source_finalist_run_id=source_id,
            freeze_run_id=freeze_id,
            final_test_run_id=final_test_run_id,
        )
        previous_uri = mlflow.get_tracking_uri()
        try:
            mlflow.set_tracking_uri(uri)
            model = mlflow.sklearn.load_model(provenance.model_uri)
        finally:
            mlflow.set_tracking_uri(previous_uri)
        return cls(model, provenance)
