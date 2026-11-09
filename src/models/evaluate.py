"""Shared probability metrics and train/validation integrity checks."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

from src.models.pipeline import FEATURE_COLUMNS


METRIC_FUNCTIONS = {
    "average_precision": average_precision_score,
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
    "brier_score": brier_score_loss,
}


def validate_partitions(training, validation):
    required = set(FEATURE_COLUMNS) | {"customer_id", "churn", "partition"}
    for partition, frame in (("train", training), ("validation", validation)):
        if (
            frame.empty
            or not frame.columns.is_unique
            or set(frame.columns) != required
            or not frame["partition"].eq(partition).all()
        ):
            raise ValueError(f"Invalid {partition} partition or feature schema.")
        ids = frame["customer_id"]
        if not ids.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all() or not ids.is_unique:
            raise ValueError(f"Invalid {partition} customer IDs.")
        target = frame["churn"]
        if target.isna().any() or set(target.unique()) != {0, 1}:
            raise ValueError(f"{partition} must contain both binary classes.")
    if training["customer_id"].isin(validation["customer_id"]).any():
        raise ValueError("Training and validation IDs overlap.")


def score_predictions(frame, probabilities, config):
    probabilities = np.asarray(probabilities)
    if (
        probabilities.shape != (len(frame),)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid probabilities.")
    target = frame["churn"]
    metrics = {
        name: float(METRIC_FUNCTIONS[name](target, probabilities))
        for name in config.evaluation.metrics
    }
    ranked = frame[["customer_id", "churn"]].assign(
        probability=probabilities
    ).sort_values(["probability", "customer_id"], ascending=[False, True])
    for fraction in config.evaluation.targeting_fractions:
        count = int(np.ceil(fraction * len(frame)))
        captured = int(ranked.head(count)["churn"].sum())
        suffix = f"top_{100 * fraction:g}pct"
        metrics[f"precision_{suffix}"] = captured / count
        metrics[f"capture_{suffix}"] = captured / int(target.sum())
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite evaluation metrics.")
    return metrics
