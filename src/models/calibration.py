"""Descriptive calibration diagnostics; no calibrator fitting."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import brier_score_loss, log_loss


def calibration_report(target, probabilities, *, n_bins=10):
    target = np.asarray(target)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        target.ndim != 1
        or set(target.tolist()) != {0, 1}
        or probabilities.shape != target.shape
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Expected both binary classes and valid probabilities.")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2.")

    frame = pd.DataFrame({"target": target, "probability": probabilities})
    if np.unique(probabilities).size == 1:
        frame["bin"] = 0
    else:
        frame["bin"] = pd.qcut(
            frame["probability"], q=n_bins, labels=False, duplicates="drop"
        )
    table = frame.groupby("bin", observed=True).agg(
        customers=("target", "size"),
        churners=("target", "sum"),
        probability_min=("probability", "min"),
        probability_max=("probability", "max"),
        mean_predicted=("probability", "mean"),
        observed_rate=("target", "mean"),
    ).reset_index(drop=True)

    n = table["customers"]
    rate = table["observed_rate"]
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half_width = z * np.sqrt(
        rate * (1 - rate) / n + z * z / (4 * n * n)
    ) / denominator
    table["observed_lower_95"] = (center - half_width).clip(0, 1)
    table["observed_upper_95"] = (center + half_width).clip(0, 1)
    table["predicted_minus_observed"] = table["mean_predicted"] - rate

    metrics = {
        "mean_predicted": float(probabilities.mean()),
        "observed_churn_rate": float(target.mean()),
        "predicted_minus_observed": float(probabilities.mean() - target.mean()),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities)),
        "ece_quantile": float(
            (table["predicted_minus_observed"].abs() * n).sum() / n.sum()
        ),
        "actual_bins": len(table),
    }
    return table, metrics


def reliability_figure(table):
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=table["mean_predicted"],
        y=table["observed_rate"],
        mode="lines+markers",
        name="Validation bins",
        customdata=table[["customers", "probability_min", "probability_max"]],
        error_y={
            "type": "data", "symmetric": False,
            "array": table["observed_upper_95"] - table["observed_rate"],
            "arrayminus": table["observed_rate"] - table["observed_lower_95"],
        },
        hovertemplate=(
            "Predicted: %{x:.4f}<br>Observed: %{y:.4f}"
            "<br>Customers: %{customdata[0]}"
            "<br>Probability range: %{customdata[1]:.4f} - %{customdata[2]:.4f}"
            "<extra></extra>"
        ),
    ))
    upper = min(1.0, max(
        table["mean_predicted"].max(), table["observed_upper_95"].max()
    ) * 1.1 + 0.01)
    figure.add_trace(go.Scatter(
        x=[0, upper], y=[0, upper], mode="lines",
        line={"dash": "dash"}, name="Perfect calibration",
    ))
    figure.update_layout(
        title="CatBoost validation reliability (95% pointwise Wilson intervals)",
        xaxis_title="Mean predicted churn probability",
        yaxis_title="Observed churn fraction",
        xaxis={"range": [0, upper]}, yaxis={"range": [0, upper]},
    )
    return figure
