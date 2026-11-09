"""Descriptive revenue/risk segments and assumption-based campaign overlays.

Fit thresholds once on training scores, persist their JSON with model provenance,
and reuse them. Select campaign customers on the full population before filtering.
No fitting, scoring, file access, or plotting occurs on import.
"""

from typing import Annotated

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.campaign import (
    CampaignScenario, compare_policies, select_campaign_customers,
)


REVENUE_SEGMENTS = (
    "Lower half", "Middle 40%", "High revenue", "Very high revenue",
)
RISK_SEGMENTS = ("Lower risk", "Elevated risk", "Highest risk")
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PortfolioThresholds(BaseModel):
    """Frozen training quantiles; labels describe the reference, not each filter."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    model_run_id: Annotated[str, Field(min_length=1, pattern=r"\S")]
    reference_rows: Annotated[int, Field(gt=0)]
    revenue_thresholds: tuple[NonnegativeFloat, NonnegativeFloat, NonnegativeFloat]
    risk_thresholds: tuple[Probability, Probability]

    @model_validator(mode="after")
    def ordered_thresholds(self):
        for thresholds in (self.revenue_thresholds, self.risk_thresholds):
            if tuple(sorted(thresholds)) != thresholds:
                raise ValueError("Thresholds must be nondecreasing.")
        return self


def _validate_scores(frame: pd.DataFrame, *, allow_empty: bool = False) -> None:
    required = {"customer_id", "monthlycharges", "churn_probability"}
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError("Expected unique columns including ID, charges and probability.")
    if frame.empty and not allow_empty:
        raise ValueError("Portfolio must be nonempty.")
    ids = frame["customer_id"]
    if not ids.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all() or not ids.is_unique:
        raise ValueError("Customer IDs must be unique nonblank strings.")
    for column in ("monthlycharges", "churn_probability"):
        series = frame[column]
        if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
            raise ValueError(f"{column} must be numeric, not boolean.")
        values = series.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must be finite and nonnegative.")
        if column == "churn_probability" and (values > 1).any():
            raise ValueError("Churn probabilities must be <= 1.")


def fit_portfolio_thresholds(
    training: pd.DataFrame, *, model_run_id: str,
) -> PortfolioThresholds:
    """Use training-only scores; this does not fit or change the churn model."""
    _validate_scores(training)
    if "partition" not in training or not training["partition"].eq("train").all():
        raise ValueError("Threshold fitting requires training-only rows.")
    return PortfolioThresholds(
        model_run_id=model_run_id,
        reference_rows=len(training),
        revenue_thresholds=tuple(
            float(value) for value in training["monthlycharges"].quantile([.5, .9, .99])
        ),
        risk_thresholds=tuple(
            float(value) for value in training["churn_probability"].quantile([.5, .9])
        ),
    )


def apply_portfolio_segments(
    customers: pd.DataFrame,
    thresholds: PortfolioThresholds,
    *,
    model_run_id: str,
) -> pd.DataFrame:
    """Reuse fixed thresholds. Equal values stay together, in the lower band."""
    _validate_scores(customers, allow_empty=True)
    if model_run_id != thresholds.model_run_id:
        raise ValueError("Scores and reference thresholds must use the same model run.")
    result = customers.copy()
    for column, cuts, labels, output in (
        ("monthlycharges", thresholds.revenue_thresholds,
         REVENUE_SEGMENTS, "revenue_segment"),
        ("churn_probability", thresholds.risk_thresholds,
         RISK_SEGMENTS, "risk_segment"),
    ):
        codes = np.searchsorted(
            cuts, result[column].to_numpy(dtype=float), side="left"
        )
        result[output] = pd.Categorical.from_codes(codes, categories=labels, ordered=True)
    result["risk_weighted_monthly_charges"] = (
        result["monthlycharges"] * result["churn_probability"]
    )
    return result


def _validate_segments(frame: pd.DataFrame) -> None:
    _validate_scores(frame, allow_empty=True)
    for column, labels in (
        ("revenue_segment", REVENUE_SEGMENTS), ("risk_segment", RISK_SEGMENTS),
    ):
        if column not in frame or not frame[column].isin(labels).all():
            raise ValueError(f"Missing or invalid {column}.")


def summarize_portfolio_segments(segmented: pd.DataFrame) -> pd.DataFrame:
    """Return all 12 cells. Shares refer to the supplied (possibly filtered) rows.

    Empty cells have zero counts/totals and undefined (NaN) means.
    Charge shares are undefined when total monthly charges are zero.
    """
    _validate_segments(segmented)
    frame = segmented.assign(
        risk_weighted_monthly_charges=(
            segmented["monthlycharges"] * segmented["churn_probability"]
        )
    )
    index = pd.MultiIndex.from_product(
        [REVENUE_SEGMENTS, RISK_SEGMENTS],
        names=["revenue_segment", "risk_segment"],
    )
    result = frame.groupby(
        ["revenue_segment", "risk_segment"], observed=True
    ).agg(
        customers=("customer_id", "size"),
        monthly_charge_total=("monthlycharges", "sum"),
        mean_monthly_charge=("monthlycharges", "mean"),
        mean_predicted_churn=("churn_probability", "mean"),
        risk_weighted_monthly_charges=("risk_weighted_monthly_charges", "sum"),
    ).reindex(index)
    result["customers"] = result["customers"].fillna(0).astype("int64")
    for column in ("monthly_charge_total", "risk_weighted_monthly_charges"):
        result[column] = result[column].fillna(0.0)
    result["customer_share"] = result["customers"] / (
        len(frame) if len(frame) else np.nan
    )
    total = frame["monthlycharges"].sum()
    result["monthly_charge_share"] = result["monthly_charge_total"] / (
        total if total > 0 else np.nan
    )
    return result


def apply_campaign_overlay(
    segmented: pd.DataFrame, scenario: CampaignScenario,
) -> pd.DataFrame:
    """Apply the global profit policy once, before filtering the returned rows."""
    _validate_segments(segmented)
    scored, _ = compare_policies(
        segmented[["customer_id", "monthlycharges"]],
        segmented["churn_probability"].to_numpy(dtype=float),
        scenario,
    )
    selected = select_campaign_customers(scored, scenario)["customer_id"]
    overlay_columns = set(scored) - {"customer_id", "monthlycharges", "churn_probability"}
    overlay_columns |= {
        "eligible", "selected", "eligible_expected_net_value", "selected_expected_net_value",
    }
    if overlay_columns.intersection(segmented.columns):
        raise ValueError("Portfolio already has campaign columns; remove them before recomputing.")
    result = segmented.join(
        scored.set_index("customer_id").drop(
            columns=["monthlycharges", "churn_probability"]
        ),
        on="customer_id",
        validate="one_to_one",
    )
    result["eligible"] = result["expected_net_value"] > 0
    result["selected"] = result["customer_id"].isin(selected)
    result["eligible_expected_net_value"] = result["expected_net_value"].where(
        result["eligible"], 0.0
    )
    result["selected_expected_net_value"] = result["expected_net_value"].where(
        result["selected"], 0.0
    )
    return result


def summarize_campaign_segments(overlay: pd.DataFrame) -> pd.DataFrame:
    """Summarize existing selections without reallocating capacity after filters."""
    _validate_segments(overlay)
    required = {"eligible", "selected", "expected_net_value"}
    if not required.issubset(overlay.columns):
        raise ValueError("Expected an existing campaign overlay.")
    for column in ("eligible", "selected"):
        if not is_bool_dtype(overlay[column].dtype) or overlay[column].isna().any():
            raise ValueError(f"{column} must contain boolean flags.")
    net = overlay["expected_net_value"]
    if not is_numeric_dtype(net.dtype) or not np.isfinite(
        net.to_numpy(dtype=float, na_value=np.nan)
    ).all():
        raise ValueError("Expected finite campaign net values.")
    if not overlay["eligible"].eq(net > 0).all() or (
        overlay["selected"] & ~overlay["eligible"]
    ).any():
        raise ValueError("Campaign eligibility or selection is inconsistent.")
    frame = overlay.assign(
        eligible_expected_net_value=net.where(overlay["eligible"], 0.0),
        selected_expected_net_value=net.where(overlay["selected"], 0.0),
    )
    result = summarize_portfolio_segments(frame)
    campaign = frame.groupby(
        ["revenue_segment", "risk_segment"], observed=True
    ).agg(
        eligible_customers=("eligible", "sum"),
        selected_customers=("selected", "sum"),
        mean_expected_net_value=("expected_net_value", "mean"),
        eligible_expected_net_value=("eligible_expected_net_value", "sum"),
        selected_expected_net_value=("selected_expected_net_value", "sum"),
    ).reindex(result.index)
    for column in ("eligible_customers", "selected_customers"):
        campaign[column] = campaign[column].fillna(0).astype("int64")
    for column in ("eligible_expected_net_value", "selected_expected_net_value"):
        campaign[column] = campaign[column].fillna(0.0)
    result = result.join(campaign)
    denominator = result["customers"].replace(0, np.nan)
    result["eligible_fraction"] = result["eligible_customers"] / denominator
    result["selected_fraction"] = result["selected_customers"] / denominator
    return result
