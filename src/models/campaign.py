"""Assumption-based retention economics, not measured campaign effects."""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignScenario:
    contact_cost: float = 2.0
    gross_margin: float = 0.2
    retained_months: float = 6.0
    prevention_probability: float = 0.2
    capacity_fraction: float = 0.1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        for name in ("gross_margin", "prevention_probability", "capacity_fraction"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1.")
        if self.retained_months <= 0:
            raise ValueError("retained_months must be positive.")


def select_campaign_customers(
    scored: pd.DataFrame,
    scenario: CampaignScenario,
    *,
    ranking: Literal["expected_net_value", "churn_probability"] = "expected_net_value",
) -> pd.DataFrame:
    """Select from the full scored population, before any dashboard filtering."""
    if ranking not in {"expected_net_value", "churn_probability"}:
        raise ValueError(f"Unsupported campaign ranking: {ranking}")
    required = {"customer_id", "churn_probability", "expected_net_value"}
    if not scored.columns.is_unique or not required.issubset(scored.columns):
        raise ValueError("Expected campaign-scored columns.")
    ids = scored["customer_id"]
    if not ids.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all() or not ids.is_unique:
        raise ValueError("Customer IDs must be unique nonblank strings.")
    numeric = scored[["churn_probability", "expected_net_value"]].to_numpy(dtype=float)
    if (
        not np.isfinite(numeric).all()
        or (numeric[:, 0] < 0).any()
        or (numeric[:, 0] > 1).any()
    ):
        raise ValueError("Invalid campaign scores.")
    capacity = int(np.floor(scenario.capacity_fraction * len(scored)))
    return scored.loc[scored["expected_net_value"] > 0].sort_values(
        [ranking, "customer_id"], ascending=[False, True]
    ).head(capacity)


def compare_policies(customers, probabilities, scenario):
    if customers.empty:
        raise ValueError("Customer data must be nonempty.")
    if not customers.columns.is_unique or not {
        "customer_id", "monthlycharges"
    }.issubset(customers.columns):
        raise ValueError("Expected unique columns including IDs and monthlycharges.")
    ids = customers["customer_id"]
    if not ids.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all() or not ids.is_unique:
        raise ValueError("Customer IDs must be unique nonblank strings.")
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.shape != (len(customers),)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid probabilities.")
    charges = customers["monthlycharges"].to_numpy(dtype=float)
    if not np.isfinite(charges).all() or (charges < 0).any():
        raise ValueError("Monthly charges must be finite and nonnegative.")

    scored = customers[["customer_id", "monthlycharges"]].copy()
    scored["churn_probability"] = probabilities
    scored["retained_contribution_if_prevented"] = (
        charges * scenario.gross_margin * scenario.retained_months
    )
    scored["expected_prevented_churn"] = (
        probabilities * scenario.prevention_probability
    )
    scored["expected_retained_contribution"] = (
        scored["expected_prevented_churn"]
        * scored["retained_contribution_if_prevented"]
    )
    scored["expected_net_value"] = (
        scored["expected_retained_contribution"] - scenario.contact_cost
    )
    if not np.isfinite(scored.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Scenario calculations overflowed.")

    capacity = int(np.floor(scenario.capacity_fraction * len(customers)))
    selections = {"no_contact": scored.iloc[:0]}
    for policy, ranking in (
        ("risk_positive_only", "churn_probability"),
        ("expected_profit_positive_only", "expected_net_value"),
    ):
        selections[policy] = select_campaign_customers(
            scored, scenario, ranking=ranking
        )

    results = []
    for policy, selected in selections.items():
        cost = float(len(selected) * scenario.contact_cost)
        contribution = float(selected["expected_retained_contribution"].sum())
        net = contribution - cost
        results.append({
            "policy": policy,
            "capacity_customers": capacity,
            "selected_customers": len(selected),
            "selected_fraction": len(selected) / len(customers),
            "unused_capacity": capacity - len(selected),
            "expected_churners_contacted": float(selected["churn_probability"].sum()),
            "expected_prevented_churn": float(selected["expected_prevented_churn"].sum()),
            "expected_retained_contribution": contribution,
            "campaign_cost": cost,
            "expected_net_value": net,
            "scenario_roi": net / cost if cost > 0 else None,
        })
    return scored, pd.DataFrame(results)
