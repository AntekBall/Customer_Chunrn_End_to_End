"""Reusable scenario sensitivity helpers.

Each scenario reselects customers under its own profitability assumptions.
These are not measured returns or robustness estimates for one fixed audience.
"""

from dataclasses import asdict, replace
from itertools import product

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.models.campaign import CampaignScenario, compare_policies


CONTACT_COSTS = (1.0, 2.0, 5.0, 10.0)
PREVENTION_PROBABILITIES = (0.1, 0.2, 0.3)
RETAINED_MONTHS = (3.0, 6.0, 12.0)


def run_sensitivity(customers, probabilities, base: CampaignScenario):
    results = []
    for cost, prevention, months in product(
        CONTACT_COSTS, PREVENTION_PROBABILITIES, RETAINED_MONTHS
    ):
        scenario = replace(
            base,
            contact_cost=cost,
            prevention_probability=prevention,
            retained_months=months,
        )
        _, summary = compare_policies(customers, probabilities, scenario)
        reference = summary.set_index("policy")
        risk = reference.loc["risk_positive_only", "expected_net_value"]
        profit = reference.loc["expected_profit_positive_only", "expected_net_value"]
        if profit < risk - 1e-8:
            raise RuntimeError("Profit ranking underperformed risk on expected value.")
        if profit < -1e-8:
            raise RuntimeError("Positive-only selection produced negative net value.")
        assumptions = asdict(scenario)
        for row in summary.to_dict(orient="records"):
            results.append({
                **assumptions,
                **row,
                "profit_minus_risk_expected_net_value": float(profit - risk),
            })
    return pd.DataFrame(results)


def advantage_figure(results, *, retained_months=6.0):
    selected = results.loc[
        (results["policy"] == "expected_profit_positive_only")
        & (results["retained_months"] == retained_months)
    ]
    if selected.empty:
        raise ValueError("No sensitivity results for the requested retained months.")
    grid = selected.pivot(
        index="prevention_probability",
        columns="contact_cost",
        values="profit_minus_risk_expected_net_value",
    ).sort_index().sort_index(axis=1)
    if grid.isna().any().any() or not np.isfinite(grid.to_numpy()).all():
        raise ValueError("Incomplete or non-finite sensitivity grid.")
    figure = go.Figure(go.Heatmap(
        z=grid.to_numpy(),
        x=[str(value) for value in grid.columns],
        y=[f"{100 * value:g}%" for value in grid.index],
        colorscale="Blues",
        texttemplate="%{z:,.0f}",
        colorbar={"title": "Expected value gain"},
        hovertemplate=(
            "Cost/contact: %{x}<br>Assumed prevention: %{y}"
            "<br>Profit minus risk expected net value: %{z:,.2f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title=(
            f"Scenario-based profit-ranking advantage: {retained_months:g} months"
            "<br><sup>Audiences reselected per scenario; not measured campaign ROI</sup>"
        ),
        xaxis_title="Total cost per contacted customer",
        yaxis_title="Assumed churn-prevention probability",
    )
    return figure
