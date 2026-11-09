from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from src.models.campaign import CampaignScenario, compare_policies
from src.models.sensitivity import advantage_figure, run_sensitivity


@pytest.fixture
def inputs():
    customers = pd.DataFrame({
        "customer_id": [f"id-{i:02d}" for i in range(20)],
        "monthlycharges": np.linspace(20, 200, 20),
    })
    return customers, np.linspace(0.7, 0.05, 20)


def test_full_grid_and_base_reproduction(inputs):
    customers, probabilities = inputs
    base = CampaignScenario(contact_cost=2.0)
    original = asdict(base)
    result = run_sensitivity(customers, probabilities, base)
    assert len(result) == 36 * 3
    assert not result.duplicated([
        "contact_cost", "prevention_probability", "retained_months", "policy"
    ]).any()
    assert set(result["contact_cost"]) == {1, 2, 5, 10}
    assert result["gross_margin"].eq(base.gross_margin).all()
    assert result["capacity_fraction"].eq(base.capacity_fraction).all()
    assert (result["selected_customers"] <= result["capacity_customers"]).all()
    assert (result["profit_minus_risk_expected_net_value"] >= -1e-8).all()
    assert asdict(base) == original
    selected = result.loc[
        (result["contact_cost"] == 2)
        & (result["prevention_probability"] == 0.2)
        & (result["retained_months"] == 6)
    ].set_index("policy")
    _, expected = compare_policies(customers, probabilities, base)
    np.testing.assert_allclose(
        selected.loc[expected["policy"], "expected_net_value"],
        expected["expected_net_value"],
    )


def test_profit_is_nondecreasing_with_effectiveness(inputs):
    result = run_sensitivity(*inputs, CampaignScenario(contact_cost=2.0))
    selected = result.loc[result["policy"] == "expected_profit_positive_only"]
    for _, group in selected.groupby(["contact_cost", "retained_months"]):
        assert (
            group.sort_values("prevention_probability")["expected_net_value"]
            .diff().dropna() >= -1e-8
        ).all()


def test_no_effect_means_no_contacts(inputs):
    result = run_sensitivity(
        *inputs, CampaignScenario(contact_cost=2.0, gross_margin=0)
    )
    assert result["selected_customers"].eq(0).all()
    assert result["expected_net_value"].eq(0).all()


def test_heatmap(inputs):
    result = run_sensitivity(*inputs, CampaignScenario(contact_cost=2.0))
    figure = advantage_figure(result)
    assert np.asarray(figure.data[0].z).shape == (3, 4)
    assert "plotly" in figure.to_html(include_plotlyjs=True)
    with pytest.raises(ValueError, match="No sensitivity"):
        advantage_figure(result, retained_months=99)
