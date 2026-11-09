import numpy as np
import pytest

from src.models.calibration import calibration_report, reliability_figure


def test_constant_probabilities():
    target = np.tile([0, 0, 0, 1], 25)
    table, metrics = calibration_report(target, np.full(100, 0.25))
    assert len(table) == 1
    assert table["customers"].sum() == 100
    assert table["churners"].sum() == 25
    assert metrics["ece_quantile"] == 0
    assert metrics["predicted_minus_observed"] == 0
    assert metrics["brier_score"] == pytest.approx(0.1875)
    assert table.loc[0, "observed_lower_95"] < 0.25
    assert table.loc[0, "observed_upper_95"] > 0.25


def test_quantiles_counts_and_plot():
    probabilities = np.linspace(0.01, 0.4, 100)
    target = np.tile([0, 1], 50)
    table, metrics = calibration_report(target, probabilities)
    assert len(table) == 10
    assert table["customers"].eq(10).all()
    assert table["churners"].sum() == target.sum()
    assert metrics["mean_predicted"] == pytest.approx(0.205)
    assert metrics["predicted_minus_observed"] == pytest.approx(-0.295)
    assert (table["observed_lower_95"] <= table["observed_rate"]).all()
    assert (table["observed_upper_95"] >= table["observed_rate"]).all()
    assert "plotly" in reliability_figure(table).to_html(include_plotlyjs=True)


def test_ties_are_not_split():
    table, _ = calibration_report(
        np.tile([0, 1], 50), np.repeat([0.1, 0.2], 50)
    )
    assert table["customers"].sum() == 100
    assert len(table) <= 2
    assert all(
        row.probability_min == row.probability_max
        for row in table.itertuples()
    )


@pytest.mark.parametrize("target,probabilities", [
    ([], []),
    ([0, 0], [0.1, 0.2]),
    ([0, 2], [0.1, 0.2]),
    ([0, 1], [float("nan"), 0.2]),
    ([0, 1], [-0.1, 1.1]),
    ([0, 1], [[0.1], [0.2]]),
])
def test_invalid_inputs(target, probabilities):
    with pytest.raises(ValueError):
        calibration_report(target, probabilities)


@pytest.mark.parametrize("n_bins", [True, 1, 2.5])
def test_invalid_bin_counts(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        calibration_report([0, 1], [0.1, 0.2], n_bins=n_bins)
