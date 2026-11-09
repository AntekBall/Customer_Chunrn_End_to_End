import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.portfolio import (
    PortfolioThresholds,
    apply_campaign_overlay,
    apply_portfolio_segments,
    fit_portfolio_thresholds,
    summarize_campaign_segments,
    summarize_portfolio_segments,
)
from src.models.campaign import (
    CampaignScenario, compare_policies, select_campaign_customers,
)


MODEL_ID = "synthetic-model-run"


@pytest.fixture
def training():
    return pd.DataFrame({
        "customer_id": [f"customer-{i:03d}" for i in range(100)],
        "monthlycharges": np.linspace(10, 200, 100),
        "churn_probability": np.linspace(0.01, 0.6, 100),
        "partition": "train",
        "churn": np.tile([0, 1], 50),
    }, index=np.repeat(np.arange(50) + 100, 2))


def segment(training):
    thresholds = fit_portfolio_thresholds(training, model_run_id=MODEL_ID)
    return apply_portfolio_segments(training, thresholds, model_run_id=MODEL_ID)


def test_training_quantiles_json_and_immutability(training, tmp_path):
    original = training.copy(deep=True)
    reference = fit_portfolio_thresholds(training, model_run_id=MODEL_ID)
    np.testing.assert_allclose(
        reference.revenue_thresholds, training["monthlycharges"].quantile([.5, .9, .99])
    )
    np.testing.assert_allclose(
        reference.risk_thresholds, training["churn_probability"].quantile([.5, .9])
    )
    path = tmp_path / "portfolio_thresholds.json"
    path.write_text(reference.model_dump_json(indent=2), encoding="utf-8")
    restored = PortfolioThresholds.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored == reference
    assert restored.reference_rows == 100
    with pytest.raises(ValidationError):
        restored.reference_rows = 5
    pd.testing.assert_frame_equal(training, original)


def test_boundaries_and_full_matrix():
    thresholds = PortfolioThresholds(
        model_run_id=MODEL_ID, reference_rows=100,
        revenue_thresholds=(50.0, 100.0, 150.0),
        risk_thresholds=(0.1, 0.2),
    )
    frame = pd.DataFrame({
        "customer_id": ["a", "b", "c", "d"],
        "monthlycharges": [50., 100., 150., 151.],
        "churn_probability": [0.1, 0.2, 0.3, 0.4],
    })
    result = apply_portfolio_segments(frame, thresholds, model_run_id=MODEL_ID)
    assert result["revenue_segment"].tolist() == [
        "Lower half", "Middle 40%", "High revenue", "Very high revenue"
    ]
    assert result["risk_segment"].tolist() == [
        "Lower risk", "Elevated risk", "Highest risk", "Highest risk"
    ]
    summary = summarize_portfolio_segments(result)
    assert len(summary) == 12
    assert summary["customers"].sum() == 4
    assert summary["monthly_charge_total"].sum() == frame["monthlycharges"].sum()
    empty = summary.loc[("Lower half", "Highest risk")]
    assert empty["customers"] == 0
    assert empty["monthly_charge_total"] == 0
    assert np.isnan(empty["mean_predicted_churn"])


def test_filter_and_row_order_do_not_change_bands(training):
    reference = fit_portfolio_thresholds(training, model_run_id=MODEL_ID)
    full = apply_portfolio_segments(training, reference, model_run_id=MODEL_ID)
    subset = training.iloc[70:].sample(frac=1, random_state=123)
    filtered = apply_portfolio_segments(subset, reference, model_run_id=MODEL_ID)
    expected = full.set_index("customer_id").loc[filtered["customer_id"]]
    assert filtered["revenue_segment"].tolist() == expected["revenue_segment"].tolist()
    assert filtered["risk_segment"].tolist() == expected["risk_segment"].tolist()
    pd.testing.assert_index_equal(filtered.index, subset.index)


def test_tied_reference_values_do_not_split_customers(training):
    training["monthlycharges"] = 100.0
    training["churn_probability"] = 0.2
    result = segment(training)
    assert result["revenue_segment"].eq("Lower half").all()
    assert result["risk_segment"].eq("Lower risk").all()
    assert summarize_portfolio_segments(result)["customers"].sum() == 100


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 1.0])
def test_overlay_reconciles_to_existing_policy(training, fraction):
    frame = segment(training)
    original = frame.copy(deep=True)
    scenario = CampaignScenario(capacity_fraction=fraction)
    overlay = apply_campaign_overlay(frame, scenario)
    summary = summarize_campaign_segments(overlay)
    scored, policies = compare_policies(
        training[["customer_id", "monthlycharges"]],
        training["churn_probability"].to_numpy(), scenario,
    )
    selected = select_campaign_customers(scored, scenario)
    policy = policies.set_index("policy").loc["expected_profit_positive_only"]
    assert set(overlay.loc[overlay["selected"], "customer_id"]) == set(selected["customer_id"])
    assert summary["customers"].sum() == len(training)
    assert summary["selected_customers"].sum() == policy["selected_customers"]
    assert summary["selected_expected_net_value"].sum() == pytest.approx(
        policy["expected_net_value"]
    )
    assert summary["eligible_customers"].sum() == (scored["expected_net_value"] > 0).sum()
    assert summary["customer_share"].sum() == pytest.approx(1)
    assert summary["monthly_charge_share"].sum() == pytest.approx(1)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_index_equal(overlay.index, frame.index)


def test_selected_ids_are_stable_with_ties_and_reordering(training):
    training["monthlycharges"] = 100.0
    training["churn_probability"] = 0.5
    frame = segment(training)
    scenario = CampaignScenario(capacity_fraction=0.1)
    for data in (frame, frame.sample(frac=1, random_state=7)):
        overlay = apply_campaign_overlay(data, scenario)
        assert sorted(overlay.loc[overlay["selected"], "customer_id"]) == [
            f"customer-{i:03d}" for i in range(10)
        ]


def test_filtering_an_overlay_does_not_reallocate_capacity(training):
    overlay = apply_campaign_overlay(segment(training), CampaignScenario())
    excluded = overlay.loc[~overlay["selected"]]
    summary = summarize_campaign_segments(excluded)
    assert summary["selected_customers"].sum() == 0
    assert summary["selected_expected_net_value"].sum() == 0
    assert summary["customers"].sum() == len(excluded)


def test_empty_filters_and_zero_charges(training):
    training["monthlycharges"] = 0.0
    frame = segment(training)
    summary = summarize_portfolio_segments(frame)
    assert summary["monthly_charge_share"].isna().all()
    assert summary["risk_weighted_monthly_charges"].eq(0).all()
    overlay = apply_campaign_overlay(frame, CampaignScenario())
    assert not overlay["eligible"].any()
    empty = summarize_campaign_segments(overlay.iloc[:0])
    assert len(empty) == 12
    assert empty["customers"].eq(0).all()
    assert empty["customer_share"].isna().all()
    assert empty["eligible_fraction"].isna().all()
    reference = fit_portfolio_thresholds(training, model_run_id=MODEL_ID)
    assert apply_portfolio_segments(
        training.iloc[:0], reference, model_run_id=MODEL_ID
    ).empty


@pytest.mark.parametrize("case", [
    "empty", "duplicate_id", "missing_id", "blank_id", "missing_column",
    "duplicate_column", "negative_charge", "infinite_charge", "null_charge",
    "string_charge", "boolean_charge", "null_probability", "large_probability",
    "negative_probability", "validation", "missing_partition",
])
def test_invalid_reference_data_raises(training, case):
    if case == "empty":
        training = training.iloc[:0]
    elif case == "duplicate_id":
        training["customer_id"] = "same"
    elif case == "missing_id":
        training.iloc[0, training.columns.get_loc("customer_id")] = None
    elif case == "blank_id":
        training["customer_id"] = " "
    elif case == "missing_column":
        training = training.drop(columns="monthlycharges")
    elif case == "duplicate_column":
        training = pd.concat([training, training[["monthlycharges"]]], axis=1)
    elif case == "negative_charge":
        training["monthlycharges"] = -1.0
    elif case == "infinite_charge":
        training["monthlycharges"] = np.inf
    elif case == "null_charge":
        training["monthlycharges"] = np.nan
    elif case == "string_charge":
        training["monthlycharges"] = "100"
    elif case == "boolean_charge":
        training["monthlycharges"] = True
    elif case == "null_probability":
        training["churn_probability"] = np.nan
    elif case == "large_probability":
        training["churn_probability"] = 1.1
    elif case == "negative_probability":
        training["churn_probability"] = -0.1
    elif case == "validation":
        training["partition"] = "validation"
    else:
        training = training.drop(columns="partition")
    with pytest.raises(ValueError):
        fit_portfolio_thresholds(training, model_run_id=MODEL_ID)


@pytest.mark.parametrize("updates", [
    {"revenue_thresholds": (100.0, 50.0, 150.0)},
    {"risk_thresholds": (0.3, 0.1)},
    {"risk_thresholds": (0.1, 1.1)},
    {"revenue_thresholds": (10.0, 20.0, np.inf)},
    {"reference_rows": 0},
    {"model_run_id": " "},
])
def test_invalid_thresholds(updates):
    fields = {
        "model_run_id": MODEL_ID, "reference_rows": 100,
        "revenue_thresholds": (50.0, 100.0, 150.0),
        "risk_thresholds": (0.1, 0.2),
    }
    with pytest.raises(ValidationError):
        PortfolioThresholds(**(fields | updates))


def test_model_mismatch_and_existing_overlay_rejected(training):
    reference = fit_portfolio_thresholds(training, model_run_id=MODEL_ID)
    with pytest.raises(ValueError, match="same model"):
        apply_portfolio_segments(training, reference, model_run_id="another")
    overlay = apply_campaign_overlay(segment(training), CampaignScenario())
    with pytest.raises(ValueError, match="already has campaign"):
        apply_campaign_overlay(overlay, CampaignScenario())
    overlay["eligible"] = False
    with pytest.raises(ValueError, match="inconsistent"):
        summarize_campaign_segments(overlay)


def test_invalid_segment_labels_rejected(training):
    frame = segment(training)
    frame["risk_segment"] = "unexpected"
    with pytest.raises(ValueError, match="risk_segment"):
        summarize_portfolio_segments(frame)


def test_labels_are_not_used(training):
    first = apply_campaign_overlay(segment(training), CampaignScenario())
    training["churn"] = 1 - training["churn"]
    second = apply_campaign_overlay(segment(training), CampaignScenario())
    pd.testing.assert_frame_equal(
        summarize_campaign_segments(first), summarize_campaign_segments(second)
    )


def test_selection_rejects_bad_scores():
    with pytest.raises(ValueError, match="columns"):
        select_campaign_customers(pd.DataFrame({"customer_id": ["a"]}), CampaignScenario())
    frame = pd.DataFrame({
        "customer_id": ["a"], "churn_probability": [0.2], "expected_net_value": [1.0],
    })
    with pytest.raises(ValueError, match="ranking"):
        select_campaign_customers(frame, CampaignScenario(), ranking="churn")
    frame["expected_net_value"] = np.nan
    with pytest.raises(ValueError, match="scores"):
        select_campaign_customers(frame, CampaignScenario())
