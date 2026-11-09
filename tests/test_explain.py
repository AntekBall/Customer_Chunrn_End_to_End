from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from tests.helpers import model_config
from src.models.explain import ChurnExplainer, waterfall_figure
from src.models.model_builder import build_model_pipeline
from src.models.pipeline import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
)
from src.models.predict import ChurnPredictor, ModelProvenance


@pytest.fixture(scope="module")
def customers():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        column: rng.integers(1, 10, 40).astype(float) for column in NUMERIC_COLUMNS
    })
    for column in CATEGORICAL_COLUMNS:
        frame[column] = rng.choice(["A", "B"], len(frame))
    for column in BINARY_COLUMNS:
        frame[column] = rng.integers(0, 2, len(frame))
    frame["paperless_billing"] = frame["paperless_billing"].astype(bool)
    frame["annual_income"] = np.nan
    frame["customer_id"] = [f"customer-{i:02}" for i in range(len(frame))]
    return frame


@pytest.fixture(scope="module")
def predictor(customers):
    pipeline = build_model_pipeline("catboost", model_config())
    pipeline.set_params(
        classifier__iterations=8, classifier__depth=2, classifier__thread_count=1,
    )
    with threadpool_limits(limits=1):
        pipeline.fit(customers[list(FEATURE_COLUMNS)], customers["age"].gt(4).astype(int))
    return ChurnPredictor(
        pipeline, ModelProvenance("model", "source", "freeze", "final"),
    )


def test_local_reconstruction_and_no_mutation(customers, predictor):
    before = customers.copy(deep=True)
    explainer = ChurnExplainer(predictor)
    customer = customers.iloc[0].to_dict()
    customer["contract"] = "previously-unseen"
    local = explainer.explain_one(customer)
    assert local.customer_id == customer["customer_id"]
    assert local.contributions["feature"].tolist() == list(FEATURE_COLUMNS)
    assert pd.isna(local.contributions.set_index("feature").loc["annual_income", "feature_value"])
    assert (
        local.base_log_odds + local.contributions["shap_log_odds"].sum()
    ) == pytest.approx(local.raw_log_odds, abs=1e-8)
    assert 1 / (1 + np.exp(-local.raw_log_odds)) == pytest.approx(local.churn_probability)
    assert local.churn_probability == pytest.approx(predictor.predict_one(customer)["churn_probability"])
    assert local.provenance == predictor.provenance
    pd.testing.assert_frame_equal(customers, before)


def test_all_global_features_and_reproducibility(customers, predictor):
    explainer = ChurnExplainer(predictor)
    training = customers.assign(partition="train")
    first = explainer.global_importance(training, sample_size=7)
    second = explainer.global_importance(training.iloc[::-1], sample_size=7)
    assert first.sampled_customer_ids == second.sampled_customer_ids
    assert first.population_rows == len(training)
    assert len(first.sampled_customer_ids) == 7
    assert len(first.importance) == 28
    assert set(first.importance["feature"]) == set(FEATURE_COLUMNS)
    assert first.importance["importance_share"].sum() == pytest.approx(1)
    assert first.importance["mean_abs_shap_log_odds"].ge(0).all()
    assert first.importance["mean_abs_shap_log_odds"].is_monotonic_decreasing
    pd.testing.assert_frame_equal(first.importance, second.importance)
    assert first.provenance == predictor.provenance
    small = explainer.global_importance(training.iloc[:3], sample_size=100)
    assert len(small.sampled_customer_ids) == 3


@pytest.mark.parametrize("top_n", [1, 10, 28])
def test_waterfall_preserves_omitted_contributions(customers, predictor, top_n):
    local = ChurnExplainer(predictor).explain_one(customers.iloc[0].to_dict())
    figure = waterfall_figure(local, top_n=top_n)
    trace = figure.data[0]
    assert trace.measure[0] == "absolute"
    assert trace.measure[-1] == "total"
    assert sum(trace.y[:-1]) == pytest.approx(local.raw_log_odds)
    assert len(trace.y) == top_n + 2 + (top_n < 28)
    assert "log-odds" in figure.layout.yaxis.title.text
    figure.to_json()


@pytest.mark.parametrize("value", [0, -1, 29, True, 1.5])
def test_invalid_top_n(customers, predictor, value):
    local = ChurnExplainer(predictor).explain_one(customers.iloc[0].to_dict())
    with pytest.raises(ValueError):
        waterfall_figure(local, top_n=value)


@pytest.mark.parametrize("case", [
    "test", "validation", "mixed", "label", "empty", "duplicate_id",
    "bad_unsampled_row", "invalid_size", "oversize", "bool_size",
    "negative_seed", "large_seed", "bool_seed",
])
def test_global_validation(customers, predictor, case):
    training = customers.assign(partition="train")
    kwargs = {"sample_size": 2}
    if case in {"test", "validation"}:
        training["partition"] = case
    elif case == "mixed":
        training.loc[0, "partition"] = "test"
    elif case == "label":
        training["churn"] = 0
    elif case == "empty":
        training = training.iloc[:0]
    elif case == "duplicate_id":
        training["customer_id"] = "same"
    elif case == "bad_unsampled_row":
        training.loc[39, "age"] = -1
    elif case == "invalid_size":
        kwargs["sample_size"] = 0
    elif case == "oversize":
        kwargs["sample_size"] = 5001
    elif case == "bool_size":
        kwargs["sample_size"] = True
    elif case == "negative_seed":
        kwargs["random_seed"] = -1
    elif case == "large_seed":
        kwargs["random_seed"] = 2**32
    elif case == "bool_seed":
        kwargs["random_seed"] = True
    with pytest.raises(ValueError):
        ChurnExplainer(predictor).global_importance(training, **kwargs)


def test_wrong_model_and_calibration(predictor):
    class OtherModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, frame):
            return np.tile([.9, .1], (len(frame), 1))

    with pytest.raises(ValueError, match="CatBoost pipeline"):
        ChurnExplainer(ChurnPredictor(OtherModel(), predictor.provenance))
    with pytest.raises(ValueError, match="uncalibrated"):
        ChurnExplainer(ChurnPredictor(
            predictor.model, replace(predictor.provenance, calibration="sigmoid"),
        ))


@pytest.mark.parametrize("case", ["shape", "nan", "raw", "probability"])
def test_shap_integrity_guards(customers, predictor, monkeypatch, case):
    classifier = predictor.model.named_steps["classifier"]
    original = classifier.get_feature_importance
    if case == "probability":
        def changed(frame):
            result = ChurnPredictor.predict_batch(predictor, frame)
            result["churn_probability"] = .99
            return result
        monkeypatch.setattr(predictor, "predict_batch", changed)
        expected = "predictor probabilities"
    else:
        def corrupt(*args, **kwargs):
            values = original(*args, **kwargs)
            if case == "shape":
                return values[:, :-1]
            values[:, 0] = np.nan if case == "nan" else 999.
            return values
        monkeypatch.setattr(classifier, "get_feature_importance", corrupt)
        expected = "shape" if case == "shape" else "reconstruction"
    with pytest.raises(ValueError, match=expected):
        ChurnExplainer(predictor).explain_one(customers.iloc[0].to_dict())


def test_loaded_artifact_roundtrip(tmp_path, customers, predictor):
    import cloudpickle

    path = tmp_path / "synthetic-pipeline.pkl"
    path.write_bytes(cloudpickle.dumps(predictor.model))
    restored = ChurnPredictor(cloudpickle.loads(path.read_bytes()), predictor.provenance)
    first = ChurnExplainer(predictor).explain_one(customers.iloc[0].to_dict())
    second = ChurnExplainer(restored).explain_one(customers.iloc[0].to_dict())
    pd.testing.assert_frame_equal(first.contributions, second.contributions)
    assert first.raw_log_odds == second.raw_log_odds
