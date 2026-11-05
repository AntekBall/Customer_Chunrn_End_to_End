import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.exceptions import NotFittedError

from src.config import DEFAULT_CONFIG_PATH, TrainingConfig
from src.models.model_builder import (
    MODEL_NAMES,
    build_model_pipeline,
    prepare_catboost_features,
)
from src.models.pipeline import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_boosting_pipeline,
)


@pytest.fixture
def config():
    contents = yaml.safe_load(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    )

    for model in contents["models"].values():
        model["enabled"] = True

    return TrainingConfig.model_validate(contents)


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(42)
    rows = 300

    features = pd.DataFrame(
        {
            column: rng.uniform(1, 100, rows)
            for column in NUMERIC_COLUMNS
        }
    )

    for column in CATEGORICAL_COLUMNS:
        features[column] = rng.choice(["a", "b"], rows)

    for column in BINARY_COLUMNS:
        features[column] = rng.integers(0, 2, rows)

    features.loc[::10, "annual_income"] = np.nan
    features.loc[::13, "customer_satisfaction"] = np.nan

    target = np.tile([0, 0, 0, 0, 1], rows // 5)

    return features, target


@pytest.fixture(params=MODEL_NAMES)
def fitted_model(request, config, synthetic_data):
    model_name = request.param
    pipeline = build_model_pipeline(model_name, config)

    iteration_parameter = {
        "hist_gradient_boosting": "max_iter",
        "catboost": "iterations",
    }.get(model_name, "n_estimators")

    pipeline.set_params(
        **{f"classifier__{iteration_parameter}": 10}
    )

    features, target = synthetic_data
    pipeline.fit(features, target)

    return pipeline


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fixed_parameters_and_seed_are_applied(config, model_name):
    config.random_seed = 123

    settings = getattr(config.models, model_name)

    if model_name in {"lightgbm", "xgboost", "random_forest"}:
        settings.fixed_parameters.n_jobs = 1
    elif model_name == "catboost":
        settings.fixed_parameters.thread_count = 1

    pipeline = build_model_pipeline(model_name, config)
    parameters = pipeline.named_steps["classifier"].get_params()

    for name, value in settings.fixed_parameters.model_dump().items():
        assert parameters[name] == value

    seed_parameter = (
        "random_seed" if model_name == "catboost" else "random_state"
    )
    assert parameters[seed_parameter] == 123


def test_histogram_defaults_match_existing_baseline(config):
    actual = build_model_pipeline(
        "hist_gradient_boosting", config
    )
    expected = build_boosting_pipeline()

    assert (
        actual.named_steps["classifier"].get_params()
        == expected.named_steps["classifier"].get_params()
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_disabled_model_raises(config, model_name):
    getattr(config.models, model_name).enabled = False

    with pytest.raises(ValueError, match="Model is disabled"):
        build_model_pipeline(model_name, config)


def test_unknown_model_raises(config):
    with pytest.raises(ValueError, match="Unknown model"):
        build_model_pipeline("unknown_model", config)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_builder_returns_fresh_unfitted_pipelines(
    config, model_name, synthetic_data
):
    first = build_model_pipeline(model_name, config)
    second = build_model_pipeline(model_name, config)

    assert first is not second
    assert first.named_steps["classifier"] is not second.named_steps["classifier"]
    assert first.named_steps["preprocessor"] is not second.named_steps["preprocessor"]

    features, _ = synthetic_data

    with pytest.raises(NotFittedError):
        second.predict_proba(features)


def test_probabilities_with_numeric_nulls(fitted_model, synthetic_data):
    features, _ = synthetic_data
    probabilities = fitted_model.predict_proba(features)

    assert fitted_model.classes_.tolist() == [0, 1]
    assert probabilities.shape == (len(features), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_unseen_category_is_supported(fitted_model, synthetic_data):
    features, _ = synthetic_data
    unseen = features.iloc[:10].copy()
    unseen["contract"] = "previously_unseen_contract"

    probabilities = fitted_model.predict_proba(unseen)

    assert probabilities.shape == (10, 2)
    assert np.isfinite(probabilities).all()


def test_metadata_is_excluded(fitted_model, synthetic_data):
    features, target = synthetic_data
    extra = features.copy()
    extra["customer_id"] = "CUSTOMER"
    extra["churn"] = 1 - target
    extra["partition"] = "test"
    extra["signup_date"] = "2099-01-01"
    extra["senior_citizen"] = 1

    np.testing.assert_allclose(
        fitted_model.predict_proba(features),
        fitted_model.predict_proba(extra),
    )


def test_save_load_preserves_predictions(
    fitted_model, synthetic_data, tmp_path
):
    features, _ = synthetic_data
    path = tmp_path / "model.joblib"

    joblib.dump(fitted_model, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(
        fitted_model.predict_proba(features),
        restored.predict_proba(features),
    )


def test_catboost_preparation_preserves_input(synthetic_data):
    features, _ = synthetic_data
    original = features.copy(deep=True)

    prepared = prepare_catboost_features(features)

    pd.testing.assert_frame_equal(features, original)
    assert prepared.columns.tolist() == list(FEATURE_COLUMNS)
    assert prepared.index.equals(features.index)

    for column in CATEGORICAL_COLUMNS:
        assert prepared[column].map(
            lambda value: isinstance(value, str)
        ).all()

    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        assert prepared[column].dtype == np.dtype("float64")

    assert prepared["annual_income"].isna().equals(
        features["annual_income"].isna()
    )


def test_catboost_rejects_missing_category(synthetic_data):
    features, _ = synthetic_data
    invalid = features.copy()
    invalid.loc[0, "contract"] = None

    with pytest.raises(ValueError, match="missing categorical values"):
        prepare_catboost_features(invalid)