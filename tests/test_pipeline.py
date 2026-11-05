from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.data.features import EXCLUDED_COLUMNS, METADATA_COLUMNS
from src.data.loader import SCHEMA
from src.models.pipeline import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    build_boosting_pipeline,
)


@pytest.fixture(scope="module")
def training_data():
    rng = np.random.default_rng(42)
    rows = 600

    features = pd.DataFrame(
        {
            column: rng.uniform(1, 100, rows)
            for column in NUMERIC_COLUMNS
        }
    )

    for column in CATEGORICAL_COLUMNS:
        features[column] = rng.choice(["category_a", "category_b"], rows)

    for column in BINARY_COLUMNS:
        features[column] = rng.integers(0, 2, rows)

    features.loc[::10, "annual_income"] = np.nan
    features.loc[::13, "customer_satisfaction"] = np.nan

    target = pd.Series(
        (
            (features["monthlycharges"].to_numpy() > 65)
            & (rng.random(rows) < 0.6)
        ).astype(int),
        name="churn",
    )

    return features, target


@pytest.fixture(scope="module")
def fitted_pipeline(training_data):
    features, target = training_data
    pipeline = build_boosting_pipeline()
    pipeline.fit(features, target)
    return pipeline


def test_feature_groups_match_baseline():
    expected = (
        set(SCHEMA)
        - EXCLUDED_COLUMNS
        - set(METADATA_COLUMNS)
    )

    assert len(FEATURE_COLUMNS) == 28
    assert len(set(FEATURE_COLUMNS)) == 28
    assert set(FEATURE_COLUMNS) == expected


def test_configuration_matches_notebook():
    pipeline = build_boosting_pipeline()
    classifier = pipeline.named_steps["classifier"]

    expected = {
        "learning_rate": 0.1,
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
    }

    for parameter, value in expected.items():
        assert classifier.get_params()[parameter] == value

    preprocessor = pipeline.named_steps["preprocessor"]
    branches = {
        name: (transformer, columns)
        for name, transformer, columns in preprocessor.transformers
    }

    assert branches["numeric"] == (
        "passthrough", list(NUMERIC_COLUMNS)
    )
    assert branches["binary"] == (
        "passthrough", list(BINARY_COLUMNS)
    )

    encoder, columns = branches["categorical"]
    assert columns == list(CATEGORICAL_COLUMNS)
    assert encoder.handle_unknown == "ignore"
    assert encoder.sparse_output is False
    assert preprocessor.remainder == "drop"
    assert preprocessor.sparse_threshold == 0


def test_builder_returns_independent_unfitted_models(training_data):
    features, target = training_data
    first = build_boosting_pipeline()
    second = build_boosting_pipeline()

    assert first is not second
    assert first.named_steps["classifier"] is not second.named_steps["classifier"]
    assert first.named_steps["preprocessor"] is not second.named_steps["preprocessor"]

    first.fit(features, target)

    with pytest.raises(NotFittedError):
        second.predict_proba(features)


def test_probabilities_are_valid(fitted_pipeline, training_data):
    features, _ = training_data
    probabilities = fitted_pipeline.predict_proba(features)

    assert fitted_pipeline.classes_.tolist() == [0, 1]
    assert probabilities.shape == (len(features), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_numeric_nulls_reach_classifier(fitted_pipeline, training_data):
    features, _ = training_data
    transformed = fitted_pipeline.named_steps["preprocessor"].transform(
        features
    )

    assert isinstance(transformed, np.ndarray)

    income_index = NUMERIC_COLUMNS.index("annual_income")

    assert np.isnan(transformed[0, income_index])
    np.testing.assert_allclose(
        transformed[:, :len(NUMERIC_COLUMNS)],
        features[list(NUMERIC_COLUMNS)].to_numpy(),
        equal_nan=True,
    )


def test_unseen_category_is_supported(fitted_pipeline, training_data):
    features, _ = training_data
    unseen = features.iloc[:10].copy()
    unseen["contract"] = "previously_unseen_contract"

    probabilities = fitted_pipeline.predict_proba(unseen)

    assert probabilities.shape == (10, 2)
    assert np.isfinite(probabilities).all()


def test_metadata_does_not_affect_predictions(
    fitted_pipeline, training_data
):
    features, target = training_data
    with_metadata = features.copy()
    with_metadata["customer_id"] = [
        f"C{i:06}" for i in range(len(features))
    ]
    with_metadata["churn"] = 1 - target
    with_metadata["partition"] = "test"
    with_metadata["signup_date"] = "2099-01-01"
    with_metadata["senior_citizen"] = 1

    np.testing.assert_allclose(
        fitted_pipeline.predict_proba(features),
        fitted_pipeline.predict_proba(with_metadata),
    )


def test_missing_required_column_raises(
    fitted_pipeline, training_data
):
    features, _ = training_data

    with pytest.raises(ValueError, match="monthlycharges"):
        fitted_pipeline.predict_proba(
            features.drop(columns="monthlycharges")
        )


def test_save_load_preserves_predictions(
    fitted_pipeline, training_data, tmp_path: Path
):
    features, _ = training_data
    path = tmp_path / "pipeline.joblib"

    joblib.dump(fitted_pipeline, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(
        fitted_pipeline.predict_proba(features),
        restored.predict_proba(features),
    )