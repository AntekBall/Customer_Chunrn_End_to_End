from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from catboost import CatBoostClassifier, Pool
from sklearn.pipeline import Pipeline

from src.models.pipeline import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from src.models.predict import ChurnPredictor, ModelProvenance, validate_customers


@dataclass(frozen=True)
class CustomerExplanation:
    customer_id: str
    churn_probability: float
    base_log_odds: float
    raw_log_odds: float
    contributions: pd.DataFrame
    provenance: ModelProvenance


@dataclass(frozen=True)
class GlobalExplanation:
    importance: pd.DataFrame
    sampled_customer_ids: tuple[str, ...]
    population_rows: int
    random_seed: int
    provenance: ModelProvenance


class ChurnExplainer:
    """Reuse the predictor's fitted pipeline; do not fit or modify it."""

    def __init__(self, predictor: ChurnPredictor) -> None:
        model = predictor.model
        if (
            not isinstance(model, Pipeline)
            or list(model.named_steps) != ["preprocessor", "classifier"]
            or not isinstance(model.named_steps["classifier"], CatBoostClassifier)
        ):
            raise ValueError("Native explanations require the fitted CatBoost pipeline.")
        if (
            predictor.provenance.calibration != "none"
            or not np.array_equal(model.classes_, [0, 1])
        ):
            raise ValueError("Expected uncalibrated CatBoost with class order [0, 1].")
        classifier = model.named_steps["classifier"]
        if classifier.feature_names_ != list(FEATURE_COLUMNS):
            raise ValueError("CatBoost feature names/order differ from the 28 predictors.")
        self._predictor = predictor
        self._pipeline = model
        self._classifier = classifier

    def _explain_batch(
        self, customers: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        validated = validate_customers(customers)
        features = validated.loc[:, list(FEATURE_COLUMNS)]
        transformed = self._pipeline.named_steps["preprocessor"].transform(features)
        if (
            not isinstance(transformed, pd.DataFrame)
            or list(transformed.columns) != list(FEATURE_COLUMNS)
            or not transformed.index.equals(features.index)
        ):
            raise ValueError("Preprocessing changed feature order or row alignment.")
        pool = Pool(transformed, cat_features=list(CATEGORICAL_COLUMNS))
        shap = np.asarray(self._classifier.get_feature_importance(
            pool, type="ShapValues", thread_count=1,
        ), dtype=float)
        if shap.shape != (len(validated), len(FEATURE_COLUMNS) + 1):
            raise ValueError("Unexpected native SHAP output shape.")
        raw = np.asarray(
            self._classifier.predict(pool, prediction_type="RawFormulaVal", thread_count=1),
            dtype=float,
        )
        probabilities = self._predictor.predict_batch(customers)[
            "churn_probability"
        ].to_numpy(dtype=float)
        if (
            raw.shape != (len(validated),)
            or not np.isfinite(shap).all()
            or not np.isfinite(raw).all()
            or not np.allclose(shap.sum(axis=1), raw, rtol=1e-6, atol=1e-8)
        ):
            raise ValueError("SHAP values failed raw-output reconstruction.")
        # Stable sigmoid: exp never receives a positive argument.
        exp_negative_abs = np.exp(-np.abs(raw))
        reconstructed = np.where(
            raw >= 0, 1 / (1 + exp_negative_abs),
            exp_negative_abs / (1 + exp_negative_abs),
        )
        if not np.allclose(reconstructed, probabilities, rtol=1e-6, atol=1e-8):
            raise ValueError("Explained raw output differs from predictor probabilities.")
        return validated, shap, raw, probabilities

    def explain_one(self, customer: dict[str, object]) -> CustomerExplanation:
        validated, shap, raw, probabilities = self._explain_batch(
            pd.DataFrame([customer]),
        )
        contributions = pd.DataFrame({
            "feature": list(FEATURE_COLUMNS),
            "feature_value": validated.iloc[0].loc[list(FEATURE_COLUMNS)].tolist(),
            "shap_log_odds": shap[0, :-1],
        })
        return CustomerExplanation(
            customer_id=str(validated.iloc[0]["customer_id"]),
            churn_probability=float(probabilities[0]),
            base_log_odds=float(shap[0, -1]),
            raw_log_odds=float(raw[0]),
            contributions=contributions,
            provenance=self._predictor.provenance,
        )

    def global_importance(
        self, training: pd.DataFrame, *, sample_size: int = 1000, random_seed: int = 42,
    ) -> GlobalExplanation:
        """Explain at most sample_size training customers; never use churn labels.

        Accept exactly inference columns plus partition. Sample size is capped at
        5000 to bound native SHAP work; use this offline, not on every API request.
        Validate all source rows before sampling so invalid input cannot hide.
        """
        expected = {"customer_id", "partition", *FEATURE_COLUMNS}
        if (
            not training.columns.is_unique or set(training.columns) != expected
            or not training["partition"].eq("train").all()
        ):
            raise ValueError("Global importance requires training-only rows without labels.")
        if (
            isinstance(sample_size, bool) or not isinstance(sample_size, int)
            or not 1 <= sample_size <= 5000
        ):
            raise ValueError("sample_size must be an integer between 1 and 5000.")
        if (
            isinstance(random_seed, bool) or not isinstance(random_seed, int)
            or not 0 <= random_seed < 2**32
        ):
            raise ValueError("random_seed must be an integer in [0, 2**32).")
        inputs = validate_customers(training.drop(columns="partition"))
        # Sorting IDs makes seeded sampling independent of incoming row order.
        sample = inputs.sort_values("customer_id").sample(
            n=min(sample_size, len(inputs)), random_state=random_seed,
        )
        _, shap, _, _ = self._explain_batch(sample)
        mean_abs = np.abs(shap[:, :-1]).mean(axis=0)
        importance = pd.DataFrame({
            "feature": list(FEATURE_COLUMNS),
            "mean_abs_shap_log_odds": mean_abs,
            "mean_shap_log_odds": shap[:, :-1].mean(axis=0),
        })
        total = mean_abs.sum()
        importance["importance_share"] = mean_abs / total if total > 0 else np.nan
        importance = importance.sort_values(
            ["mean_abs_shap_log_odds", "feature"], ascending=[False, True],
        ).reset_index(drop=True)
        return GlobalExplanation(
            importance=importance,
            sampled_customer_ids=tuple(sample["customer_id"]),
            population_rows=len(inputs),
            random_seed=random_seed,
            provenance=self._predictor.provenance,
        )


def waterfall_figure(
    explanation: CustomerExplanation, *, top_n: int = 10,
) -> go.Figure:
    """Show largest absolute contributions; combine omitted features, never drop them."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 28:
        raise ValueError("top_n must be an integer between 1 and 28.")
    contributions = explanation.contributions
    order = contributions["shap_log_odds"].abs().sort_values(
        ascending=False, kind="stable",
    ).index
    ordered = contributions.loc[order]
    kept = ordered.iloc[:top_n]
    labels = ["Baseline", *kept["feature"].tolist()]
    values = [explanation.base_log_odds, *kept["shap_log_odds"].tolist()]
    if len(ordered) > top_n:
        labels.append(f"Other {len(ordered) - top_n} features")
        values.append(float(ordered.iloc[top_n:]["shap_log_odds"].sum()))
    measures = ["absolute"] + ["relative"] * (len(values) - 1) + ["total"]
    figure = go.Figure(go.Waterfall(
        x=[*labels, "Model output"], y=[*values, 0],
        measure=measures,
        increasing={"marker": {"color": "#c0392b"}},
        decreasing={"marker": {"color": "#2471a3"}},
    ))
    figure.update_layout(
        title=f"Churn explanation | predicted probability {explanation.churn_probability:.1%}",
        yaxis_title="Raw log-odds of churn (not probability points)",
        xaxis_title="Model contributions, not causal effects",
        showlegend=False,
    )
    return figure
