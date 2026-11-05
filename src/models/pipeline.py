from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_COLUMNS = (
    "age",
    "annual_income",
    "dependents",
    "tenure",
    "monthlycharges",
    "totalcharges",
    "num_services",
    "customer_satisfaction",
    "num_complaints",
    "num_service_calls",
    "late_payments",
    "avg_monthly_gb",
    "days_since_last_interaction",
    "credit_score",
)

CATEGORICAL_COLUMNS = (
    "gender",
    "education",
    "marital_status",
    "contract",
    "payment_method",
)

BINARY_COLUMNS = (
    "paperless_billing",
    "has_phone_service",
    "has_internet_service",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "has_tech_support",
    "has_streaming_tv",
    "has_streaming_movies",
)

FEATURE_COLUMNS = (
    NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS
)


def build_boosting_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_COLUMNS)),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(CATEGORICAL_COLUMNS),
            ),
            ("binary", "passthrough", list(BINARY_COLUMNS)),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.1,
                    max_iter=150,
                    max_leaf_nodes=15,
                    min_samples_leaf=100,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]
    )