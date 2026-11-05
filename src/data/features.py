from pathlib import Path

import duckdb

from src.data.loader import PROJECT_ROOT, SCHEMA, load_customers
from src.data.splits import load_split_views


FEATURE_SQL_PATH = PROJECT_ROOT / "sql" / "02_feature_engineering.sql"

METADATA_COLUMNS = ("customer_id", "churn", "partition")
EXCLUDED_COLUMNS = {"signup_date", "senior_citizen"}


def load_feature_view(
    connection: duckdb.DuckDBPyConnection,
    sql_path: Path = FEATURE_SQL_PATH,
) -> duckdb.DuckDBPyRelation:
    query = sql_path.read_text(encoding="utf-8").strip().rstrip(";")

    if not query:
        raise ValueError(f"Feature SQL is empty: {sql_path}")

    features = connection.sql(query)

    expected_columns = (
        set(SCHEMA) - EXCLUDED_COLUMNS
    ) | {"partition"}

    if (
        len(features.columns) != len(expected_columns)
        or set(features.columns) != expected_columns
    ):
        raise ValueError("Feature query does not match the baseline schema.")

    mismatches = connection.sql(f"""
        WITH features AS (
            {query}
        ),
        missing_or_changed AS (
            SELECT customer_id, churn, partition
            FROM customer_split_manifest

            EXCEPT ALL

            SELECT customer_id, churn, partition
            FROM features
        ),
        extra_or_changed AS (
            SELECT customer_id, churn, partition
            FROM features

            EXCEPT ALL

            SELECT customer_id, churn, partition
            FROM customer_split_manifest
        )
        SELECT
            (SELECT COUNT(*) FROM missing_or_changed)
            + (SELECT COUNT(*) FROM extra_or_changed)
                AS mismatches
    """).fetchall()[0][0]

    if mismatches:
        raise ValueError(
            "Feature query changed customer coverage, labels, "
            f"or assignments: {mismatches:,} unmatched metadata rows."
        )

    features.create_view("baseline_features", replace=True)

    return connection.table("baseline_features")


def main() -> None:
    with duckdb.connect() as connection:
        load_customers(connection)
        load_split_views(connection)
        features = load_feature_view(connection)

        predictor_columns = [
            column
            for column in features.columns
            if column not in METADATA_COLUMNS
        ]

        print(f"Baseline ready: {len(predictor_columns)} predictors.")
        print("Metadata columns are not model inputs.")

        print(
            connection.sql("""
                SELECT
                    partition,
                    COUNT(*) AS customers
                FROM baseline_features
                GROUP BY partition
                ORDER BY CASE partition
                    WHEN 'train' THEN 1
                    WHEN 'validation' THEN 2
                    WHEN 'test' THEN 3
                END
            """)
            .df()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()