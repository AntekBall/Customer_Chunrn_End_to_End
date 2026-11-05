import csv
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "customer-churn-1m"
    / "customer_churn_1M.csv"
)

SCHEMA = {
    "customer_id": "VARCHAR",
    "signup_date": "TIMESTAMP",
    "age": "BIGINT",
    "gender": "VARCHAR",
    "annual_income": "DOUBLE",
    "education": "VARCHAR",
    "marital_status": "VARCHAR",
    "dependents": "BIGINT",
    "tenure": "BIGINT",
    "contract": "VARCHAR",
    "payment_method": "VARCHAR",
    "paperless_billing": "BOOLEAN",
    "senior_citizen": "BIGINT",
    "monthlycharges": "DOUBLE",
    "totalcharges": "DOUBLE",
    "num_services": "BIGINT",
    "has_phone_service": "BIGINT",
    "has_internet_service": "BIGINT",
    "has_online_security": "BIGINT",
    "has_online_backup": "BIGINT",
    "has_device_protection": "BIGINT",
    "has_tech_support": "BIGINT",
    "has_streaming_tv": "BIGINT",
    "has_streaming_movies": "BIGINT",
    "customer_satisfaction": "DOUBLE",
    "num_complaints": "DOUBLE",
    "num_service_calls": "BIGINT",
    "late_payments": "BIGINT",
    "avg_monthly_gb": "DOUBLE",
    "days_since_last_interaction": "BIGINT",
    "credit_score": "DOUBLE",
    "churn": "BIGINT",
}

NULLABLE_COLUMNS = {
    "annual_income",
    "customer_satisfaction",
    "num_complaints",
    "avg_monthly_gb",
    "credit_score",
}

BINARY_COLUMNS = (
    "senior_citizen",
    "has_phone_service",
    "has_internet_service",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "has_tech_support",
    "has_streaming_tv",
    "has_streaming_movies",
    "churn",
)


def validate_header(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as source:
        columns = next(csv.reader(source), [])

    if len(columns) != len(set(columns)):
        raise ValueError("CSV contains duplicate column names.")

    missing = sorted(set(SCHEMA) - set(columns))
    unexpected = sorted(set(columns) - set(SCHEMA))

    if missing or unexpected:
        raise ValueError(
            f"CSV schema mismatch. "
            f"Missing columns: {missing}. "
            f"Unexpected columns: {unexpected}."
        )


def validate_customers(customers: duckdb.DuckDBPyRelation) -> int:
    checks = {
        "duplicate_customer_ids": (
            "COUNT(customer_id) - COUNT(DISTINCT customer_id)"
        ),
        "blank_customer_ids": (
            "COUNT(*) FILTER (WHERE TRIM(customer_id) = '')"
        ),
    }

    for column, dtype in SCHEMA.items():
        if column not in NULLABLE_COLUMNS:
            checks[f"{column}_missing"] = (
                f'COUNT(*) FILTER (WHERE "{column}" IS NULL)'
            )

        if dtype in {"BIGINT", "DOUBLE"}:
            checks[f"{column}_invalid_numeric"] = (
                f'COUNT(*) FILTER ('
                f'WHERE "{column}" < 0 OR NOT isfinite("{column}"))'
            )

    for column in BINARY_COLUMNS:
        checks[f"{column}_invalid_binary"] = (
            f'COUNT(*) FILTER (WHERE "{column}" NOT IN (0, 1))'
        )

    expressions = ["COUNT(*) AS row_count"]
    expressions.extend(
        f'{expression} AS "{name}"'
        for name, expression in checks.items()
    )

    result = customers.aggregate(", ".join(expressions)).fetchall()[0]
    row_count = result[0]

    if row_count == 0:
        raise ValueError("Dataset contains no customer rows.")

    violations = [
        f"{name}: {count:,}"
        for name, count in zip(checks, result[1:], strict=True)
        if count > 0
    ]

    if violations:
        raise ValueError(
            "Customer data validation failed:\n"
            + "\n".join(f"  - {violation}" for violation in violations)
        )

    return row_count


def load_customers(
    connection: duckdb.DuckDBPyConnection,
    path: Path = DATA_PATH,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    validate_header(path)

    customers = connection.read_csv(
        str(path),
        header=True,
        dtype=SCHEMA,
    )

    row_count = validate_customers(customers)
    customers.create("raw_customers")

    return row_count


def main() -> None:
    with duckdb.connect() as connection:
        row_count = load_customers(connection)

        print(f"Validation passed: {row_count:,} customer rows.")
        print("Loaded into the in-memory raw_customers table.")

        print("\nMISSING VALUES")
        print(
            connection.sql("""
                SELECT column_name, null_percentage
                FROM (SUMMARIZE raw_customers)
                ORDER BY null_percentage DESC, column_name
            """)
            .df()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()