import csv
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from src.data.loader import (
    BINARY_COLUMNS,
    NULLABLE_COLUMNS,
    SCHEMA,
    load_customers,
)


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect() as database:
        yield database


@pytest.fixture
def valid_row() -> dict[str, object]:
    return {
        "customer_id": "CUST0001",
        "signup_date": "2024-01-15 12:00:00",
        "age": 35,
        "gender": "Female",
        "annual_income": 60000.0,
        "education": "bachelor",
        "marital_status": "single",
        "dependents": 0,
        "tenure": 12,
        "contract": "one_year",
        "payment_method": "credit_card",
        "paperless_billing": True,
        "senior_citizen": 0,
        "monthlycharges": 80.0,
        "totalcharges": 960.0,
        "num_services": 2,
        "has_phone_service": 1,
        "has_internet_service": 1,
        "has_online_security": 0,
        "has_online_backup": 0,
        "has_device_protection": 0,
        "has_tech_support": 0,
        "has_streaming_tv": 0,
        "has_streaming_movies": 0,
        "customer_satisfaction": 7.0,
        "num_complaints": 0.0,
        "num_service_calls": 1,
        "late_payments": 0,
        "avg_monthly_gb": 40.0,
        "days_since_last_interaction": 10,
        "credit_score": 700.0,
        "churn": 0,
    }


def write_csv(
    tmp_path: Path,
    rows: list[dict[str, object]],
    columns: list[str] | None = None,
) -> Path:
    path = tmp_path / "customers.csv"

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(SCHEMA) if columns is None else columns,
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def test_loads_valid_customers(connection, tmp_path, valid_row):
    second_row = {
        **valid_row,
        "customer_id": "CUST0002",
        "churn": 1,
    }
    path = write_csv(tmp_path, [valid_row, second_row])
    original_bytes = path.read_bytes()

    count = load_customers(connection, path)

    assert count == 2
    assert connection.sql("""
        SELECT customer_id, churn
        FROM raw_customers
        ORDER BY customer_id
    """).fetchall() == [
        ("CUST0001", 0),
        ("CUST0002", 1),
    ]
    assert path.read_bytes() == original_bytes

    schema = connection.sql("DESCRIBE raw_customers").fetchall()
    assert {row[0]: row[1] for row in schema} == SCHEMA


def test_missing_file_raises(connection, tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_customers(connection, tmp_path / "missing.csv")


def test_missing_column_raises(connection, tmp_path, valid_row):
    del valid_row["age"]
    columns = [column for column in SCHEMA if column != "age"]
    path = write_csv(tmp_path, [valid_row], columns)

    with pytest.raises(ValueError, match="Missing columns: .*age"):
        load_customers(connection, path)


def test_unexpected_column_raises(connection, tmp_path, valid_row):
    valid_row["unexpected"] = "value"
    path = write_csv(
        tmp_path,
        [valid_row],
        [*SCHEMA, "unexpected"],
    )

    with pytest.raises(ValueError, match="Unexpected columns: .*unexpected"):
        load_customers(connection, path)


def test_duplicate_column_names_raise(connection, tmp_path, valid_row):
    path = write_csv(tmp_path, [valid_row], [*SCHEMA, "age"])

    with pytest.raises(ValueError, match="duplicate column names"):
        load_customers(connection, path)


def test_reordered_columns_are_accepted(connection, tmp_path, valid_row):
    path = write_csv(tmp_path, [valid_row], list(reversed(SCHEMA)))

    assert load_customers(connection, path) == 1
    assert connection.sql(
        "SELECT customer_id, age FROM raw_customers"
    ).fetchall() == [("CUST0001", 35)]


def test_duplicate_customer_ids_raise(connection, tmp_path, valid_row):
    path = write_csv(tmp_path, [valid_row, valid_row.copy()])

    with pytest.raises(ValueError, match="duplicate_customer_ids: 1"):
        load_customers(connection, path)


@pytest.mark.parametrize(
    ("column", "value", "violation"),
    [
        ("customer_id", None, "customer_id_missing"),
        ("customer_id", "   ", "blank_customer_ids"),
        ("age", None, "age_missing"),
        ("churn", None, "churn_missing"),
        ("monthlycharges", -1.0, "monthlycharges_invalid_numeric"),
        ("tenure", -1, "tenure_invalid_numeric"),
        ("annual_income", float("inf"), "annual_income_invalid_numeric"),
        ("credit_score", float("-inf"), "credit_score_invalid_numeric"),
    ],
)
def test_invalid_values_raise(
    connection, tmp_path, valid_row, column, value, violation
):
    valid_row[column] = value
    path = write_csv(tmp_path, [valid_row])

    with pytest.raises(ValueError, match=violation):
        load_customers(connection, path)

    assert connection.sql("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'raw_customers'
    """).fetchall() == [(0,)]


@pytest.mark.parametrize("column", BINARY_COLUMNS)
def test_binary_columns_reject_two(
    connection, tmp_path, valid_row, column
):
    valid_row[column] = 2
    path = write_csv(tmp_path, [valid_row])

    with pytest.raises(ValueError, match=f"{column}_invalid_binary"):
        load_customers(connection, path)


@pytest.mark.parametrize("column", sorted(NULLABLE_COLUMNS))
def test_allowed_nulls_are_preserved(
    connection, tmp_path, valid_row, column
):
    valid_row[column] = None
    path = write_csv(tmp_path, [valid_row])

    assert load_customers(connection, path) == 1
    assert connection.sql(
        f'SELECT "{column}" FROM raw_customers'
    ).fetchall() == [(None,)]


def test_invalid_numeric_text_raises(connection, tmp_path, valid_row):
    valid_row["age"] = "not-a-number"
    path = write_csv(tmp_path, [valid_row])

    with pytest.raises(duckdb.Error):
        load_customers(connection, path)


def test_header_only_csv_raises(connection, tmp_path):
    path = write_csv(tmp_path, [])

    with pytest.raises(ValueError, match="no customer rows"):
        load_customers(connection, path)