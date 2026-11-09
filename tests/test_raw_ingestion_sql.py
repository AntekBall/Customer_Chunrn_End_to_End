import csv

import duckdb
import pytest

from src.data.loader import PROJECT_ROOT, SCHEMA, load_customers


AUDIT_SQL = PROJECT_ROOT / "sql" / "01_raw_ingestion.sql"


@pytest.fixture
def connection():
    with duckdb.connect() as database:
        columns = ", ".join(
            f'"{name}" {dtype}' for name, dtype in SCHEMA.items()
        )
        database.execute(f"CREATE TABLE raw_customers ({columns})")
        yield database


def insert_rows(connection, rows):
    placeholders = ", ".join("?" for _ in SCHEMA)
    connection.executemany(
        f"INSERT INTO raw_customers VALUES ({placeholders})",
        [[row.get(name) for name in SCHEMA] for row in rows],
    )


def audit(connection):
    result = connection.sql(AUDIT_SQL.read_text(encoding="utf-8"))
    assert result.columns == ["section", "column_name", "metric", "value"]
    rows = result.fetchall()
    values = {
        (section, column, metric): value
        for section, column, metric, value in rows
    }
    assert len(values) == len(rows)
    return values


def test_counts_missingness_and_numeric_statistics(connection):
    insert_rows(connection, [
        {"customer_id": "a", "churn": 0, "age": 20, "annual_income": 100.0},
        {"customer_id": "b", "churn": 1, "age": 40, "annual_income": None},
        {"customer_id": "c", "churn": 0, "age": 60, "annual_income": 300.0},
    ])
    before = connection.sql("SELECT * FROM raw_customers").fetchall()
    before_schema = connection.sql("DESCRIBE raw_customers").fetchall()
    values = audit(connection)

    assert values["overview", None, "row_count"] == 3
    assert values["overview", None, "distinct_customer_ids"] == 3
    assert values["overview", None, "retained_customers"] == 2
    assert values["overview", None, "churned_customers"] == 1
    assert values["churn", "churn", "observed_churn_rate"] == pytest.approx(1 / 3)
    assert values["missingness", "annual_income", "missing_count"] == 1
    assert values["missingness", "annual_income", "missing_fraction"] == pytest.approx(1 / 3)
    assert values["numeric", "annual_income", "mean"] == 200
    assert values["numeric", "age", "minimum"] == 20
    assert values["numeric", "age", "mean"] == 40
    assert values["numeric", "age", "maximum"] == 60
    assert values["numeric", "credit_score", "mean"] is None
    assert {
        column for section, column, metric in values
        if section == "missingness" and metric == "missing_count"
    } == set(SCHEMA)
    assert {
        column for section, column, metric in values if section == "numeric"
    } == {name for name, dtype in SCHEMA.items() if dtype in {"BIGINT", "DOUBLE"}}
    assert connection.sql("SELECT * FROM raw_customers").fetchall() == before
    assert connection.sql("DESCRIBE raw_customers").fetchall() == before_schema
    assert audit(connection) == values
    assert connection.sql("SHOW TABLES").fetchall() == [("raw_customers",)]


def test_id_and_label_diagnostics_on_unvalidated_table(connection):
    insert_rows(connection, [
        {"customer_id": "duplicate", "churn": 0},
        {"customer_id": "duplicate", "churn": 1},
        {"customer_id": "   ", "churn": 2},
        {"customer_id": None, "churn": None},
    ])
    values = audit(connection)
    assert values["overview", None, "duplicate_id_excess_rows"] == 1
    assert values["overview", None, "missing_customer_ids"] == 1
    assert values["overview", None, "blank_customer_ids"] == 1
    assert values["overview", None, "invalid_churn_labels"] == 1
    assert values["overview", None, "missing_churn_labels"] == 1


def test_empty_table_has_zero_counts_and_undefined_rates(connection):
    values = audit(connection)
    assert values["overview", None, "row_count"] == 0
    assert values["churn", "churn", "observed_churn_rate"] is None
    for column in SCHEMA:
        assert values["missingness", column, "missing_count"] == 0
        assert values["missingness", column, "missing_fraction"] is None
    assert values["numeric", "monthlycharges", "mean"] is None


def test_requires_loaded_table():
    with duckdb.connect() as connection:
        with pytest.raises(duckdb.CatalogException, match="raw_customers"):
            audit(connection)


def test_audit_runs_after_real_loader_on_synthetic_csv(tmp_path):
    row = {
        name: (
            "customer-1" if name == "customer_id"
            else "2024-01-01" if name == "signup_date"
            else "example" if dtype == "VARCHAR"
            else True if dtype == "BOOLEAN"
            else 0
        )
        for name, dtype in SCHEMA.items()
    }
    row["annual_income"] = None
    row["monthlycharges"] = 80.0
    path = tmp_path / "customers.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SCHEMA))
        writer.writeheader()
        writer.writerow(row)
    with duckdb.connect() as connection:
        assert load_customers(connection, path) == 1
        values = audit(connection)
        assert values["overview", None, "row_count"] == 1
        assert values["numeric", "monthlycharges", "mean"] == 80
        assert values["missingness", "annual_income", "missing_count"] == 1
