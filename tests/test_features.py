from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from src.data.features import FEATURE_SQL_PATH, load_feature_view
from src.data.loader import NULLABLE_COLUMNS, SCHEMA


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    sample_values = {
        "VARCHAR": "'example'",
        "BIGINT": "1",
        "DOUBLE": "10.0",
        "BOOLEAN": "TRUE",
        "TIMESTAMP": "TIMESTAMP '2024-01-01'",
    }

    expressions = []

    for column, dtype in SCHEMA.items():
        value = (
            column
            if column in {"customer_id", "churn"}
            else sample_values[dtype]
        )

        if column in NULLABLE_COLUMNS:
            value = (
                f"CASE WHEN customer_id = 'C001' "
                f"THEN NULL ELSE {value} END"
            )

        expressions.append(f'CAST({value} AS {dtype}) AS "{column}"')

    with duckdb.connect() as database:
        database.sql(
            "CREATE TABLE raw_customers AS SELECT "
            + ", ".join(expressions)
            + """
                FROM (
                    VALUES
                        ('C001', 0),
                        ('C002', 1),
                        ('C003', 0),
                        ('C004', 1),
                        ('C005', 0),
                        ('C006', 1)
                ) AS seed(customer_id, churn)
            """
        )

        database.sql("""
            CREATE TABLE customer_split_manifest AS
            SELECT
                customer_id,
                churn,
                CASE
                    WHEN customer_id <= 'C002' THEN 'train'
                    WHEN customer_id <= 'C004' THEN 'validation'
                    ELSE 'test'
                END AS partition
            FROM raw_customers
        """)

        yield database


@pytest.fixture
def baseline_sql() -> str:
    return FEATURE_SQL_PATH.read_text(
        encoding="utf-8"
    ).strip().rstrip(";")


def write_query(tmp_path: Path, query: str) -> Path:
    path = tmp_path / "features.sql"
    path.write_text(query, encoding="utf-8")
    return path


def test_baseline_schema_and_predictor_count(connection):
    features = load_feature_view(connection)

    expected_columns = (
        set(SCHEMA) - {"signup_date", "senior_citizen"}
    ) | {"partition"}

    assert len(features.columns) == 31
    assert set(features.columns) == expected_columns

    predictors = set(features.columns) - {
        "customer_id", "churn", "partition"
    }
    assert len(predictors) == 28


def test_preserves_ids_labels_and_partitions(connection):
    load_feature_view(connection)

    actual = connection.sql("""
        SELECT customer_id, churn, partition
        FROM baseline_features
        ORDER BY customer_id
    """).fetchall()

    assert actual == [
        ("C001", 0, "train"),
        ("C002", 1, "train"),
        ("C003", 0, "validation"),
        ("C004", 1, "validation"),
        ("C005", 0, "test"),
        ("C006", 1, "test"),
    ]


def test_preserves_all_selected_source_values(connection):
    load_feature_view(connection)

    selected_columns = [
        column
        for column in SCHEMA
        if column not in {"signup_date", "senior_citizen"}
    ]
    selection = ", ".join(f'"{column}"' for column in selected_columns)

    expected = connection.sql(f"""
        SELECT {selection}
        FROM raw_customers
        ORDER BY customer_id
    """).fetchall()

    actual = connection.sql(f"""
        SELECT {selection}
        FROM baseline_features
        ORDER BY customer_id
    """).fetchall()

    assert actual == expected


@pytest.mark.parametrize("column", sorted(NULLABLE_COLUMNS))
def test_preserves_allowed_nulls(connection, column):
    load_feature_view(connection)

    assert connection.sql(f"""
        SELECT "{column}"
        FROM baseline_features
        WHERE customer_id = 'C001'
    """).fetchall() == [(None,)]

    assert connection.sql(f"""
        SELECT COUNT(*)
        FROM baseline_features
        WHERE "{column}" IS NULL
    """).fetchall() == [(1,)]


def test_repeated_loading_is_supported(connection):
    load_feature_view(connection)
    load_feature_view(connection)

    assert connection.sql("""
        SELECT COUNT(*)
        FROM baseline_features
    """).fetchall() == [(6,)]


@pytest.mark.parametrize(
    "change",
    [
        "drop_row",
        "duplicate_row",
        "change_id",
        "change_label",
        "change_partition",
    ],
)
def test_rejects_changed_metadata(
    connection, tmp_path, baseline_sql, change
):
    source = f"({baseline_sql}) AS baseline"

    queries = {
        "drop_row": (
            f"SELECT * FROM {source} WHERE customer_id <> 'C001'"
        ),
        "duplicate_row": (
            f"SELECT * FROM {source} "
            f"UNION ALL "
            f"SELECT * FROM {source} WHERE customer_id = 'C001'"
        ),
        "change_id": (
            "SELECT * REPLACE ('UNKNOWN' AS customer_id) "
            f"FROM {source}"
        ),
        "change_label": (
            f"SELECT * REPLACE (1 - churn AS churn) FROM {source}"
        ),
        "change_partition": (
            "SELECT * REPLACE ('train' AS partition) "
            f"FROM {source}"
        ),
    }

    path = write_query(tmp_path, queries[change])

    with pytest.raises(ValueError, match="unmatched metadata rows"):
        load_feature_view(connection, path)

    assert connection.sql("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = 'baseline_features'
    """).fetchall() == [(0,)]


@pytest.mark.parametrize("change", ["missing_column", "extra_column"])
def test_rejects_changed_schema(
    connection, tmp_path, baseline_sql, change
):
    if change == "missing_column":
        query = f"SELECT * EXCLUDE (age) FROM ({baseline_sql})"
    else:
        query = (
            "SELECT *, 1 AS unexpected "
            f"FROM ({baseline_sql})"
        )

    path = write_query(tmp_path, query)

    with pytest.raises(ValueError, match="baseline schema"):
        load_feature_view(connection, path)


def test_empty_sql_raises(connection, tmp_path):
    path = write_query(tmp_path, " \n ")

    with pytest.raises(ValueError, match="Feature SQL is empty"):
        load_feature_view(connection, path)


def test_missing_sql_file_raises(connection, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_view(connection, tmp_path / "missing.sql")


def test_invalid_sql_raises(connection, tmp_path):
    path = write_query(tmp_path, "SELECT FROM")

    with pytest.raises(duckdb.ParserException):
        load_feature_view(connection, path)