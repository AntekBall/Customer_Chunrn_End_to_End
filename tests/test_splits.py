from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.data.splits import PARTITIONS, load_split_views


@pytest.fixture
def connection() -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect() as database:
        database.sql("""
            CREATE TABLE raw_customers AS
            SELECT *
            FROM (
                VALUES
                    ('C001', 0, 50.0),
                    ('C002', 1, 60.0),
                    ('C003', 0, 70.0),
                    ('C004', 1, 80.0),
                    ('C005', 0, 90.0),
                    ('C006', 1, 100.0)
            ) AS customers(customer_id, churn, monthlycharges)
        """)
        yield database


@pytest.fixture
def manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [
                "C001", "C002", "C003", "C004", "C005", "C006"
            ],
            "churn": [0, 1, 0, 1, 0, 1],
            "partition": [
                "train", "train",
                "validation", "validation",
                "test", "test",
            ],
        }
    )


def save_manifest(tmp_path: Path, manifest: pd.DataFrame) -> Path:
    path = tmp_path / "split.parquet"
    manifest.to_parquet(path, index=False)
    return path


def test_loads_saved_assignments(connection, tmp_path, manifest):
    path = save_manifest(tmp_path, manifest)
    original_bytes = path.read_bytes()

    summary = load_split_views(connection, path)

    assert summary.index.tolist() == list(PARTITIONS)
    assert summary["customers"].tolist() == [2, 2, 2]
    assert summary["churn_percentage"].tolist() == [50.0, 50.0, 50.0]

    for partition in PARTITIONS:
        actual = connection.sql(f"""
            SELECT customer_id, churn
            FROM {partition}_customers
            ORDER BY customer_id
        """).fetchall()

        expected = list(
            manifest.loc[
                manifest["partition"] == partition,
                ["customer_id", "churn"],
            ].itertuples(index=False, name=None)
        )

        assert actual == expected

    assert connection.sql("""
        SELECT monthlycharges
        FROM train_customers
        ORDER BY customer_id
    """).fetchall() == [(50.0,), (60.0,)]

    assert path.read_bytes() == original_bytes


def test_repeated_loading_is_supported(connection, tmp_path, manifest):
    path = save_manifest(tmp_path, manifest)

    first = load_split_views(connection, path)
    second = load_split_views(connection, path)

    pd.testing.assert_frame_equal(first, second)

    assert connection.sql("""
        SELECT COUNT(*)
        FROM train_customers
    """).fetchall() == [(2,)]


def test_missing_manifest_raises(connection, tmp_path):
    with pytest.raises(FileNotFoundError, match="Split manifest not found"):
        load_split_views(connection, tmp_path / "missing.parquet")


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("customer_id", "C002", "exactly one assignment"),
        ("customer_id", "UNKNOWN", "IDs do not exactly match"),
        ("customer_id", "   ", "nonblank strings"),
        ("churn", 1, "Saved churn labels differ"),
        ("churn", 2, "labels must be binary"),
        ("partition", "holdout", "unknown partition"),
        ("partition", None, "contain no nulls"),
    ],
)
def test_invalid_manifest_raises(
    connection, tmp_path, manifest, column, value, message
):
    manifest.loc[0, column] = value
    path = save_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match=message):
        load_split_views(connection, path)


def test_missing_customer_assignment_raises(connection, tmp_path, manifest):
    manifest = manifest.iloc[1:].copy()
    path = save_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="IDs do not exactly match"):
        load_split_views(connection, path)


def test_missing_partition_raises(connection, tmp_path, manifest):
    manifest.loc[manifest["partition"] == "test", "partition"] = "train"
    path = save_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="all three partitions"):
        load_split_views(connection, path)


@pytest.mark.parametrize("change", ["missing_column", "extra_column"])
def test_invalid_schema_raises(connection, tmp_path, manifest, change):
    if change == "missing_column":
        manifest = manifest.drop(columns="partition")
    else:
        manifest["unexpected"] = "value"

    path = save_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="Expected exactly"):
        load_split_views(connection, path)


def test_empty_manifest_raises(connection, tmp_path, manifest):
    path = save_manifest(tmp_path, manifest.iloc[:0])

    with pytest.raises(ValueError, match="must be nonempty"):
        load_split_views(connection, path)