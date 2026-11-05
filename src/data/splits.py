from pathlib import Path

import duckdb
import pandas as pd

from src.data.loader import PROJECT_ROOT, load_customers


SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_split_seed42_v1.parquet"
)

PARTITIONS = ("train", "validation", "test")


def load_split_views(
    connection: duckdb.DuckDBPyConnection,
    path: Path = SPLIT_PATH,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    manifest = pd.read_parquet(path)

    if (
        not manifest.columns.is_unique
        or set(manifest.columns) != {"customer_id", "churn", "partition"}
    ):
        raise ValueError(
            "Expected exactly customer_id, churn, and partition columns."
        )

    if manifest.empty or manifest.isna().any().any():
        raise ValueError("Split manifest must be nonempty and contain no nulls.")

    if not manifest["customer_id"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Manifest customer IDs must be nonblank strings.")

    if manifest["customer_id"].duplicated().any():
        raise ValueError("Each customer must have exactly one assignment.")

    if not manifest["churn"].isin([0, 1]).all():
        raise ValueError("Manifest churn labels must be binary.")

    if not manifest["partition"].isin(PARTITIONS).all():
        raise ValueError("Manifest contains an unknown partition.")

    if set(manifest["partition"]) != set(PARTITIONS):
        raise ValueError("Manifest must contain all three partitions.")

    current_customers = connection.sql("""
        SELECT customer_id, churn
        FROM raw_customers
    """).df()

    comparison = current_customers.merge(
        manifest,
        on="customer_id",
        how="outer",
        suffixes=("_current", "_saved"),
        indicator=True,
        validate="one_to_one",
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError(
            "Manifest IDs do not exactly match the loaded dataset."
        )

    if not comparison["churn_current"].eq(
        comparison["churn_saved"]
    ).all():
        raise ValueError(
            "Saved churn labels differ from the loaded dataset."
        )

    connection.register("customer_split_manifest", manifest)

    for partition in PARTITIONS:
        connection.sql(f"""
            CREATE OR REPLACE TEMP VIEW {partition}_customers AS
            SELECT customers.*
            FROM raw_customers AS customers
            INNER JOIN customer_split_manifest AS assignments
                ON customers.customer_id = assignments.customer_id
            WHERE assignments.partition = '{partition}'
        """)

    summary = (
        manifest.groupby("partition")
        .agg(
            customers=("customer_id", "size"),
            churn_rate=("churn", "mean"),
        )
        .reindex(PARTITIONS)
    )

    summary["churn_percentage"] = 100 * summary.pop("churn_rate")

    return summary


def main() -> None:
    with duckdb.connect() as connection:
        load_customers(connection)
        summary = load_split_views(connection)

        print("Saved assignments validated and partition views created.")
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()