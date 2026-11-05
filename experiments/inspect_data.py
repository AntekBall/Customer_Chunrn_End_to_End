from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "customer-churn-1m"
    / "customer_churn_1M.csv"
)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    with duckdb.connect() as connection:
        connection.read_csv(
            str(DATA_PATH),
            header=True,
            sample_size=-1,
        ).create_view("customers")

        row_count = connection.sql(
            "SELECT COUNT(*) FROM customers"
        ).fetchall()[0][0]

        print(f"\nFILE: {DATA_PATH.name}")
        print(f"ROWS: {row_count:,}")

        print("\nCOLUMN NAMES AND INFERRED TYPES")
        print(
            connection.sql("DESCRIBE customers")
            .df()
            .to_string(index=False)
        )

        print("\nFIRST FIVE ROWS")
        print(
            connection.sql("SELECT * FROM customers LIMIT 5")
            .df()
            .to_string(index=False)
        )

        print("\nCOLUMN STATISTICS AND NULL PERCENTAGES")
        print(
            connection.sql("SUMMARIZE customers")
            .df()
            .to_string(index=False)
        )

        print("\nCUSTOMER ID QUALITY")
        print(
            connection.sql("""
                SELECT
                    COUNT(*) AS total_rows,
                    COUNT(DISTINCT customer_id) AS unique_customer_ids,
                    COUNT(*) FILTER (
                        WHERE customer_id IS NULL
                    ) AS missing_customer_ids,
                    COUNT(customer_id)
                        - COUNT(DISTINCT customer_id)
                        AS duplicate_non_null_ids
                FROM customers
            """)
            .df()
            .to_string(index=False)
        )

        print("\nCHURN LABEL DISTRIBUTION")
        print(
            connection.sql("""
                SELECT
                    churn,
                    COUNT(*) AS customers,
                    ROUND(
                        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
                        2
                    ) AS percentage
                FROM customers
                GROUP BY churn
                ORDER BY churn
            """)
            .df()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()