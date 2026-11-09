import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data.loader import PROJECT_ROOT
from src.data.portfolio import PortfolioThresholds
from src.models.artifacts import file_sha256
from src.models.pipeline import FEATURE_COLUMNS


SUMMARY_SQL = PROJECT_ROOT / "sql" / "03_portfolio_summary.sql"


class PortfolioFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    partition: Literal["train", "validation", "test"] | None = None
    revenue_segment: Literal[
        "Lower half", "Middle 40%", "High revenue", "Very high revenue",
    ] | None = None
    risk_segment: Literal["Lower risk", "Elevated risk", "Highest risk"] | None = None
    contract: str | None = Field(default=None, min_length=1, pattern=r"\S")
    payment_method: str | None = Field(default=None, min_length=1, pattern=r"\S")
    has_internet_service: bool | None = None


@dataclass(frozen=True)
class PortfolioSummary:
    overview: pd.DataFrame
    segments: pd.DataFrame


def _validate_snapshot(
    connection: duckdb.DuckDBPyConnection, manifest: dict,
    thresholds: PortfolioThresholds,
) -> None:
    required = {
        "customer_id", "partition", *FEATURE_COLUMNS, "churn_probability",
        "model_run_id", "freeze_run_id", "final_test_run_id", "revenue_segment",
        "risk_segment", "risk_weighted_monthly_charges",
    }
    if set(connection.table("portfolio").columns) != required:
        raise ValueError("Unexpected scored portfolio schema.")
    if (
        manifest["schema_version"] != 1
        or manifest["purpose"] != "descriptive_portfolio_not_model_evaluation"
        or manifest["contains_churn_labels"] is not False
        or manifest["contains_campaign_selection"] is not False
        or manifest["training_scores_are_in_sample"] is not True
        or manifest["threshold_reference"] != "train"
        or manifest["model"]["calibration"] != "none"
        or thresholds.model_run_id != manifest["model"]["source_finalist_run_id"]
    ):
        raise ValueError("Unsupported or inconsistent snapshot manifest.")
    rows, unique_ids, invalid = connection.sql("""
        SELECT COUNT(*), COUNT(DISTINCT customer_id),
          COUNT(*) FILTER (WHERE
            customer_id IS NULL OR trim(customer_id) = ''
            OR partition IS NULL OR partition NOT IN ('train', 'validation', 'test')
            OR monthlycharges IS NULL OR NOT isfinite(monthlycharges) OR monthlycharges < 0
            OR churn_probability IS NULL OR NOT isfinite(churn_probability)
            OR churn_probability NOT BETWEEN 0 AND 1
            OR has_internet_service IS NULL OR has_internet_service NOT IN (0, 1)
            OR model_run_id IS DISTINCT FROM ?
            OR freeze_run_id IS DISTINCT FROM ?
            OR final_test_run_id IS DISTINCT FROM ?
            OR revenue_segment IS DISTINCT FROM CASE
                WHEN monthlycharges <= ? THEN 'Lower half'
                WHEN monthlycharges <= ? THEN 'Middle 40%'
                WHEN monthlycharges <= ? THEN 'High revenue'
                ELSE 'Very high revenue' END
            OR risk_segment IS DISTINCT FROM CASE
                WHEN churn_probability <= ? THEN 'Lower risk'
                WHEN churn_probability <= ? THEN 'Elevated risk'
                ELSE 'Highest risk' END
            OR risk_weighted_monthly_charges IS NULL
            OR NOT isfinite(risk_weighted_monthly_charges)
            OR abs(risk_weighted_monthly_charges - monthlycharges * churn_probability)
                > 1e-8 * greatest(1, abs(monthlycharges * churn_probability))
          )
        FROM portfolio
    """, params=[
        thresholds.model_run_id, manifest["model"]["freeze_run_id"],
        manifest["model"]["final_test_run_id"],
        *thresholds.revenue_thresholds, *thresholds.risk_thresholds,
    ]).fetchall()[0]
    counts = dict(connection.sql(
        "SELECT partition, COUNT(*) FROM portfolio GROUP BY partition"
    ).fetchall())
    if (
        not rows or rows != manifest["rows"] or unique_ids != rows or invalid
        or counts != manifest["partition_rows"]
        or counts.get("train", 0) != thresholds.reference_rows
        or manifest["includes_source_test_customers"] is not (counts.get("test", 0) > 0)
    ):
        raise ValueError("Snapshot rows, values, segments or provenance are inconsistent.")


@contextmanager
def open_portfolio(directory: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Verify local checksums, then copy Parquet into an isolated in-memory table.

    Checksums detect accidental changes; they do not authenticate untrusted input.
    No original CSV, model files or MLflow database are needed.
    """
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    paths = [directory / name for name in ("customers.parquet", "thresholds.json")]
    for path in paths:
        if file_sha256(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {path.name}")
    thresholds = PortfolioThresholds.model_validate_json(
        paths[1].read_text(encoding="utf-8")
    )
    with duckdb.connect() as connection:
        connection.sql(
            "CREATE TABLE portfolio AS SELECT * FROM read_parquet(?)",
            params=[str(paths[0])],
        )
        for path in paths:
            if file_sha256(path) != manifest["files"][path.name]["sha256"]:
                raise ValueError(f"Snapshot changed during loading: {path.name}")
        _validate_snapshot(connection, manifest, thresholds)
        yield connection


def summarize_portfolio(
    connection: duckdb.DuckDBPyConnection,
    filters: PortfolioFilters | None = None,
    *,
    sql_path: Path = SUMMARY_SQL,
) -> PortfolioSummary:
    """Return one overview row and all 12 ordered cells, including empty cells.

    Undefined means/shares are NaN; empty counts and totals are zero.
    mean_predicted_churn is not observed churn or an accuracy metric.
    """
    if filters is None:
        filters = PortfolioFilters()
    clauses = []
    parameters = []
    for column, value in filters.model_dump(exclude_none=True).items():
        clauses.append(f'"{column}" = ?')
        parameters.append(value)
    condition = " AND ".join(clauses) if clauses else "TRUE"
    query = sql_path.read_text(encoding="utf-8").strip().rstrip(";")
    # Nest the SQL so both its aggregate passes see the same parameterized filter.
    result = connection.sql(
        f"WITH filtered_portfolio AS (SELECT * FROM portfolio WHERE {condition}) "
        f"SELECT * FROM ({query}) AS summary",
        params=parameters,
    ).df()
    return PortfolioSummary(
        overview=result.loc[result["section"].eq("overview")].drop(
            columns=["section", "revenue_segment", "risk_segment"]
        ).reset_index(drop=True),
        segments=result.loc[result["section"].eq("segment")].drop(
            columns="section",
        ).set_index(["revenue_segment", "risk_segment"]),
    )
