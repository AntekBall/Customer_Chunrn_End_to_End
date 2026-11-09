-- Read-only ingestion audit, not a second CSV loader or validation gate.
-- Prerequisite: src.data.loader.load_customers(connection) has materialized
-- validated raw_customers in the SAME DuckDB connection.
-- Execute this entire file as one query. Rates are fractions in [0, 1].
-- Undefined rates/statistics (for example on empty input) remain NULL.
-- Monthly charges are a revenue proxy, not verified active MRR or profit.

WITH overview AS (
    SELECT
        COUNT(*) AS row_count,
        COUNT(DISTINCT customer_id) AS distinct_customer_ids,
        COUNT(*) - COUNT(customer_id) AS missing_customer_ids,
        COUNT(customer_id) - COUNT(DISTINCT customer_id) AS duplicate_id_excess_rows,
        COUNT(*) FILTER (WHERE TRIM(customer_id) = '') AS blank_customer_ids,
        COUNT(*) FILTER (WHERE churn = 0) AS retained_customers,
        COUNT(*) FILTER (WHERE churn = 1) AS churned_customers,
        COUNT(*) FILTER (WHERE churn IS NULL) AS missing_churn_labels,
        COUNT(*) FILTER (WHERE churn NOT IN (0, 1)) AS invalid_churn_labels
    FROM raw_customers
),
overview_metrics AS (
    UNPIVOT overview
    ON COLUMNS(*)
    INTO NAME metric VALUE value
),
missing_counts AS (
    UNPIVOT (
        SELECT COUNT(*) - COUNT(COLUMNS(*))
        FROM raw_customers
    )
    ON COLUMNS(*)
    INTO NAME column_name VALUE missing_count
),
numeric_summary AS (
    SELECT
        column_name,
        CAST(min AS DOUBLE) AS minimum,
        CAST(avg AS DOUBLE) AS mean,
        CAST(max AS DOUBLE) AS maximum
    FROM (SUMMARIZE raw_customers)
    WHERE column_type IN ('BIGINT', 'DOUBLE')
),
numeric_metrics AS (
    SELECT column_name, metric, value
    FROM numeric_summary
    UNPIVOT INCLUDE NULLS (
        value FOR metric IN (minimum, mean, maximum)
    )
),
audit AS (
    SELECT
        'overview' AS section,
        CAST(NULL AS VARCHAR) AS column_name,
        metric,
        CAST(value AS DOUBLE) AS value
    FROM overview_metrics

    UNION ALL

    SELECT
        'churn', 'churn', 'observed_churn_rate',
        churned_customers::DOUBLE / NULLIF(row_count, 0)
    FROM overview

    UNION ALL

    SELECT 'missingness', column_name, 'missing_count', missing_count::DOUBLE
    FROM missing_counts

    UNION ALL

    SELECT
        'missingness', column_name, 'missing_fraction',
        missing_count::DOUBLE / NULLIF(overview.row_count, 0)
    FROM missing_counts
    CROSS JOIN overview

    UNION ALL

    SELECT 'numeric', column_name, metric, value
    FROM numeric_metrics
)
SELECT section, column_name, metric, value
FROM audit
ORDER BY section, column_name NULLS FIRST, metric;