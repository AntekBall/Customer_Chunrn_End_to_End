WITH revenue_labels(revenue_segment, revenue_order) AS (
    VALUES ('Lower half', 1), ('Middle 40%', 2),
           ('High revenue', 3), ('Very high revenue', 4)
),
risk_labels(risk_segment, risk_order) AS (
    VALUES ('Lower risk', 1), ('Elevated risk', 2), ('Highest risk', 3)
),
totals AS (
    SELECT COUNT(*) AS customers,
           COALESCE(SUM(monthlycharges), 0.0) AS monthly_charge_total,
           AVG(monthlycharges) AS mean_monthly_charge,
           AVG(churn_probability) AS mean_predicted_churn,
           COALESCE(SUM(monthlycharges * churn_probability), 0.0)
               AS risk_weighted_monthly_charges
    FROM filtered_portfolio
),
cells AS (
    SELECT revenue_segment, risk_segment, COUNT(*) AS customers,
           SUM(monthlycharges) AS monthly_charge_total,
           AVG(monthlycharges) AS mean_monthly_charge,
           AVG(churn_probability) AS mean_predicted_churn,
           SUM(monthlycharges * churn_probability) AS risk_weighted_monthly_charges
    FROM filtered_portfolio
    GROUP BY revenue_segment, risk_segment
),
combined AS (
    SELECT 'overview' AS section, NULL::VARCHAR AS revenue_segment,
           NULL::VARCHAR AS risk_segment, 0 AS revenue_order, 0 AS risk_order,
           totals.*, customers / NULLIF(customers, 0) AS customer_share,
           monthly_charge_total / NULLIF(monthly_charge_total, 0) AS monthly_charge_share
    FROM totals
    UNION ALL
    SELECT 'segment', r.revenue_segment, k.risk_segment,
           r.revenue_order, k.risk_order,
           COALESCE(c.customers, 0), COALESCE(c.monthly_charge_total, 0.0),
           c.mean_monthly_charge, c.mean_predicted_churn,
           COALESCE(c.risk_weighted_monthly_charges, 0.0),
           COALESCE(c.customers, 0) / NULLIF(t.customers, 0),
           COALESCE(c.monthly_charge_total, 0.0) / NULLIF(t.monthly_charge_total, 0)
    FROM revenue_labels r CROSS JOIN risk_labels k CROSS JOIN totals t
    LEFT JOIN cells c USING (revenue_segment, risk_segment)
)
SELECT * EXCLUDE (revenue_order, risk_order)
FROM combined
ORDER BY revenue_order, risk_order;
