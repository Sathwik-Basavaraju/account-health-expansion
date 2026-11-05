-- Account-month snapshots (DuckDB). Run via src/build_snapshots.py,
-- which loads data/*.csv, builds account_month_fact, then runs this SQL.
-- Snapshot month t: features aggregated over t-2..t (3 months, inclusive).
-- Gap: month t+1 is excluded from both features and target.
-- Target: expansion in t+2..t+4 (3 months), where expansion means any
--   month in the window with MRR > 1.05 * MRR at t.
--   Accounts churned before the target window have target = 0.
--   Snapshots are kept only if t+4 <= '2025-12' and the account has
--   billing in all of t-2..t (no leakage: no feature uses any month after t).
-- Split (documented in README / metrics.json):
--   train: snapshot_month <= '2024-12'
--   buffer (excluded): '2025-01'..'2025-04'
--   test:  snapshot_month >= '2025-05'
-- The buffer ensures train target windows (ending <= 2025-04) do not
-- overlap test target windows (starting >= 2025-07).
CREATE OR REPLACE TABLE account_month_snapshots AS
WITH fact AS (
    SELECT account_id, month, month_date, seats, mrr, plan,
           usage_events, tickets
    FROM account_month_fact
),
first_month AS (
    SELECT account_id, min(month_date) AS first_month_date
    FROM fact GROUP BY account_id
),
feat AS (
    SELECT
        f.account_id,
        f.month AS snapshot_month,
        f.month_date AS snapshot_date,
        f.seats,
        f.mrr,
        f.plan,
        (date_diff('month', fm.first_month_date, f.month_date) + 1) AS tenure_months,
        (SELECT sum(f2.usage_events) FROM fact f2
          WHERE f2.account_id = f.account_id
            AND f2.month_date BETWEEN f.month_date - INTERVAL 2 MONTH AND f.month_date) AS usage_events_3m,
        (SELECT sum(f2.tickets) FROM fact f2
          WHERE f2.account_id = f.account_id
            AND f2.month_date BETWEEN f.month_date - INTERVAL 2 MONTH AND f.month_date) AS tickets_3m,
        (SELECT count(*) FROM fact f2
          WHERE f2.account_id = f.account_id
            AND f2.month_date BETWEEN f.month_date - INTERVAL 2 MONTH AND f.month_date) AS months_in_window,
        (SELECT mrr FROM fact f2
          WHERE f2.account_id = f.account_id
            AND f2.month_date = f.month_date - INTERVAL 2 MONTH) AS mrr_t_minus2,
        f.usage_events AS usage_last_month,
        (SELECT avg(usage_events) FROM fact f2
          WHERE f2.account_id = f.account_id
            AND f2.month_date BETWEEN f.month_date - INTERVAL 2 MONTH AND f.month_date - INTERVAL 1 MONTH) AS usage_avg_prev2
    FROM fact f
    JOIN first_month fm ON fm.account_id = f.account_id
)
SELECT
    account_id,
    snapshot_month,
    seats,
    mrr,
    plan,
    tenure_months,
    usage_events_3m,
    usage_events_3m::double / nullif(seats, 0) AS usage_per_seat_3m,
    tickets_3m,
    CASE WHEN mrr_t_minus2 > 0 THEN (mrr - mrr_t_minus2) / mrr_t_minus2 ELSE 0 END AS mrr_growth_3m,
    CASE WHEN mrr_t_minus2 > 0 AND mrr > mrr_t_minus2 THEN 1 ELSE 0 END AS prior_expansion_flag,
    CASE WHEN usage_avg_prev2 > 0 THEN (usage_last_month - usage_avg_prev2) / usage_avg_prev2 ELSE 0 END AS usage_growth,
    CASE WHEN EXISTS (
        SELECT 1 FROM fact f2
        WHERE f2.account_id = feat.account_id
          AND f2.month_date BETWEEN feat.snapshot_date + INTERVAL 2 MONTH AND feat.snapshot_date + INTERVAL 4 MONTH
          AND f2.mrr > 1.05 * feat.mrr
    ) THEN 1 ELSE 0 END AS expanded_next_3m
FROM feat
WHERE months_in_window = 3
  AND snapshot_date + INTERVAL 4 MONTH <= DATE '2025-12-01'
  AND usage_events_3m IS NOT NULL
  AND tickets_3m IS NOT NULL;
