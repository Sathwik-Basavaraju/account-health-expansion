"""Build account-month snapshots via DuckDB SQL (sql/snapshots.sql).

Loads data/*.csv (same generator/schema as nrr-data-warehouse),
builds account_month_fact, runs sql/snapshots.sql, writes
outputs/account_month_snapshots.csv and prints split counts.
Run from the project root: python src/build_snapshots.py
Requires: duckdb, pandas
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def main():
    con = duckdb.connect()
    for raw, csv in [("raw_billing", "billing.csv"), ("raw_usage", "usage_events.csv"),
                     ("raw_tickets", "support_tickets.csv")]:
        con.execute(f"CREATE TABLE {raw} AS SELECT * FROM read_csv_auto('{ROOT}/data/{csv}')")
    con.execute("""
        CREATE OR REPLACE TABLE account_month_fact AS
        SELECT b.account_id, b.month, CAST(b.month || '-01' AS DATE) AS month_date,
               CAST(b.seats AS INTEGER) AS seats, CAST(b.mrr AS DOUBLE) AS mrr, b.plan,
               COALESCE(u.usage_events, 0) AS usage_events,
               COALESCE(t.tickets, 0) AS tickets
        FROM raw_billing b
        LEFT JOIN (SELECT account_id, strftime(CAST(event_date AS DATE), '%Y-%m') AS month, count(*) AS usage_events
                   FROM raw_usage GROUP BY 1,2) u
          ON u.account_id = b.account_id AND u.month = b.month
        LEFT JOIN (SELECT account_id, month, count(*) AS tickets FROM raw_tickets GROUP BY 1,2) t
          ON t.account_id = b.account_id AND t.month = b.month
    """)
    con.execute((ROOT / "sql" / "snapshots.sql").read_text())
    (ROOT / "outputs").mkdir(exist_ok=True)
    con.execute(f"COPY (SELECT * FROM account_month_snapshots ORDER BY snapshot_month, account_id) "
                f"TO '{ROOT}/outputs/account_month_snapshots.csv' (HEADER, DELIMITER ',')")
    df = con.execute("SELECT * FROM account_month_snapshots").fetchdf()
    train = df[df["snapshot_month"] <= "2024-12"]
    test = df[df["snapshot_month"] >= "2025-05"]
    buffer = df[(df["snapshot_month"] >= "2025-01") & (df["snapshot_month"] <= "2025-04")]
    print(f"snapshots total={len(df)} train={len(train)} buffer={len(buffer)} test={len(test)}")
    print(f"train positive rate={train['expanded_next_3m'].mean():.3f} "
          f"test positive rate={test['expanded_next_3m'].mean():.3f}")
    return df


if __name__ == "__main__":
    main()
