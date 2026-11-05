# Account Health Score and Expansion Propensity Model

Python + scikit-learn + SQL (DuckDB). Synthetic SaaS dataset
shared with the companion repo `nrr-data-warehouse` — see
DATA_SCHEMA.md; same generator, seed 42, same account IDs, billing
and support data, so the two repos align.

## What this is
An account-month snapshot model that scores whether a SaaS account
will expand (MRR growth) in a future window, without using any
future information in its features, and a 3-tier health playbook
built from held-out expansion rates.

## Data (sample, not production volume)
- Accounts: 600 (same as nrr-data-warehouse)
- Billing rows: 10,888 (2024-01 to 2025-12)
- Usage events (sampled): 132,952
- Support tickets: 3,694
- Seed: 42. Usage is sampled at mean 12 events per account-month to
  keep the repo under 20 MB. This repo does NOT contain 7M usage
  events. The generator (`src/generate_data.py`) can scale locally
  with `--usage-mean` / `--accounts`.

## Snapshot definition (no leakage)
- Snapshot month t: features aggregated over t−2..t (3 months, inclusive).
- Gap: month t+1 is excluded from both features and target.
- Target: expansion in t+2..t+4 (3 months), where expansion means
  any month in the window with MRR > 1.05 × MRR at t.
  Accounts churned before the target window have target = 0.
- Kept only if billing exists in all of t−2..t and t+4 ≤ 2025-12.
  No feature uses any month after t.
- Split: train snapshot_month ≤ 2024-12; buffer excluded
  2025-01 to 2025-04; test snapshot_month ≥ 2025-05.
  The buffer ensures train target windows (ending ≤ 2025-04) do not
  overlap test target windows (starting ≥ 2025-07).
- Features: seats, mrr, tenure_months, usage_events_3m,
  usage_per_seat_3m, tickets_3m, mrr_growth_3m,
  prior_expansion_flag, usage_growth, plan.
  Latent health (`health_score_latent` in accounts.csv) is NOT a
  feature — it only drives the generator's churn/expansion/ticket
  probabilities (see DATA_SCHEMA.md). Only its observable proxies
  (usage, tickets, prior MRR growth) are features.

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py --out data
python src/build_snapshots.py   # loads CSVs, runs sql/snapshots.sql
python src/train_model.py       # logistic vs gradient boosting, bootstrap CI
python src/make_charts.py       # ROC + tier expansion-rate charts
```

## Results — actually computed on this run (see outputs/metrics.json)
- Snapshots: train n = **4,592** (positive rate 0.2537),
  test n = **1,706** (positives 438, positive rate 0.2567)
- Logistic regression AUC: **0.7008**, 95% CI **0.6726–0.7279**
  (bootstrap, 200 resamples of the test set, 0 skipped)
- Gradient boosting AUC: **0.6351**, 95% CI **0.6045–0.6661**
  (HistGradientBoostingClassifier, max_iter 200, lr 0.05, seed 42;
  same 200-resample bootstrap)
- Best model by AUC: **logistic regression**
- Top decile (best model, test set): n = **171**,
  expansion rate **0.4795**, capture rate **0.1872**,
  lift vs random **1.8721**
- 3-tier health playbook (best model scores, test set;
  High = top 30%, Medium = middle 40%, Low = bottom 30%;
  score cutoffs: Low/Medium 0.1872, Medium/High 0.3033):
  - High: n = **512**, expansion rate **0.4297**
    → prioritise for expansion conversations, executive business reviews
  - Medium: n = **682**, expansion rate **0.2493**
    → nurture, usage-adoption and training offers
  - Low: n = **512**, expansion rate **0.0938**
    → save/retention playbook first, not an expansion push
- These numbers are from this synthetic, sampled dataset only.
  They are not a claim about any real company or customer base,
  and they are not the placeholder figures from the original brief.

## Files
- `sql/snapshots.sql` — snapshot creation SQL (DuckDB)
- `src/` — generate_data.py, build_snapshots.py, train_model.py, make_charts.py
- `data/` — accounts.csv, billing.csv, usage_events.csv, support_tickets.csv
- `outputs/` — account_month_snapshots.csv, test_predictions.csv,
  metrics.json, roc.png, tier_expansion_rates.png
- `DATA_SCHEMA.md` — shared schema with nrr-data-warehouse
- `MANIFEST.txt` — files to upload (excludes __pycache__ and large raw files)
