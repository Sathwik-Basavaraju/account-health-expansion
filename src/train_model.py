"""Train and evaluate expansion propensity models.

Split (time-based, with gap):
  train: snapshot_month <= 2024-12
  buffer excluded: 2025-01..2025-04
  test:  snapshot_month >= 2025-05
Features (no leakage: all from t-2..t only):
  seats, mrr, tenure_months, usage_events_3m, usage_per_seat_3m,
  tickets_3m, mrr_growth_3m, prior_expansion_flag, usage_growth, plan
Target: expanded_next_3m (expansion in t+2..t+4, gap month t+1 excluded).

Models: LogisticRegression (scaled, one-hot plan) vs
HistGradientBoostingClassifier. AUC with 95% bootstrap CI
(200 resamples of the test set, percentiles 2.5/97.5; resamples with
a single class are skipped and the count is reported).
Top-decile lift/capture and a 3-tier playbook (High = top 30% of
test scores, Medium = middle 40%, Low = bottom 30%) are computed
on the held-out test set only.

Writes outputs/metrics.json, outputs/test_predictions.csv,
and prints all metrics. Run after src/build_snapshots.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parent.parent
NUM = ["seats", "mrr", "tenure_months", "usage_events_3m", "usage_per_seat_3m",
       "tickets_3m", "mrr_growth_3m", "prior_expansion_flag", "usage_growth"]
CAT = ["plan"]


def bootstrap_auc(y, p, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    aucs, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), len(aucs), skipped


def main():
    df = pd.read_csv(ROOT / "outputs" / "account_month_snapshots.csv")
    train = df[df["snapshot_month"] <= "2024-12"].copy()
    test = df[df["snapshot_month"] >= "2025-05"].copy()
    Xtr, ytr = train[NUM + CAT], train["expanded_next_3m"]
    Xte, yte = test[NUM + CAT], test["expanded_next_3m"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    models = {
        "logistic_regression": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]),
        "gradient_boosting": Pipeline([
            ("pre", ColumnTransformer([
                ("num", "passthrough", NUM),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ])),
            ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)),
        ]),
    }
    results = {}
    preds = {}
    for name, model in models.items():
        print(f"fitting {name}...", flush=True)
        model.fit(Xtr, ytr)
        print(f"fitted {name}, scoring...", flush=True)
        p = model.predict_proba(Xte)[:, 1]
        print(f"scored {name}, bootstrapping...", flush=True)
        preds[name] = p
        lo, hi, n_boot, skipped = bootstrap_auc(yte, p)
        results[name] = {
            "auc": round(float(roc_auc_score(yte, p)), 4),
            "auc_95ci_low": round(lo, 4), "auc_95ci_high": round(hi, 4),
            "bootstrap_resamples_used": n_boot, "bootstrap_skipped_single_class": skipped,
        }
        print(name, results[name])

    # Best model by AUC for tier/playbook reporting (report both, use best for tiers)
    best_name = max(results, key=lambda k: results[k]["auc"])
    p = preds[best_name]
    test_out = test[["account_id", "snapshot_month"]].copy()
    test_out["y_true"] = yte.values
    test_out["y_score"] = p
    test_out.to_csv(ROOT / "outputs" / "test_predictions.csv", index=False)

    order = np.argsort(-p)
    n_top = max(1, int(round(0.10 * len(p))))
    top_idx = order[:n_top]
    positives = int(yte.sum())
    top_positives = int(yte.values[top_idx].sum())
    capture = top_positives / positives if positives else 0.0
    lift = capture / 0.10 if positives else 0.0
    top_rate = float(yte.values[top_idx].mean())

    # Tiers on test scores: High top 30%, Medium middle 40%, Low bottom 30%
    thresholds = np.quantile(p, [0.30, 0.70])
    tier = np.where(p >= thresholds[1], "High", np.where(p >= thresholds[0], "Medium", "Low"))
    tier_rows = []
    for t in ["High", "Medium", "Low"]:
        mask = tier == t
        tier_rows.append({"tier": t, "n": int(mask.sum()),
                          "expansion_rate": round(float(yte.values[mask].mean()), 4) if mask.sum() else None})
    metrics = {
        "split": {"train_snapshot_month_max": "2024-12", "test_snapshot_month_min": "2025-05",
                  "buffer_excluded": "2025-01 to 2025-04", "gap_months": 1,
                  "target_window": "t+2 to t+4 months, expansion = MRR > 1.05 * MRR at t"},
        "train_n": int(len(train)), "test_n": int(len(test)),
        "train_positive_rate": round(float(ytr.mean()), 4),
        "test_positive_rate": round(float(yte.mean()), 4),
        "test_positives": positives,
        "features": NUM + CAT,
        "models": results,
        "best_model_by_auc": best_name,
        "top_decile": {"n": int(n_top), "expansion_rate": round(top_rate, 4),
                       "capture_rate": round(float(capture), 4), "lift_vs_random": round(float(lift), 4)},
        "tiers": tier_rows,
        "tier_thresholds": {"low_medium_cutoff": round(float(thresholds[0]), 4),
                            "medium_high_cutoff": round(float(thresholds[1]), 4)},
        "note": "All metrics are computed on this run's synthetic data and held-out test set. "
                "No placeholder metrics are used.",
    }
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
