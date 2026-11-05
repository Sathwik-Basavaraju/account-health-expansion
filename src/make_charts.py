"""Charts: ROC curves, tier expansion rates, top-decile lift context."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent


def main():
    # ROC needs model predictions: recompute quickly from test_predictions
    # for the best model is not stored per-model, so ROC is drawn from
    # the best-model test_predictions file (score column).
    pred = pd.read_csv(ROOT / "outputs" / "test_predictions.csv")
    fpr, tpr, _ = roc_curve(pred["y_true"], pred["y_score"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label="Best model (by AUC, see metrics.json)")
    ax.plot([0, 1], [0, 1], "--", label="Random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — held-out test set"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "outputs" / "roc.png", dpi=150); plt.close(fig)

    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    tiers = m["tiers"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([t["tier"] for t in tiers], [t["expansion_rate"] for t in tiers])
    ax.set_ylabel("Expansion rate (test set)"); ax.set_title("Expansion rate by health tier")
    for i, t in enumerate(tiers):
        ax.text(i, t["expansion_rate"], f"{t['expansion_rate']:.1%} (n={t['n']})", ha="center")
    fig.tight_layout(); fig.savefig(ROOT / "outputs" / "tier_expansion_rates.png", dpi=150)
    print("charts written")


if __name__ == "__main__":
    main()
