"""Generate the shared synthetic SaaS dataset (seed 42).

Same generator, same seed and same schema are used in both SaaS repos:
  nrr-data-warehouse and account-health-expansion.
Account IDs, billing rows and support tickets are therefore identical
across the two repos when run with the same arguments.

Outputs (into --out, default data/):
  accounts.csv, billing.csv, usage_events.csv, support_tickets.csv

Repo sample size (default): 600 accounts, 2024-01..2025-12.
Usage events are SAMPLED: --usage-mean (default 12) events per
account-month on average. This keeps the repo <20MB. To scale locally
(e.g. towards millions of events), run with a larger --usage-mean
and/or --accounts; do NOT upload a >20MB file to GitHub.
A full-volume run is a local experiment, not a repo file.

Billing logic (documented, deterministic with seed 42):
  - Each account starts in its signup month with seats by plan and
    MRR = seats * plan price-per-seat * noise.
  - Each subsequent month, seats/MRR may expand (+10-30%), contract
    (-10-20%) or stay flat, with small per-seat price noise.
    Expansion/contraction probabilities are health-dependent (see code),
    and ticket probability is also health-dependent, so usage,
    tickets and prior MRR growth are genuine (not random) predictors
    of future expansion. Latent health itself is NOT a model feature.
  - Monthly churn probability depends on a latent health score
    (0..1, higher = healthier): p = clip(0.03 - 0.02*health, 0.005, 0.05).
    If churn occurs, billing stops after that month.
Usage: one sampled event row per sampled event, with event_date in
the billing month, event_type, and a synthetic user id.
Support: per account-month, 0 tickets (usually), 1 or 2 tickets.
"""
import argparse
import calendar
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
INDUSTRIES = ["Technology", "Healthcare", "Financial Services", "Retail",
              "Manufacturing", "Education", "Media", "Logistics"]
REGIONS = ["North America", "Europe", "APAC", "LATAM"]
PLANS = {"Starter": (0.40, 5, 20, 49), "Growth": (0.40, 20, 80, 79),
         "Enterprise": (0.20, 80, 250, 119)}
EVENT_TYPES = ["login", "api_call", "dashboard_view", "report_export",
               "integration_sync"]
TICKET_CATEGORIES = ["bug", "billing", "feature_request", "onboarding"]
PREFIX = ["North", "Blue", "Bright", "Swift", "Ever", "Prime", "Nova", "Atlas"]
SUFFIX = ["Labs", "Works", "Systems", "Health", "Pay", "Retail", "Cloud", "Data"]


def months_between(start, end):
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def generate(n_accounts=600, usage_mean=12, seed=SEED, out="data"):
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    months = months_between((2024, 1), (2025, 12))

    accounts, billing, usage, tickets = [], [], [], []
    event_n = 0
    ticket_n = 0
    for i in range(1, n_accounts + 1):
        aid = f"ACCT-{i:04d}"
        plan = rng.choice(list(PLANS.keys()), p=[PLANS[k][0] for k in PLANS])
        _, lo, hi, pps = PLANS[plan]
        seats = int(rng.integers(lo, hi + 1))
        health = float(rng.uniform(0, 1))
        signup_y = int(rng.integers(2023, 2025))
        signup_m = int(rng.integers(1, 13)) if signup_y == 2023 else int(rng.integers(1, 7))
        signup = f"{signup_y}-{signup_m:02d}-15"
        accounts.append({
            "account_id": aid,
            "company_name": f"{rnd.choice(PREFIX)}{rnd.choice(SUFFIX)} {i}",
            "industry": str(rng.choice(INDUSTRIES)),
            "region": str(rng.choice(REGIONS)),
            "plan": plan,
            "signup_date": signup,
            "initial_seats": seats,
            "health_score_latent": round(health, 4),
        })
        mrr = seats * pps * float(rng.uniform(0.9, 1.1))
        cur_seats = seats
        # billing starts at max(signup, 2024-01); if signup in 2024, start then
        start_idx = 0
        for idx, (y, m) in enumerate(months):
            if (y, m) < (signup_y, signup_m):
                continue
            start_idx = idx
            ym = f"{y}-{m:02d}"
            billing.append({"account_id": aid, "month": ym, "seats": cur_seats,
                            "mrr": round(float(mrr), 2), "plan": plan})
            # usage sampled events for this account-month
            mean_events = usage_mean * (0.6 + 0.8 * health)
            n_ev = int(rng.poisson(max(1.0, mean_events)))
            for _ in range(n_ev):
                day = int(rng.integers(1, calendar.monthrange(y, m)[1] + 1))
                event_n += 1
                usage.append({
                    "event_id": f"EVT-{event_n:07d}",
                    "account_id": aid,
                    "event_date": f"{y}-{m:02d}-{day:02d}",
                    "event_type": str(rng.choice(EVENT_TYPES)),
                    "user_id": f"{aid}-U{int(rng.integers(1, max(2, cur_seats + 1)))}",
                    "duration_seconds": int(rng.integers(5, 1800)),
                })
            # support tickets
            # Tickets are health-dependent: lower-health accounts
            # file more tickets, so ticket count is a real (not random)
            # signal for future expansion in this synthetic world.
            r = rng.random()
            p_one = 0.30 - 0.15 * health
            p_two = p_one + (0.08 - 0.04 * health)
            n_tk = 1 if r < p_one else (2 if r < p_two else 0)
            for _ in range(n_tk):
                day = int(rng.integers(1, calendar.monthrange(y, m)[1] + 1))
                ticket_n += 1
                tickets.append({
                    "ticket_id": f"TKT-{ticket_n:06d}",
                    "account_id": aid,
                    "month": ym,
                    "created_date": f"{y}-{m:02d}-{day:02d}",
                    "category": str(rng.choice(TICKET_CATEGORIES)),
                    "priority": str(rng.choice(["Low", "Medium", "High"], p=[0.5, 0.35, 0.15])),
                    "resolution_days": int(rng.integers(1, 15)),
                })
            # churn decision at end of month
            p_churn = float(np.clip(0.03 - 0.02 * health, 0.005, 0.05))
            if rng.random() < p_churn:
                break
            # evolve seats/mrr for next month
            # Expansion/contraction are health-dependent, so that
            # observable proxies (usage, tickets, prior MRR growth) can
            # legitimately predict future expansion. Health itself is
            # NOT a model feature (it is latent); only its proxies are.
            if health > 0.7:
                p_exp, p_con = 0.14, 0.03
            elif health > 0.4:
                p_exp, p_con = 0.08, 0.05
            else:
                p_exp, p_con = 0.03, 0.09
            r = rng.random()
            if r < p_exp:
                cur_seats = int(round(cur_seats * float(rng.uniform(1.10, 1.30))))
            elif r < p_exp + p_con:
                cur_seats = max(1, int(round(cur_seats * float(rng.uniform(0.80, 0.90)))))
            mrr = cur_seats * pps * float(rng.uniform(0.98, 1.02))
    pd.DataFrame(accounts).to_csv(out / "accounts.csv", index=False)
    pd.DataFrame(billing).to_csv(out / "billing.csv", index=False)
    pd.DataFrame(usage).to_csv(out / "usage_events.csv", index=False)
    pd.DataFrame(tickets).to_csv(out / "support_tickets.csv", index=False)
    print(f"accounts={len(accounts)} billing_rows={len(billing)} "
          f"usage_events={len(usage)} tickets={len(tickets)}")
    return {"accounts": len(accounts), "billing_rows": len(billing),
            "usage_events": len(usage), "tickets": len(tickets)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", type=int, default=600)
    ap.add_argument("--usage-mean", type=float, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    generate(a.accounts, a.usage_mean, a.seed, a.out)
