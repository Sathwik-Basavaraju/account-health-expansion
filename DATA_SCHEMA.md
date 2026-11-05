# Shared Synthetic SaaS Data Schema

This schema is SHARED by both SaaS repos in this set:
`nrr-data-warehouse` and `account-health-expansion`.
Both repos use the identical generator (`src/generate_data.py`,
seed 42, same arguments), so account IDs, billing rows and support
tickets are identical across the two repos. Usage events are sampled
(mean 12 per account-month) in both repos from the same generator,
so usage files are also identical when run with the same arguments.

## accounts.csv (600 rows in the repo sample)
| column | type | meaning |
|---|---|---|
| account_id | string, e.g. ACCT-0001 | Primary key, shared across repos |
| company_name | string | Synthetic company name |
| industry | string | Technology, Healthcare, Financial Services, Retail, Manufacturing, Education, Media, Logistics |
| region | string | North America, Europe, APAC, LATAM |
| plan | string | Starter, Growth, Enterprise |
| signup_date | date | Signup date (2023-01 to 2024-06 in the sample) |
| initial_seats | integer | Seats at signup, by plan |
| health_score_latent | float 0-1 | Latent health used ONLY by the generator to drive churn, expansion and ticket rates. It is NOT a model feature in the health-score repo. |

## billing.csv (one row per account per active month, 2024-01 to 2025-12)
| column | type | meaning |
|---|---|---|
| account_id | string | FK to accounts.csv |
| month | string YYYY-MM | Billing month |
| seats | integer | Seats in that month (may expand/contract) |
| mrr | float | Monthly recurring revenue that month (seats × plan price-per-seat × noise) |
| plan | string | Plan in that month (unchanged in the sample generator) |

Billing stops after an account's churn month. MRR magnitudes are
synthetic (seats × price-per-seat) and are not a real company's revenue.

## usage_events.csv (sampled, ~133k rows in the repo sample)
| column | type | meaning |
|---|---|---|
| event_id | string, e.g. EVT-0000001 | Primary key |
| account_id | string | FK to accounts.csv |
| event_date | date | Date within the billing month |
| event_type | string | login, api_call, dashboard_view, report_export, integration_sync |
| user_id | string | Synthetic user within the account (ACCT-xxxx-U<n>) |
| duration_seconds | integer | Event duration, synthetic |

Usage is SAMPLED (mean 12 events per account-month by default) to keep
the repo under 20 MB. Run the generator locally with a larger
`--usage-mean` (and/or `--accounts`) to scale up; do NOT upload a
>20 MB file to GitHub. The sampled file is NOT a claim of full
production usage volume.

## support_tickets.csv
| column | type | meaning |
|---|---|---|
| ticket_id | string, e.g. TKT-000001 | Primary key |
| account_id | string | FK to accounts.csv |
| month | string YYYY-MM | Ticket month |
| created_date | date | Date within the month |
| category | string | bug, billing, feature_request, onboarding |
| priority | string | Low, Medium, High |
| resolution_days | integer | Synthetic resolution time |

## Generator behaviour (documented, not hidden)
- Monthly churn probability: clip(0.03 − 0.02 × health, 0.005, 0.05).
- Monthly expansion/contraction probabilities are health-dependent:
  health > 0.7: expansion 14%, contraction 3%;
  health 0.4–0.7: expansion 8%, contraction 5%;
  health < 0.4: expansion 3%, contraction 9%.
  Expansion = seats × 1.10–1.30; contraction = seats × 0.80–0.90.
- Ticket probability is also health-dependent (lower health → more tickets).
- Usage mean per account-month = 12 × (0.6 + 0.8 × health), Poisson-sampled.
These rules are why observable proxies (usage, tickets, prior MRR
growth) can predict future expansion in the health-score repo, while
latent health itself is never a feature.
