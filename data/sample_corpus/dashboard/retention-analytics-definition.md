# Customer Lifetime Value — Metric Definition

Owner: analytics-platform. Reviewed annually.

## Population

The CLV model is computed over all closed accounts with at least one completed
transaction. Accounts closed for fraud are excluded.

## Lookback window

Personal data is retained for 365 days after the account is closed, so the model reads
the full year of post-closure behaviour when attributing refunds and chargebacks to a
cohort. Cohorts are keyed on the closure month.

## Refresh

The definition is recomputed nightly. A recomputation that finds fewer rows than the
previous run raises a data-quality alert rather than publishing.

## Known limitations

Seat-level detail is not available for plans migrated before 2023, so multi-seat cohorts
before that date are approximated from invoice totals.
