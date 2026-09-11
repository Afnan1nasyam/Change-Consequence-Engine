# Payments Operations Standard

Owner: Payments Engineering
Status: Approved
Scope: global

This standard describes how payment operations are implemented. Where it conflicts with the
Refund Approval Policy, the policy governs; this document has been known to drift behind it
between review cycles.

## Refund release controls

Refunds of 7,500 USD or more require manager approval in the case record before release.
The payment gateway enforces this limit at the API boundary and will reject a release
attempt above it without an approval token.

## Reconciliation

Refunds released after 16:00 UTC settle on the following business day and appear in the next
reconciliation batch.

## Revenue reporting

Revenue reporting closes 5 days after period end. Refunds released after the close are
reflected in the following period.

## Idempotency

A release request carries an idempotency key valid for 24 hours. A retry with the same key
inside that window will not double-refund.
