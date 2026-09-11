# INC-4471 — Refund approval backlog after threshold change

Status: Closed
Severity: SEV3
Opened: 2024-06-03
Closed: 2024-06-27

## Summary

Following the June 2024 reduction of the refund approval threshold from 25,000 USD to
10,000 USD, the volume of refunds requiring manager approval rose sharply and the approval
queue depth tripled for approximately three weeks.

## Timeline

- 2024-06-03: threshold change released. Approval requests rise from ~40/day to ~130/day.
- 2024-06-05: first customer complaints about refund delays.
- 2024-06-11: two additional managers granted temporary approval authority.
- 2024-06-27: queue depth back to baseline. Incident closed.

## What we learned

The change itself was correct and stayed in place. The failure was that nobody modelled the
approval volume before releasing it, and the training material still described the old
threshold for a further two weeks, so agents escalated cases that did not need escalating.

## Observed outcome

Approval queue depth tripled for three weeks. Median refund turnaround went from 4 hours to
19 hours at the peak. No refunds were lost, and no regulatory breach occurred.

## Follow-up

Model approval volume before any future change to an approval threshold. Update training
material in the same change window as the policy.
