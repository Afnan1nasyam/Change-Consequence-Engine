# Major Incident Response Runbook

Owner: reliability-engineering. Supersedes the 2023 paging guide.

## Declaring a P1

Any incident with customer-visible data loss, a full-region outage, or a failed payment
path is a P1. Declare early; downgrading is cheap and late declaration is not.

## Acknowledgement

A P1 incident is acknowledged within 30 minutes. The on-call engineer acknowledges in the
incident channel and takes the incident commander role until relieved.

## Escalation ladder

1. On-call engineer acknowledges and assesses.
2. If unresolved after 30 minutes, page the service owner.
3. If unresolved after 60 minutes, page the duty director.

## Communications

Status page updates go out on declaration and then every 30 minutes until resolution.
Customer support is briefed before the first public update, not after.
