# Data Retention Policy

Owner: Privacy Office
Status: Approved
Scope: global

## Retention period

Personal data is retained for 90 days after the account is closed, then deleted. This
applies globally except where a regional policy sets a shorter period, in which case the
shorter period governs for that region.

## Account dormancy

An account with no activity for 400 days is marked dormant. A dormant account is excluded
from active-user reporting and its personal data enters the retention clock only once the
account is closed.

## Deletion mechanics

Deletion is a hard delete from primary storage and a tombstone in the audit log. Backups age
out on their own schedule; a restore from backup must re-apply outstanding deletions.

## Exports

A data export requested by a customer is generated from primary storage only. An export
cannot resurrect data already deleted under this policy.
