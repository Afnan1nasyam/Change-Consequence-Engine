# Password Reset — Support Runbook

Owner: Support Operations
Status: Approved

## When to use this

A customer cannot sign in and self-service reset has not worked for them.

## Steps

1. Verify the customer's identity against the account record. Do not proceed on an email
   address alone.
2. Trigger a reset from the admin console. The customer receives an email with a reset link.
3. Tell the customer the link is valid for 24 hours and can be used once. If they do not use
   it in time, trigger a new one; the old link cannot be revived.
4. Confirm the customer has signed in before closing the case.

## Response targets

A P1 incident is acknowledged within 15 minutes. A password reset request is not a P1 unless
it is blocking a whole tenant.

An inbound customer contact receives a first response within 60 minutes during staffed
hours.

## If the reset email does not arrive

Check the suppression list before re-triggering. A hard bounce suppresses the address and no
number of retries will deliver to it.
