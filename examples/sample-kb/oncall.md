# On-call guide

## Rotation

The on-call engineer rotates every Monday at 09:00 UTC. The schedule lives in
the team calendar.

## Escalation

1. Page the primary on-call engineer.
2. If there is no acknowledgement within 10 minutes, page the secondary.
3. For severity 1 incidents, also notify the engineering manager.

## Severity levels

| Level | Meaning | Response time |
|---|---|---|
| SEV1 | Platform down for all customers | 5 minutes |
| SEV2 | Major feature degraded | 30 minutes |
| SEV3 | Minor issue, workaround exists | next business day |
