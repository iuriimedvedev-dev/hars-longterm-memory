# Database failover

Use this procedure when the PostgreSQL primary is unreachable.

## Symptoms

- Gateway returns 503 with `db_unavailable`.
- Replication lag alert fires.

## Procedure

1. Confirm the primary is really down, not just the network path.
2. Promote the replica:

```bash
acme-db promote --cluster main --replica db-replica-1
```

3. Point the services at the new primary by updating `DB_HOST` in the config.
4. Rebuild the old primary as a new replica once it is back.

## After the incident

Write a post-mortem within 3 business days.
