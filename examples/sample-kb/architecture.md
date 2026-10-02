# Acme Platform architecture

## Overview

Acme Platform is a set of small services behind one API gateway.

## Services

### Gateway

Terminates TLS, authenticates requests and routes them to internal services.
It listens on port 8443.

### Billing service

Computes invoices and talks to the payments provider. See `billing-service.md`.

### Notification worker

Consumes the `events` queue and sends emails and webhooks.

## Data stores

| Store | Used by | Notes |
|---|---|---|
| PostgreSQL (primary + replica) | billing, gateway | nightly backups at 02:00 UTC |
| Redis | gateway | session cache, 15 minute TTL |
| Message queue `events` | notification worker | at-least-once delivery |
