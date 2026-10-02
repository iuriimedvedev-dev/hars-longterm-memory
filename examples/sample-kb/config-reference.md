# Configuration reference

## Gateway

| Key | Default | Description |
|---|---|---|
| `GATEWAY_PORT` | `8443` | listening port |
| `SESSION_TTL_MINUTES` | `15` | Redis session lifetime |
| `RATE_LIMIT_PER_MIN` | `600` | requests per client per minute |

## Billing

| Key | Default | Description |
|---|---|---|
| `INVOICE_DAY` | `1` | day of month invoices are generated |
| `PAYMENT_RETRIES` | `3` | retries before `past_due` |
