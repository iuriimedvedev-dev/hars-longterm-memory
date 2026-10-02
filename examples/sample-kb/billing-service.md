# Billing service

## Ownership

The billing service is owned by the Payments team. Contact: payments-team@acme.example.

## Behaviour

Invoices are generated on the 1st of every month. Failed payments are retried
three times, 24 hours apart. After the third failure the account is marked
`past_due`.

## Limits

- Maximum 500 invoice lines per invoice.
- Refunds above 1000 EUR need manual approval.
