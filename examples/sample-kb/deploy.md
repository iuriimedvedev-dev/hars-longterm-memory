# Deployment procedure

## Standard release

Releases go out on Tuesdays and Thursdays. Never deploy on Fridays.

### Steps

1. Merge to `main` and wait for CI to pass.
2. Tag the release:

```bash
git tag v1.8.0
git push origin v1.8.0
```

3. Run the canary deploy and watch the error rate for 15 minutes.
4. Promote the canary to 100% if the error rate stays below 1%.

## Rollback

If the error rate exceeds 1% during the canary, roll back immediately:

```bash
acme-deploy rollback --service billing --to previous
```

Then open an incident and notify the on-call engineer.
