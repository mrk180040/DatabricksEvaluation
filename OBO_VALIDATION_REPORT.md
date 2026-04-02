# OBO Validation Report

## What "OBO" means here

In this repository, OBO means **passing a delegated Databricks access token for a specific request** instead of always using the shared fallback token from environment variables.

This app currently validates **token propagation and precedence**, not a full OAuth token-exchange protocol inside the app.

### Plain-English summary

- Without OBO: the app uses one shared token from `.env` (`DATABRICKS_TOKEN`).
- With OBO: the caller can send a token in the request header, and that token is used for that request.
- The key behavior being validated is:
  1. request token wins first,
  2. then `DATABRICKS_OBO_TOKEN`,
  3. then `DATABRICKS_TOKEN`.

## What changed in the app

The following files were updated to support OBO-style delegated token usage:

- [project/utils/llm_client.py](project/utils/llm_client.py) — resolves token precedence and supports runtime token override.
- [project/api.py](project/api.py) — reads request-scoped token from `Authorization: Bearer ...` or `X-Databricks-OBO-Token`.
- [project/main.py](project/main.py) — adds CLI `--obo-token`.
- [project/app_streamlit.py](project/app_streamlit.py) — supports session-level OBO token input.
- [DEPLOYMENT.md](DEPLOYMENT.md) — includes curl examples.

## What you are actually validating

### Validation 1: The API can accept a request-scoped token
If the app works when you pass `Authorization: Bearer <token>`, then request token intake is working.

### Validation 2: Request token overrides environment fallback
If the app still works **even when `DATABRICKS_TOKEN` is intentionally wrong**, then header precedence is working.

### Validation 3: Fallback still works when no request token is supplied
If removing the header makes the app fall back to `.env`, then the fallback path is working.

## Why responses may look the same

If you use the same Databricks identity in both places, the response may look identical.

That is normal.

OBO changes **which identity is used**, not necessarily the final wording of the model response.

You will see the biggest difference when:
- two different users have different permissions,
- one token is invalid and one is valid,
- one user can access a resource and another cannot.

## Reproducible end-to-end check

### Step 0: Start the API

```bash
PYTHONPATH=. python project/api.py
```

### Step 1: Health check

```bash
curl -s http://127.0.0.1:8080/api/status
```

Expected result:
- JSON response with `"status": "healthy"`

### Step 2: Baseline request using `.env` fallback

```bash
curl -sS -X POST "http://127.0.0.1:8080/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Show failed Databricks job diagnostics"}'
```

Expected result:
- Success if `.env` token is valid and the serving endpoint is reachable.

### Step 3: Override `.env` token with an intentionally bad value

Run this in a **new terminal** so your API process still keeps its original `.env` settings:

```bash
export BAD_TOKEN="not-a-real-token"
export VALID_TOKEN="${DATABRICKS_TOKEN}"
```

Then call with request-scoped header token:

```bash
curl -sS -X POST "http://127.0.0.1:8080/api/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VALID_TOKEN}" \
  -d '{"query":"Show failed Databricks job diagnostics"}'
```

Expected result:
- Success, because the header token should be used for that request.

### Step 4: Remove the header and compare

```bash
curl -sS -X POST "http://127.0.0.1:8080/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Show failed Databricks job diagnostics"}'
```

Expected result:
- This uses the token loaded by the API process from `.env`.
- If it returns the same answer, it means both requests used effectively the same identity or same permissions.

## Stronger validation pattern

The strongest proof is **difference in outcome**, not difference in wording.

Examples:
- valid header token succeeds while invalid fallback fails,
- user A can access something user B cannot,
- audit trail shows a different user identity downstream.

## How to interpret outcomes

### Case A: Baseline succeeds, header succeeds, both answers look same
Meaning:
- OBO wiring is probably correct,
- both paths likely resolve to the same Databricks identity,
- no permission difference is visible in your Free Edition setup.

### Case B: Header succeeds, fallback fails
Meaning:
- precedence is definitely working,
- request-scoped OBO token is overriding fallback as intended.

### Case C: Both fail
Meaning:
- token is invalid, expired, missing permissions, or model serving is unavailable in your tier.

### Case D: Health works but query fails
Meaning:
- Flask app is up,
- auth or serving endpoint access is the likely issue.

## What this does NOT prove

This app does **not** currently implement a full OAuth OBO exchange inside the server.

It proves:
- request token capture,
- request token precedence,
- delegated token propagation into the Databricks-backed LLM client.

It does **not** prove:
- token minting by an identity provider,
- refresh-token handling,
- full multi-user enterprise OBO session brokerage.

## Free Edition expectations

In Databricks Free Edition, it is common to see limited visible difference because:
- you usually have one user identity,
- there may be limited permission separation,
- some serving capabilities may be constrained.

So the most realistic success criterion is:
- **header token path works**,
- **fallback path works**,
- **precedence behaves as designed**.

## Best next step

Run the helper script:

```bash
bash scripts/verify_obo.sh
```

It prints each validation step and what the result means.

For a stronger proof that header-based OBO overrides fallback, run:

```bash
bash scripts/verify_obo.sh prove-precedence
```

That mode:
- starts a temporary API on a separate port,
- forces `DATABRICKS_TOKEN` to an invalid value for that temporary server,
- sends one request with no header,
- sends one request with `Authorization: Bearer ...`,
- sends one request with `X-Databricks-OBO-Token: ...`.

If the no-header request fails while one or both header-based requests work, precedence is proven end-to-end.

## Security note

If a real Databricks token was pasted into `.env` or chat history, rotate it in Databricks and replace it immediately.
