#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-basic}"
API_URL="${API_URL:-http://127.0.0.1:8080}"
QUERY='{"query":"Show failed Databricks job diagnostics"}'
TEMP_PORT="${TEMP_PORT:-8091}"
TEMP_API_URL="http://127.0.0.1:${TEMP_PORT}"
TEMP_LOG="${TEMP_LOG:-/tmp/obo_verify_api.log}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif [[ -x .venv312/bin/python ]]; then
    PYTHON_BIN=".venv312/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${DATABRICKS_TOKEN:-}" ]]; then
  echo "[FAIL] DATABRICKS_TOKEN is not set. Add it to .env or export it in the shell."
  exit 1
fi

VALID_TOKEN="${VALID_TOKEN:-$DATABRICKS_TOKEN}"

cleanup() {
  if [[ -n "${TEMP_API_PID:-}" ]]; then
    kill "${TEMP_API_PID}" >/dev/null 2>&1 || true
    wait "${TEMP_API_PID}" >/dev/null 2>&1 || true
  fi
}

wait_for_health() {
  local url="$1"
  local attempts=20
  local i
  for ((i = 1; i <= attempts; i++)); do
    if curl -sS "${url}/api/status" | grep -q 'healthy'; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_basic_checks() {
  echo "== OBO validation helper =="
  echo "API_URL=${API_URL}"
  echo

  echo "[1/4] Checking API health"
  STATUS_RESPONSE="$(curl -sS "${API_URL}/api/status" || true)"
  echo "$STATUS_RESPONSE"
  echo

  if [[ "$STATUS_RESPONSE" != *'"status":"healthy"'* && "$STATUS_RESPONSE" != *'"status": "healthy"'* ]]; then
    echo "[WARN] API health check did not return healthy. Start the server first:"
    echo "       PYTHONPATH=. python project/api.py"
    exit 1
  fi

  echo "[2/4] Baseline request using fallback token from API process"
  BASELINE_RESPONSE="$(curl -sS -X POST "${API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -d "$QUERY" || true)"
  echo "$BASELINE_RESPONSE"
  echo

  echo "[3/4] Request-scoped token via Authorization header"
  HEADER_RESPONSE="$(curl -sS -X POST "${API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${VALID_TOKEN}" \
    -d "$QUERY" || true)"
  echo "$HEADER_RESPONSE"
  echo

  echo "[4/4] Optional alternate header path"
  ALT_HEADER_RESPONSE="$(curl -sS -X POST "${API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -H "X-Databricks-OBO-Token: ${VALID_TOKEN}" \
    -d "$QUERY" || true)"
  echo "$ALT_HEADER_RESPONSE"
  echo

  echo "== Interpretation =="
  echo "- If [3/4] works, request-scoped OBO token intake is working."
  echo "- If [4/4] works, the fallback OBO header is also wired correctly."
  echo "- If [2/4], [3/4], and [4/4] all look the same, that usually means the same identity/permissions are being used."
  echo "- The strongest proof is when header-based calls succeed while fallback or bad-token cases fail."
}

run_precedence_proof() {
  trap cleanup EXIT

  echo "== OBO precedence proof mode =="
  echo "Starting isolated API on ${TEMP_API_URL} with intentionally bad DATABRICKS_TOKEN"
  echo "Logs: ${TEMP_LOG}"
  echo

  env -u DATABRICKS_OBO_TOKEN PORT="${TEMP_PORT}" DATABRICKS_TOKEN="not-a-real-token" PYTHONPATH=. "${PYTHON_BIN}" project/api.py >"${TEMP_LOG}" 2>&1 &
  TEMP_API_PID=$!

  if ! wait_for_health "${TEMP_API_URL}"; then
    echo "[FAIL] Temporary API did not become healthy."
    echo "Check logs: ${TEMP_LOG}"
    exit 1
  fi

  echo "[1/3] No header -> should use bad fallback token"
  echo "Status: $(curl -sS "${TEMP_API_URL}/api/status" || true)"
  NO_HEADER_RESPONSE="$(curl -sS -X POST "${TEMP_API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -d "$QUERY" || true)"
  echo "$NO_HEADER_RESPONSE"
  echo

  echo "[2/3] Authorization header with VALID_TOKEN -> should override bad fallback token"
  HEADER_RESPONSE="$(curl -sS -X POST "${TEMP_API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${VALID_TOKEN}" \
    -d "$QUERY" || true)"
  echo "$HEADER_RESPONSE"
  echo

  echo "[3/3] Alternate OBO header with VALID_TOKEN -> should also override bad fallback token"
  ALT_HEADER_RESPONSE="$(curl -sS -X POST "${TEMP_API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -H "X-Databricks-OBO-Token: ${VALID_TOKEN}" \
    -d "$QUERY" || true)"
  echo "$ALT_HEADER_RESPONSE"
  echo

  echo "== Interpretation =="
  echo "- If [1/3] fails or returns an auth error while [2/3] works, precedence is proven."
  echo "- If [2/3] and [3/3] both work, both header paths override fallback correctly."
  echo "- If all three fail, your token may be invalid, expired, or the endpoint may be unavailable."
}

run_compare_users() {
  local token_a="${TOKEN_A:-}"
  local token_b="${TOKEN_B:-}"
  local query_a="${QUERY_A:-How do I grant SELECT on catalog.sales.orders to analyst_group?}"
  local query_b="${QUERY_B:-How do I grant SELECT on catalog.sales.orders to analyst_group?}"

  if [[ -z "$token_a" || -z "$token_b" ]]; then
    echo "[FAIL] TOKEN_A and TOKEN_B are required for compare-users mode."
    echo "Example: TOKEN_A='<user-a-token>' TOKEN_B='<user-b-token>' bash scripts/verify_obo.sh compare-users"
    exit 1
  fi

  echo "== OBO multi-user compare mode =="
  echo "API_URL=${API_URL}"
  echo

  echo "[1/3] Health check"
  STATUS_RESPONSE="$(curl -sS "${API_URL}/api/status" || true)"
  echo "$STATUS_RESPONSE"
  echo

  if [[ "$STATUS_RESPONSE" != *'"status":"healthy"'* && "$STATUS_RESPONSE" != *'"status": "healthy"'* ]]; then
    echo "[FAIL] API health check did not return healthy."
    echo "Start server: PYTHONPATH=. .venv/bin/python project/api.py"
    exit 1
  fi

  echo "[2/3] User A request"
  USER_A_RESPONSE="$(curl -sS -X POST "${API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${token_a}" \
    -d "{\"query\":\"${query_a}\"}" || true)"
  echo "$USER_A_RESPONSE"
  echo

  echo "[3/3] User B request"
  USER_B_RESPONSE="$(curl -sS -X POST "${API_URL}/api/query" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${token_b}" \
    -d "{\"query\":\"${query_b}\"}" || true)"
  echo "$USER_B_RESPONSE"
  echo

  echo "== Interpretation =="
  echo "- Both responses should show trace.auth_source=request_access_token."
  echo "- If one succeeds and one fails, you have confirmed permission/identity differences."
  echo "- If both succeed with similar outputs, users likely have similar permissions for this query."
  echo "- For stronger separation, test a restricted resource or admin action one user should not access."
}

case "$MODE" in
  basic)
    run_basic_checks
    ;;
  prove-precedence)
    run_precedence_proof
    ;;
  compare-users)
    run_compare_users
    ;;
  *)
    echo "Usage: bash scripts/verify_obo.sh [basic|prove-precedence|compare-users]"
    exit 1
    ;;
esac
