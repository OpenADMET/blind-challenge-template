#!/usr/bin/env bash
# Renders every *.json.tmpl in this directory into gitignored output under
# policies/.rendered/, substituting the values below. These policies are applied
# by hand via the AWS CLI (not OpenTofu) to avoid bootstrapping circularity —
# they are what let a developer / CI create everything else. Currently:
#
#   challenge-developers-policy.json.tmpl          -> docs/SETUP.md Step 0
#       Account-wide policy for an IAM group of challenge developers, scoped to
#       *challenge* resources. Rendered here but attached once per account.
#   github-actions-oidc-trust-policy.json.tmpl     -> docs/SETUP.md Step 2
#   github-actions-deploy-policy.json.tmpl         -> docs/SETUP.md Step 2
#       The per-challenge GitHub Actions deploy role.
#
# CHALLENGE_NAME / GITHUB_ORG are read from ../variables.tf and STATE_BUCKET /
# LOCK_TABLE from ../backend.hcl when those files are filled in, so normally the
# only thing you must supply is AWS_ACCOUNT_ID. Any of them can be overridden by
# exporting it.
#
# Usage:
#   AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text) ./render.sh
#   # or override everything explicitly:
#   CHALLENGE_NAME=x GITHUB_ORG=y AWS_ACCOUNT_ID=123456789012 \
#   STATE_BUCKET=x-tofu-state LOCK_TABLE=x-tofu-locks ./render.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENTOFU_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pull `default = "..."` out of a `variable "<name>" { ... }` block in a .tf file.
tf_default() {
  awk -v want="$1" '
    $0 ~ "variable[ \t]+\"" want "\"" { inblock = 1 }
    inblock && /default[ \t]*=/ {
      if (match($0, /"[^"]*"/)) { print substr($0, RSTART + 1, RLENGTH - 2) }
      exit
    }
    inblock && /^}/ { inblock = 0 }
  ' "$2" 2>/dev/null || true
}

# Pull `<key> = "..."` out of an .hcl file.
hcl_value() {
  sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$2" 2>/dev/null | head -1
}

CHALLENGE_NAME="${CHALLENGE_NAME:-$(tf_default challenge_name "${OPENTOFU_DIR}/variables.tf")}"
GITHUB_ORG="${GITHUB_ORG:-$(tf_default github_org "${OPENTOFU_DIR}/variables.tf")}"
if [[ -f "${OPENTOFU_DIR}/backend.hcl" ]]; then
  STATE_BUCKET="${STATE_BUCKET:-$(hcl_value bucket "${OPENTOFU_DIR}/backend.hcl")}"
  LOCK_TABLE="${LOCK_TABLE:-$(hcl_value dynamodb_table "${OPENTOFU_DIR}/backend.hcl")}"
fi
AWS_REGION="${AWS_REGION:-us-east-1}"

: "${CHALLENGE_NAME:?CHALLENGE_NAME is required (not found in variables.tf; export it)}"
: "${GITHUB_ORG:?GITHUB_ORG is required (not found in variables.tf; export it)}"
: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID is required (e.g. \$(aws sts get-caller-identity --query Account --output text))}"
: "${STATE_BUCKET:?STATE_BUCKET is required (not found in backend.hcl; export it)}"
: "${LOCK_TABLE:?LOCK_TABLE is required (not found in backend.hcl; export it)}"

case "${STATE_BUCKET}${LOCK_TABLE}${GITHUB_ORG}" in
  *REPLACE_ME*)
    echo "error: fill in the REPLACE_ME placeholders in backend.hcl / variables.tf first" >&2
    exit 1 ;;
esac

if ! command -v envsubst >/dev/null 2>&1; then
  echo "error: envsubst not found (part of the 'gettext' package)" >&2
  exit 1
fi

OUT_DIR="${SCRIPT_DIR}/.rendered"
mkdir -p "${OUT_DIR}"

export CHALLENGE_NAME GITHUB_ORG AWS_ACCOUNT_ID AWS_REGION STATE_BUCKET LOCK_TABLE
VARS='${CHALLENGE_NAME} ${GITHUB_ORG} ${AWS_ACCOUNT_ID} ${AWS_REGION} ${STATE_BUCKET} ${LOCK_TABLE}'

echo "challenge=${CHALLENGE_NAME} org=${GITHUB_ORG} account=${AWS_ACCOUNT_ID} region=${AWS_REGION}"
echo "state bucket=${STATE_BUCKET} lock table=${LOCK_TABLE}"

for tmpl in "${SCRIPT_DIR}"/*.json.tmpl; do
  out_name="$(basename "${tmpl}" .tmpl)"
  envsubst "${VARS}" < "${tmpl}" > "${OUT_DIR}/${out_name}"
  echo "rendered ${OUT_DIR}/${out_name}"
done
