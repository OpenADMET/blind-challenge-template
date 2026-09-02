# OpenTofu remote state backend configuration.
#
# The state bucket and lock table can't live in main.tf's `backend "s3"` block
# (backend blocks don't take variables) and the template ships with placeholders,
# so they're supplied here as partial backend config:
#
#   tofu init -backend-config=backend.hcl \
#     -backend-config="key=<challenge_name>/terraform.tfstate"
#
# 1. Fill in your state bucket + lock table (created once per AWS account,
#    docs/SETUP.md Step 0).
# 2. Commit this file to your challenge repo — the values are not secret, and CI
#    reads it directly. Don't commit real values to the public *template* repo.
#
# `key` is passed separately because it is per-challenge (derived from
# challenge_name). To validate the config without a real backend:
#   tofu init -backend=false

bucket         = "REPLACE_ME-tofu-state"
dynamodb_table = "REPLACE_ME-tofu-locks"
# region       = "us-east-1"   # only if your state bucket is not in main.tf's default region
