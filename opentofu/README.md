# OpenTofu Infrastructure — Blind Challenge

This directory contains [OpenTofu](https://opentofu.org/) (open-source Terraform fork) configuration that provisions all AWS resources required to run the challenge backend.

All resource names are parameterised via the single variable `var.challenge_name` (default `"blind-challenge-template"`) so the same configuration can be reused for future challenges by passing a different value. A plain challenge uses just the slug (e.g. `cyp-challenge`, `pxr-challenge`); to run one for an external org, fold the org directly into the slug (e.g. `openbind-blind-challenge-1`) — there's no separate organisation variable. `var.github_org` and `var.hf_owner` name the GitHub org and Hugging Face owner separately; set all three (`challenge_name`, `hf_owner`, `github_org`) in `variables.tf` and commit them.

### Checklist: copying this repo for a new challenge

Every OpenTofu-managed resource is driven by the variable above, but a few things live **outside** Tofu's reach and won't update themselves. When copying this repo to start a new challenge repo, change all of these:

| What | Where |
|------|-------|
| `challenge_name`, `hf_owner`, `github_org` | `opentofu/variables.tf` — set and commit all three. `challenge_name` must equal the repo name; `deploy.yml` / `deploy-hf-space.yml` read `challenge_name` and `hf_owner` back via `naming-guardrail.yml`'s outputs, and the hf-restarter Lambda's `HF_SPACE_NAME` is wired to `var.challenge_name` directly |
| State backend bucket + lock table | `opentofu/backend.hcl` (shipped with `REPLACE_ME` placeholders) — fill in and commit; passed at `tofu init` via `-backend-config`, and CI reads the same file |
| State backend key | Passed at `tofu init` time via `-backend-config`, not a file to edit — see Step 1 below. Derived from `challenge_name`, so unique per challenge |
| Step 2 deploy role inputs | `opentofu/policies/render.sh` reads them from `variables.tf` + `backend.hcl`; you only pass `AWS_ACCOUNT_ID` — see Step 2 below |

---

## Architecture Overview

### Runtime Flow

```
EventBridge (S3 object-created + rate/cron schedules)
  │
  ├──► regression-evaluator Lambda (container) ──► S3 (read submissions, write scores)
  ├──► classification-evaluator Lambda (container) ──► S3 (read submissions, write scores)
  ├──► structure-evaluator Lambda (container) ──► S3 (read submissions, write scores)
  ├──► leaderboard-generator Lambda (container) ─► S3 (write leaderboard CSV)
  ├──► entries-list-generator Lambda (container) ─► S3 (write cross-track entries CSV)
  └──► hf-restarter Lambda (container) ─► HuggingFace API (POST /spaces/.../restart)

HuggingFace Space (Gradio app)
    │
    ├── submits files ──► S3 (PutObject submissions/*) via IAM user
    └── reads leaderboards, ground truth, scores, submissions
        ◄── S3 (GetObject leaderboard/*, submissions/*, ground_truth/*, scores/*) via IAM user
```

---

## Resources Created

### S3 Bucket — `s3.tf`

| Bucket | Value |
|--------|-------|
| Name | `{challenge_name}` |
| Versioning | Enabled |
| Encryption | AES-256 (SSE-S3) |
| Public access | Fully blocked |
| EventBridge integration | Enabled (`aws_s3_bucket_notification.eventbridge = true`) |

**Bucket policy rules:**

- Deny all requests not using HTTPS (`aws:SecureTransport = false`)
- Deny `s3:DeleteObject` / `s3:DeleteObjectVersion` on `submissions/*` — submission data is immutable

**Lifecycle:**

- Abort incomplete multipart uploads after 7 days

**Guardrail:** the bucket has `lifecycle { prevent_destroy = true }`. You must remove this block and re-run `tofu plan` before any `tofu destroy` will succeed. Set `var.force_destroy = true` if you also need the bucket emptied automatically on destroy (e.g. for dev teardown).

#### S3 Object Layout

Submission upload, scoring, manifests, and leaderboards are all split into three
fully independent tracks — `regression`, `classification`, and `structure` — each
with its own S3 upload prefix and Lambda. `classification` only exists once
`CLASSIFICATION_ENDPOINTS` is non-empty in `backend/config.py`. Regression and
classification still score against the same shared ground-truth dataset (the same
test-set compounds, just different columns).

```
submissions/
  regression/{username}/{submission_id}/
    metadata.json                         ← serialised Submission model
    predictions.parquet                   ← (or .csv)
  classification/{username}/{submission_id}/
    metadata.json
    predictions.parquet                   ← (or .csv)
  structure/{username}/{submission_id}/
    metadata.json
    structures.zip
  submissions/manifest/regression/
    {username}_{submission_id}.parquet    ← Central manifest of submission data for quick lookup
  submissions/manifest/classification/
    {username}_{submission_id}.parquet
  submissions/manifest/structure/
    {username}_{submission_id}.parquet

ground_truth/
  activity-dataset.parquet                ← One shared dataset (both regression and
  activity-identifiers.parquet              classification labels) — regression and
                                             classification both score against this,
                                             filtered to their own columns
  structure-dataset.zip
  structure-identifiers.parquet

scores/all/regression/                    ← Scored against the full (phase 0) test set
  {username}/{submission_id}/
    bootstrap-results.parquet
    averaged-results.parquet

scores/phase_1/regression/                ← Scored against the phase 1 compound subset
  {username}/{submission_id}/
    bootstrap-results.parquet
    averaged-results.parquet

scores/all/classification/
  {username}/{submission_id}/
    bootstrap-results.parquet
    averaged-results.parquet

scores/phase_1/classification/
  {username}/{submission_id}/
    bootstrap-results.parquet
    averaged-results.parquet

scores/all/structure/
  {username}/{submission_id}/
    per-compound-results.parquet
    bootstrap-results.parquet
    averaged-results.parquet

scores/phase_1/structure/
  {username}/{submission_id}/
    per-compound-results.parquet
    bootstrap-results.parquet
    averaged-results.parquet

leaderboard/live/regression/              ← Auto-generated by the leaderboard Lambda
  MA_leaderboard_latest.csv                 from scores/phase_1 during phase 1.
  ENDPOINT_1_leaderboard_latest.csv        One file per real regression endpoint,
  ENDPOINT_2_leaderboard_latest.csv        each ranked on that endpoint's own
  ENDPOINT_3_leaderboard_latest.csv        metrics, plus one
  ENDPOINT_4_leaderboard_latest.csv        macro-ranked
  ... (+ timestamped {endpoint}_leaderboard_{timestamp}.csv per file above)
                                             master (MA). No significance testing on
                                             "live" — see interim/final below.

leaderboard/live/classification/
  MA_leaderboard_latest.csv                 Same shape as regression, but with the
  ENDPOINT_5_leaderboard_latest.csv         classification endpoints + MA. Only present
  ENDPOINT_6_leaderboard_latest.csv         once CLASSIFICATION_ENDPOINTS is non-empty.
  ...

leaderboard/live/structure/
  Structure_leaderboard_latest.csv          Structure has only one endpoint, so it
  Structure_leaderboard_{timestamp}.csv     gets exactly one leaderboard — no "MA".

leaderboard/interim/{regression,classification,structure}/ ← Generated manually from
  MA_leaderboard_latest.csv                 scores/all, submissions before the phase 1
  {endpoint files as above per track}       deadline. CLD/tiers significance testing
  Structure_leaderboard_latest.csv          runs only on each track's master (MA, or
                                             structure's one leaderboard) — never
                                             per-endpoint files.

leaderboard/final/{regression,classification,structure}/   ← Generated manually from
  MA_leaderboard_latest.csv                 scores/all, all submissions, with invalid
  {endpoint files as above per track}       submissions (bad HF username / model
  Structure_leaderboard_latest.csv          report link) removed. Same
                                             significance-testing scope as interim.

entries/                                  ← Full cross-track (regression +
  all_entries_latest.csv                    classification + structure) entries list.
  all_entries_{timestamp}.csv               Generated daily by the entries-list
                                             Lambda; not track-scoped, not limited to
                                             the latest submission per user, and no
                                             date cutoff.
```

---

### IAM — `iam.tf`

#### HuggingFace Space IAM User

| Resource | Name |
|----------|------|
| IAM User | `{challenge_name}-hf-space` |
| Access Key | (sensitive output) |
| Inline policy | `{challenge_name}-hf-space-policy` |

**Policy grants:**

- `s3:PutObject` on `submissions/*`
- `s3:GetObject` on `leaderboard/*` and `submissions/*` (the app reads the leaderboard CSVs and lets a participant see their own past submissions)
- `s3:ListBucket` on the main bucket, limited to `submissions/regression/*`, `submissions/classification/*`, `submissions/structure/*`, and `leaderboard/*` prefixes (including `live`, `interim`, `final`)

Set the resulting credentials as HuggingFace Space secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`).

#### Lambda Execution Role

| Resource | Name |
|----------|------|
| IAM Role | `{challenge_name}-lambda-exec` |
| Managed policy | `AWSLambdaBasicExecutionRole` (CloudWatch Logs) |
| Inline policy | `{challenge_name}-lambda-s3-policy` |

**Lambda S3 policy grants:**

- `s3:GetObject`, `s3:ListBucket` on `submissions/*`, `ground_truth/*`, `scores/*`, `leaderboard/*`, `entries/*`
- `s3:PutObject` on `submissions/*/scores.json`, `submissions/manifest/*`, `scores/*`, `leaderboard/*`, and `entries/*`

Note: the Lambda role cannot delete submissions (the bucket policy denies it for all principals).

#### Lambda Secrets policy

Lambda execution role is also granted:

- `secretsmanager:DescribeSecret`
- `secretsmanager:GetSecretValue`

on the challenge's Secrets Manager secrets (`{challenge_name}-discord-webhook`, `{challenge_name}-hf-token`).

### Secrets Manager — `secrets.tf`

| Secret | Value |
|--------|-------|
| Name | `{challenge_name}-discord-webhook` |
| Purpose | Discord webhook URL used by backend notification code |

| Secret | Value |
|--------|-------|
| Name | `{challenge_name}-hf-token` |
| Purpose | HuggingFace API token used by the hf-restarter Lambda |

Both secret resources are created by OpenTofu as empty containers. Their values are synced by CI after `tofu apply`, from the GitHub secrets `DISCORD_WEBHOOK_URL` and `HF_TOKEN` respectively — never passed through an OpenTofu variable or written to Tofu state.

#### Artifacts Uploader IAM User

| Resource | Name |
|----------|------|
| IAM User | `{challenge_name}-artifacts-uploader` |
| Access Key | (sensitive output) |
| Inline policy | `{challenge_name}-artifacts-uploader-policy` |

**Policy grants (ECR only):**

- `ecr:GetAuthorizationToken`
- push/pull actions on `{challenge_name}-lambda`

Use this identity for Docker image publish operations to ECR. Do **not** use the HF Space IAM user for CI/CD image publishing.

---

### ECR Repository — `ecr.tf`

| Resource | Name |
|----------|------|
| Repository | `{challenge_name}-lambda` |
| Image tag mutability | `MUTABLE` |
| Scan on push | Enabled |

**Repository policy:** allows the `lambda.amazonaws.com` service principal `ecr:BatchGetImage` / `ecr:GetDownloadUrlForLayer`, so Lambda can pull images without a separate IAM grant.

**Lifecycle policy:** expires all but the 5 most recent images (any tag).

Must exist before the first CI push that touches `backend/**` — `deploy.yml` builds and pushes the Lambda image to this repo **before** running `tofu apply`, so on a brand-new challenge repo it has to be created up front via the targeted `tofu apply` in Step 1 below, not left to the normal apply.

---

### Lambda Functions — `lambda.tf`

#### Regression Evaluator

| Attribute | Value |
|-----------|-------|
| Function name | `{challenge_name}-regression-evaluator` |
| Package type | Container image |
| Image URI | `{lambda_ecr_repository_url}@{lambda_image_digest}` when digest is provided, otherwise `{lambda_ecr_repository_url}:{lambda_image_tag}` |
| Handler command | `backend.lambda_handler_regression.handler` (`image_config.command`) |
| Timeout | `var.lambda_timeout` (default 900 s) |
| Memory | `var.lambda_memory_mb` (default 6400 MB) |
| Concurrency | `reserved_concurrent_executions = 5` |
| Env vars | `S3_BUCKET`, `DISCORD_WEBHOOK_SECRET_NAME` |
| Log group | `/aws/lambda/{challenge_name}-regression-evaluator` (30-day retention) |

#### Classification Evaluator

| Attribute | Value |
|-----------|-------|
| Function name | `{challenge_name}-classification-evaluator` |
| Package type | Container image |
| Image URI | `{lambda_ecr_repository_url}@{lambda_image_digest}` when digest is provided, otherwise `{lambda_ecr_repository_url}:{lambda_image_tag}` |
| Handler command | `backend.lambda_handler_classification.handler` (`image_config.command`) |
| Timeout | `var.lambda_timeout` (default 900 s) |
| Memory | `var.lambda_memory_mb` (default 6400 MB) |
| Concurrency | `reserved_concurrent_executions = 5` |
| Env vars | `S3_BUCKET`, `DISCORD_WEBHOOK_SECRET_NAME` |
| Log group | `/aws/lambda/{challenge_name}-classification-evaluator` (30-day retention) |

#### Structure Evaluator

| Attribute | Value |
|-----------|-------|
| Function name | `{challenge_name}-structure-evaluator` |
| Package type | Container image |
| Image URI | `{lambda_ecr_repository_url}@{lambda_image_digest}` when digest is provided, otherwise `{lambda_ecr_repository_url}:{lambda_image_tag}` |
| Handler command | `backend.lambda_handler_structure.handler` (`image_config.command`) |
| Timeout | `var.lambda_timeout` (default 900 s) |
| Memory | `var.lambda_memory_mb` (default 6400 MB) |
| Concurrency | `reserved_concurrent_executions = 5` |
| Env vars | `S3_BUCKET`, `DISCORD_WEBHOOK_SECRET_NAME` |
| Log group | `/aws/lambda/{challenge_name}-structure-evaluator` (30-day retention) |

#### Leaderboard Generator

| Attribute | Value |
|-----------|-------|
| Function name | `{challenge_name}-leaderboard-generator` |
| Package type | Container image |
| Image URI | `{lambda_ecr_repository_url}@{lambda_image_digest}` when digest is provided, otherwise `{lambda_ecr_repository_url}:{lambda_image_tag}` |
| Handler command | `backend.lambda_handler_leaderboard.handler` (`image_config.command`) |
| Timeout | 300 s |
| Memory | 1024 MB |
| Concurrency | `reserved_concurrent_executions = 1` |
| Env vars | `S3_BUCKET`, `DISCORD_WEBHOOK_SECRET_NAME` |
| Log group | `/aws/lambda/{challenge_name}-leaderboard-generator` (30-day retention) |

#### Entries List Generator

| Attribute | Value |
|-----------|-------|
| Function name | `{challenge_name}-entries-list-generator` |
| Package type | Container image |
| Image URI | `{lambda_ecr_repository_url}@{lambda_image_digest}` when digest is provided, otherwise `{lambda_ecr_repository_url}:{lambda_image_tag}` |
| Handler command | `backend.lambda_handler_entries.handler` (`image_config.command`) |
| Timeout | 300 s |
| Memory | 1024 MB |
| Concurrency | `reserved_concurrent_executions = 1` |
| Env vars | `S3_BUCKET`, `DISCORD_WEBHOOK_SECRET_NAME` |
| Log group | `/aws/lambda/{challenge_name}-entries-list-generator` (30-day retention) |

Runs once daily (see EventBridge Schedules below). Writes an unfiltered list of every
valid, scored submission across every track (regression, classification, structure)
to `entries/all_entries_latest.csv` — not a leaderboard, and not limited to each
user's latest submission.

#### HuggingFace Space Restarter

| Attribute | Value |
|-----------|-------|
| Function name | `{challenge_name}-hf-restarter` |
| Package type | Container image |
| Image URI | `{lambda_ecr_repository_url}@{lambda_image_digest}` when digest is provided, otherwise `{lambda_ecr_repository_url}:{lambda_image_tag}` |
| Handler command | `backend.lambda_handler_hf_restart.handler` (`image_config.command`) |
| Timeout | 30 s |
| Memory | 128 MB |
| Env vars | `HF_OWNER`, `HF_SPACE_NAME`, `HF_TOKEN_SECRET_NAME`, `DISCORD_WEBHOOK_SECRET_NAME` |
| Log group | `/aws/lambda/{challenge_name}-hf-restarter` (30-day retention) |

Calls `POST https://huggingface.co/api/spaces/{HF_OWNER}/{HF_SPACE_NAME}/restart` to keep the Space warm.

---

### EventBridge Schedules — `eventbridge.tf`

| Rule | Target | Schedule |
|------|--------|----------|
| `{challenge_name}-regression-object-created` | regression evaluator Lambda | On `Object Created` for `submissions/regression/*.csv` and `submissions/regression/*.parquet` |
| `{challenge_name}-classification-object-created` | classification evaluator Lambda | On `Object Created` for `submissions/classification/*.csv` and `submissions/classification/*.parquet` |
| `{challenge_name}-structure-object-created` | structure evaluator Lambda | On `Object Created` for `submissions/structure/*.zip` |
| `{challenge_name}-leaderboard-generator` | leaderboard generator Lambda | `var.leaderboard_schedule` |
| `{challenge_name}-entries-list-generator` | entries list generator Lambda | `var.entries_list_schedule` |
| `{challenge_name}-hf-restarter` | HF restarter Lambda | `var.hf_restart_schedule` |

Each rule has a corresponding `aws_cloudwatch_event_target` and `aws_lambda_permission`.
Default schedules: `rate(15 minutes)` for leaderboard generation, `cron(0 18 * * ? *)`
(daily, 18:00 UTC) for the entries list, `rate(6 hours)` for the HF restarter.

EventBridge Rules schedule expressions are always UTC and don't shift for daylight
saving. Set `var.entries_list_schedule` to a cron that lands at a sensible local time
for whoever reviews the daily entries list, and remember it will drift by an hour
across your local DST boundary. If exact local-time correctness matters, switch this
one schedule to `aws_scheduler_schedule` (EventBridge Scheduler), which supports a
`schedule_expression_timezone` (e.g. `"Pacific/Auckland"`, `"Europe/London"`) that
handles DST automatically — not used here to keep all schedules on one consistent
mechanism.

---

## Guardrails

| Guardrail | Where | Effect |
|-----------|-------|--------|
| `lifecycle { prevent_destroy = true }` | S3 bucket | `tofu destroy` fails unless block is manually removed first |
| `force_destroy = var.force_destroy` (default `false`) | S3 bucket | Bucket must be empty to destroy unless explicitly overridden |
| `DenyDeleteSubmissions` bucket policy | S3 bucket | No principal can delete objects under `submissions/*` |
| `DenyInsecureTransport` bucket policy | S3 bucket | All non-HTTPS requests denied |
| `reserved_concurrent_executions = 5` | Evaluator Lambdas | Caps concurrent evaluation runs to limit runaway costs |
| Public access block (all four settings) | S3 bucket | Bucket can never be made public |

---

## Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `challenge_name` | `"blind-challenge-template"` | Sole slug used in all resource names — a plain challenge uses just the slug (e.g. `cyp-challenge`), or fold an external org into it (e.g. `openbind-blind-challenge-1`). Must match the GitHub repo name. Independent of `github_org` and `hf_owner`. The one identity value that stays in `variables.tf` |
| `github_org` | `""` | GitHub org/user that owns the repo. Read only by `opentofu/policies/render.sh` (Step 2), no Tofu resource uses it. Set in `variables.tf` and commit |
| `aws_region` | `"us-east-1"` | AWS region |
| `force_destroy` | `false` | Allow destroy to empty + delete the S3 bucket |
| `hf_owner` | `""` | HuggingFace org/username that owns the Space. Set in `variables.tf` and commit; CI reads it back via `naming-guardrail.yml` |
| `leaderboard_schedule` | `"rate(15 minutes)"` | EventBridge rate for leaderboard generator |
| `entries_list_schedule` | `"cron(0 18 * * ? *)"` | EventBridge cron (UTC) for the entries list generator — 18:00 UTC daily; adjust for your reviewers' timezone, see EventBridge Schedules above for the DST caveat |
| `hf_restart_schedule` | `"rate(6 hours)"` | EventBridge rate for HF restarter |
| `lambda_timeout` | `900` | Lambda timeout in seconds (max 900) |
| `lambda_memory_mb` | `6400` | Lambda memory in MB |
| `lambda_image_tag` | `"latest"` | ECR image tag used only when `lambda_image_digest` is empty |
| `lambda_image_digest` | `""` | Immutable ECR digest (`sha256:...`) for all Lambda functions; takes precedence over tag |
| `tags` | `{Project, ManagedBy}` | Default tags applied to all resources; `Challenge` tag is set from `challenge_name` |

## Required GitHub Secrets for Deploy Workflow

See "Where secrets live" at the end of the Setup Walkthrough below for what needs setting, where, and what consumes it.

---

## Setup Walkthrough

This section is written as one strict, in-order walkthrough for setting up a **new** challenge — do the steps top to bottom. Run all `tofu` commands from `opentofu/` and all other commands from the repository root, unless a step says otherwise.

### Prerequisites (local machine)

- The [OpenTofu](https://opentofu.org/) CLI (`tofu`).
- AWS CLI credentials for **your own** operator identity — not any of the IAM users this config creates — configured via `aws configure`, an SSO profile, or the standard `AWS_PROFILE` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables, with permissions to create IAM users/roles, S3 buckets, ECR repos, Lambda functions, EventBridge rules, and Secrets Manager secrets (an admin/poweruser role is the simplest option). This is whatever identity you already use to run plain `aws` CLI commands like the ones below.
- No `TF_VAR_*` environment variable is required — no OpenTofu variable holds a secret value (see "Where secrets live" at the end of this section).
- `challenge_name` / `hf_owner` / `github_org` set in `opentofu/variables.tf`, and the `REPLACE_ME` placeholders in `opentofu/backend.hcl` replaced with your state bucket + lock table. Commit both — CI reads them straight from the repo.

### Step 0 — Account-wide setup (do once per AWS account)

Skip this whole step if you've already stood up a challenge in this AWS account — the state backend and OIDC provider are shared, not per-challenge.

**Remote state backend** — an S3 bucket + a DynamoDB lock table, whose names you choose once and record in each challenge repo's `opentofu/backend.hcl`. They must exist before `tofu init` will work. Substitute your chosen names below:

```bash
STATE_BUCKET=your-tofu-state      # must match backend.hcl
LOCK_TABLE=your-tofu-locks        # must match backend.hcl

# State bucket
aws s3api create-bucket --bucket "$STATE_BUCKET" --region us-east-1
aws s3api put-bucket-versioning \
  --bucket "$STATE_BUCKET" \
  --versioning-configuration Status=Enabled

# Lock table
aws dynamodb create-table \
  --table-name "$LOCK_TABLE" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

**GitHub OIDC provider** — lets any repo's Actions workflow assume an AWS role without static keys:

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

**Optional — shared developer IAM group** — so people setting up challenges don't need admin. `policies/challenge-developers-policy.json.tmpl` is a scoped policy document covering every `*challenge*` resource plus the shared state backend; render it with `policies/render.sh` and attach it to an IAM group. See [`docs/SETUP.md`](../docs/SETUP.md) Step 0 for the commands.

### Step 1 — Create this challenge's IAM identities, get the HF Space credentials

```bash
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT/opentofu"
CHALLENGE_NAME=blind-challenge-template   # match variables.tf's challenge_name default for this repo
tofu init \
  -backend-config=backend.hcl \
  -backend-config="key=${CHALLENGE_NAME}/terraform.tfstate"
```

`backend.hcl` supplies the state `bucket` and lock `dynamodb_table`; `key` is passed separately because it's per-challenge. None of the three can be set inside `main.tf`'s `backend` block (OpenTofu limitation — backend blocks don't support interpolation), and the template ships without real names. Commit `backend.hcl` to your challenge repo; CI runs the identical `tofu init -backend-config=backend.hcl -backend-config="key=..."`. Every challenge repo must use a **different** key, or `tofu apply` will read/write another challenge's real state.

Targeted `tofu apply` for the HF Space IAM user and the ECR repository (both required) — the IAM user target also creates the S3 bucket, since the policy attached to this user references it; the ECR repo must exist before the first CI push, since `deploy.yml` builds and pushes the Lambda image to it **before** running `tofu apply`:

```bash
tofu apply \
  -target=aws_iam_user.hf_space \
  -target=aws_iam_access_key.hf_space \
  -target=aws_iam_user_policy.hf_space \
  -target=aws_ecr_repository.lambda \
  -target=aws_ecr_repository_policy.lambda \
  -target=aws_ecr_lifecycle_policy.lambda

tofu output -raw hf_space_access_key_id; echo
tofu output -raw hf_space_secret_access_key; echo
tofu output -raw s3_bucket_name; echo
```

**Set these three values now as HuggingFace Space secrets** (in the Space's own Settings → Variables and secrets — this is on huggingface.co, not a GitHub secret): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`.

Optional — only needed if you plan to publish Lambda images manually/locally instead of relying on CI (see "Manual/local operations" below):

```bash
tofu apply \
  -target=aws_iam_user.artifacts_uploader \
  -target=aws_iam_access_key.artifacts_uploader \
  -target=aws_iam_user_policy.artifacts_uploader

tofu output -raw artifacts_uploader_access_key_id; echo
tofu output -raw artifacts_uploader_secret_access_key; echo
```

### Step 2 — Create the GitHub Actions deploy role, set GitHub secrets

Policy **templates** are included in this repo:

- `opentofu/policies/github-actions-oidc-trust-policy.json.tmpl` — this step
- `opentofu/policies/github-actions-deploy-policy.json.tmpl` — this step
- `opentofu/policies/challenge-developers-policy.json.tmpl` — the optional Step 0 developer group (`render.sh` renders all three at once)

The Step 2 pair are rendered and applied by hand via the AWS CLI below — **not** OpenTofu resources. That's intentional: this role is what lets GitHub Actions run `tofu apply` in the first place, so it can't be created by that same `tofu apply` (bootstrapping circularity). Keep it as manual, pre-pushed setup rather than folding it into the Tofu config.

Render the concrete JSON for this challenge with `opentofu/policies/render.sh` (needs `envsubst`, from the `gettext` package). It reads `challenge_name` / `github_org` from `variables.tf` and the bucket / table from `backend.hcl`, so the only input is the account ID. Output goes to `opentofu/policies/.rendered/` — gitignored, nothing challenge-specific gets committed:

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text) \
  opentofu/policies/render.sh
```

(Export `CHALLENGE_NAME` / `GITHUB_ORG` / `STATE_BUCKET` / `LOCK_TABLE` to override the values read from those files; `AWS_REGION` is optional and defaults to `us-east-1`.)

`variables.tf`'s `challenge_name` must match this repo's actual name and `github_org/challenge_name` must be the real repo path — the trust policy's `job_workflow_ref`/`sub` conditions derive the allowed repo path as `${GITHUB_ORG}/${CHALLENGE_NAME}`, and the permissions policy is scoped to `${CHALLENGE_NAME}-*` resource names, so a mismatch points the rendered role at AWS resources Tofu isn't actually creating. To run a challenge for an external org, fold the org into `challenge_name` itself (e.g. `openbind-blind-challenge-1`) — `github_org` stays the org that actually owns the repo.

Each challenge repo gets its **own** dedicated deploy role — the trust policy and permissions policy are scoped only to `${GITHUB_ORG}/${CHALLENGE_NAME}` and its resources, and do not touch other challenges' deploy roles or access. Do not reuse a single shared role name across challenges; that would require widening the trust policy to an array of repos and merging every challenge's ARNs into one permissions policy.

Run this from the repository root so the rendered-file paths resolve correctly:

```bash
aws iam create-role \
  --role-name blind-challenge-template-github-actions-deploy \
  --assume-role-policy-document file://opentofu/policies/.rendered/github-actions-oidc-trust-policy.json

aws iam put-role-policy \
  --role-name blind-challenge-template-github-actions-deploy \
  --policy-name blind-challenge-template-github-actions-deploy \
  --policy-document file://opentofu/policies/.rendered/github-actions-deploy-policy.json
```

(Substitute `blind-challenge-template` in the role/policy names above for `${CHALLENGE_NAME}` if you rendered for a different challenge.)

If the role already exists, skip `create-role` and run only `put-role-policy` to update permissions — **double-check `--role-name` matches the challenge you actually intend to update first**; this name isn't derived from any variable at this step, so a stale/copy-pasted name here will silently overwrite a different challenge's deploy role permissions instead of creating a new one.

If you already created the role but didn't note its ARN, retrieve it with:

```bash
aws iam get-role --role-name blind-challenge-template-github-actions-deploy --query Role.Arn --output text
```

> Tip: if you rename the workflow file or deploy from a different branch, update `job_workflow_ref` and `sub` conditions in the trust policy.

**Set these values now in the repo's Settings → Secrets and variables → Actions.**

Secrets:

| Secret | Value |
|--------|-------|
| `AWS_DEPLOY_ROLE_ARN` | The role ARN returned by `create-role` above |
| `HF_TOKEN` | A HuggingFace API token with write access to the Space |
| `DISCORD_WEBHOOK_URL` | The Discord webhook URL for submission notifications |

Variables — all optional (the state bucket/table, challenge name and HF owner come from the committed `backend.hcl` / `variables.tf`):

| Variable | Value | Default if unset |
|----------|-------|------------------|
| `AWS_REGION` | Deploy region | `us-east-1` |
| `HF_INTERNAL_TESTING_SPACE` | `owner/space` of a shared testing Space for the `hf_testing` branch | that job is skipped |
| `NAME_MUST_CONTAIN` | Substring `challenge_name` must contain (naming guardrail) | check skipped |

The three secrets must exist, and the deploy role above must already exist in AWS, **before** the first push to `main` that touches `backend/**` or `opentofu/**` — `deploy.yml` has nothing to fall back to if the OIDC role it tries to assume doesn't exist yet.

### Step 3 — Push to `main`, let CI take over

The workflow `.github/workflows/deploy.yml` auto-deploys on pushes to `main` (including PR merges):

- If `backend/**` changes: build + push Lambda image to ECR with tags `sha-$COMMIT_SHA` and `latest`, then deploy Lambdas using the pushed **digest**
- If `opentofu/**` changes: run `tofu init/plan/apply`
- If both change: build image once, then apply infra + function updates in one `tofu apply`
- After `tofu apply` succeeds, CI syncs `DISCORD_WEBHOOK_URL` and `HF_TOKEN` into their Secrets Manager secrets automatically — no manual `put-secret-value` needed once this is working.

### Manual/local operations (optional — not needed for the CI path above)

Build and push the Lambda image locally using the `artifacts_uploader` credentials from Step 1:

```bash
cd "$ROOT"
CHALLENGE_NAME=blind-challenge-template
AWS_REGION=us-east-1
ECR_REPO="${CHALLENGE_NAME}-lambda"
ECR_REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com
COMMIT_SHA=$(git rev-parse --short=12 HEAD)

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

docker build \
  --platform linux/amd64 \
  -f backend/Dockerfile.lambda \
  -t "${ECR_REGISTRY}/${ECR_REPO}:sha-${COMMIT_SHA}" \
  -t "${ECR_REGISTRY}/${ECR_REPO}:latest" \
  .

docker push "${ECR_REGISTRY}/${ECR_REPO}:sha-${COMMIT_SHA}"
docker push "${ECR_REGISTRY}/${ECR_REPO}:latest"
```

By default this tags the image as `sha-$COMMIT_SHA` and also pushes `latest`, matching what CI does in `deploy.yml`. If Lambda reports `image manifest ... is not supported`, rebuild with `--platform linux/amd64` as above (Apple Silicon builds `arm64` by default, which Lambda can't run).

To print the digest for a specific tag:

```bash
aws ecr describe-images \
  --repository-name "${ECR_REPO}" \
  --image-ids imageTag=sha-${COMMIT_SHA} \
  --query 'imageDetails[0].imageDigest' \
  --output text
```

Run a full plan/apply locally instead of via CI:

```bash
cd "$ROOT/opentofu"
tofu plan -var="lambda_image_digest=sha256:..."   # digest-first (recommended, immutable)
tofu plan -var="lambda_image_tag=v1"               # tag-based (fallback)
tofu apply -var="lambda_image_digest=${DIGEST}"
```

Each challenge repo has its own `challenge_name` in `variables.tf` and its own state `key`, so `tofu` commands need no per-challenge flags — just run them from that repo's `opentofu/` directory.

Push the Discord webhook / HF token values into their (already-created) Secrets Manager containers manually, if not relying on CI's sync step:

```bash
# Discord webhook — store as a plain URL string, not a JSON object
aws secretsmanager put-secret-value \
  --secret-id "$(tofu output -raw discord_webhook_secret_name)" \
  --secret-string "${DISCORD_WEBHOOK_URL}"

# HF token
aws secretsmanager put-secret-value \
  --secret-id "$(tofu output -raw hf_token_secret_name)" \
  --secret-string "${HF_TOKEN}"
```

Verify a secret exists / inspect its value:

```bash
aws secretsmanager describe-secret --secret-id "${SECRET_NAME}"
aws secretsmanager get-secret-value --secret-id "${SECRET_NAME}" --query SecretString --output text
```

If you rotate or overwrite either secret, publish a new Lambda version or update the function configuration so new execution environments pick up the fresh value instead of a warm cached copy.

All sensitive outputs, for reference:

```bash
tofu output -raw hf_space_access_key_id; echo
tofu output -raw hf_space_secret_access_key; echo
tofu output -raw artifacts_uploader_access_key_id; echo
tofu output -raw artifacts_uploader_secret_access_key; echo
```

### Where secrets live

| Secret | Stored in | Consumed by |
|--------|-----------|-------------|
| `AWS_DEPLOY_ROLE_ARN` | GitHub Secrets (this repo) | `deploy.yml`, via OIDC role assumption — no static keys |
| `HF_TOKEN` (GitHub secret) | GitHub Secrets (this repo) | Synced by `deploy.yml` into the `{challenge_name}-hf-token` Secrets Manager secret |
| `DISCORD_WEBHOOK_URL` | GitHub Secrets (this repo) | Synced by `deploy.yml` into the `{challenge_name}-discord-webhook` Secrets Manager secret |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `S3_BUCKET` | **HuggingFace Space secrets** (Space Settings on huggingface.co — not GitHub) | The Gradio app (`hf_space/`) at runtime, for direct S3 access |

Non-secret per-challenge values are committed in the repo: `challenge_name` / `hf_owner` / `github_org` in `opentofu/variables.tf`, and the state bucket / lock table in `opentofu/backend.hcl`. CI reads both directly. The only GitHub Actions variables are optional overrides (`AWS_REGION`, `HF_INTERNAL_TESTING_SPACE`, `NAME_MUST_CONTAIN`).

Nothing above is ever passed as an OpenTofu variable or written into Tofu state — Secrets Manager secrets are created by Tofu as empty containers, and their values are populated separately (by CI, or manually as shown above).

> **Note:** The remote state `bucket`, `dynamodb_table`, and `key` are all supplied at `tofu init` via `-backend-config` (`backend.hcl` for the first two, a per-challenge `key=...` flag for the third) — none are stored in `main.tf`.

---

## Teardown

This section covers destroying the **Tofu-managed** resources only (S3 bucket, both IAM users, the Lambda exec role, all Lambda functions, the ECR repo, EventBridge rules, and both Secrets Manager secrets). It does **not** cover the GitHub Actions deploy role (created manually in Step 2, outside Tofu state), the HuggingFace Space, the Discord channels, or the GitHub repo itself — see the Teardown section of [`docs/SETUP.md`](../docs/SETUP.md) for the full checklist covering all of those, plus what to intentionally leave alone (the shared account-wide bootstrap resources from Step 0).

The S3 bucket has `prevent_destroy = true`. To destroy the stack:

1. Remove the `lifecycle { prevent_destroy = true }` block from `s3.tf`
2. If the bucket is not empty, also set `force_destroy = true`
3. The ECR repository has no `force_delete`, so `tofu destroy` fails on it if any images were ever pushed. Empty it first:

   ```bash
   aws ecr batch-delete-image \
     --repository-name "${CHALLENGE_NAME}-lambda" \
     --image-ids "$(aws ecr list-images --repository-name "${CHALLENGE_NAME}-lambda" --query 'imageIds[*]' --output json)"
   ```

4. From `opentofu/`, run destroy:

   ```bash
   tofu destroy \
     -var="force_destroy=true"   # only if bucket is non-empty
   ```

5. Revert the `s3.tf` edit from step 1 (`git checkout opentofu/s3.tf`) if this repo clone is going to stick around for any reason; otherwise this is moot since the whole repo is typically deleted as part of teardown.
