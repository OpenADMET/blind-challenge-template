# Blind Challenge Setup

Standard operating procedure for standing up a **new blind challenge** from this template: a new challenge repository, its AWS backend, and its Hugging Face frontend.

This document is org-agnostic — it was built and is used by [OpenADMET](https://openadmet.org) but carries no account-specific values. If your organisation keeps an internal runbook with those (state bucket names, an IAM group, a Discord server, a shared testing Space), follow it alongside this one; the two are meant to line up step for step.

> **Internal (OpenADMET):** those account-specific values and the one-time Step 0 bootstrap are in the *Blind Challenge Setup SOP* in Notion — follow it alongside this guide.

`opentofu/README.md` is the deeper reference for what each Tofu resource does and the full variable list. Wherever a step says "see `opentofu/README.md`", that's what it means.

## Architectural design choices

The infrastructure is built on four pillars:

- **Decoupled frontend & backend.** The participant-facing UI runs on Hugging Face Spaces (Docker SDK); all uploaded submissions, ground-truth data, and leaderboards are processed and stored in an isolated AWS account.
- **Scoped IAM via OIDC.** No long-lived AWS keys in GitHub. Each challenge repo assumes a dedicated AWS IAM role scoped strictly to that challenge's resources, via OpenID Connect.
- **Stateful auditing & locking.** OpenTofu state lives in an S3 bucket with a DynamoDB lock table. The bucket and table are shared account-wide; each challenge gets its own state *file* (a distinct `key`), never shared.
- **Single-variable resource naming.** Every AWS resource name derives from one OpenTofu variable, `challenge_name` — a plain challenge uses just the slug (e.g. `cyp-challenge`); to run one for an external org, fold the org into the slug (e.g. `openbind-blind-challenge-1`). This lets an optional shared IAM group/policy, scoped with a `*challenge*` wildcard, cover any challenge's resources without per-challenge IAM setup.

## What lives where (the parameterisation model)

The template ships without organisation-specific values. Your challenge repo commits them, in **two files**, and CI reads both straight from the repo:

| Where | Holds | Notes |
|-------|-------|-------|
| `opentofu/variables.tf` | `challenge_name`, `hf_owner`, `github_org` | commit these; `challenge_name` must equal the repo name. The naming guardrail parses `challenge_name` and `hf_owner` and hands them to the deploy workflows |
| `opentofu/backend.hcl` (shipped with `REPLACE_ME` placeholders) | state `bucket`, lock `dynamodb_table` | fill in and commit — the values aren't secret, and CI's `tofu init` reads this file |
| GitHub Actions **Secrets** | `AWS_DEPLOY_ROLE_ARN`, `HF_TOKEN`, `DISCORD_WEBHOOK_URL` | the only per-repo GitHub setup |
| GitHub Actions **Variables** | *(all optional)* `AWS_REGION`, `HF_INTERNAL_TESTING_SPACE`, `NAME_MUST_CONTAIN` | overrides only; unset is fine |

`opentofu/policies/render.sh` (Step 2) reads `challenge_name` / `github_org` from `variables.tf` and the bucket / table from `backend.hcl`, so the only thing you pass it is `AWS_ACCOUNT_ID`.

## Requirements

### Platform and organisation access

- **GitHub:** Membership in the org that will own the challenge repo, with rights to create repos and manage Actions secrets/variables.
- **AWS:** Credentials for an identity that can create IAM users/roles, S3 buckets, ECR repos, Lambda functions, EventBridge rules, and Secrets Manager secrets. An admin/power-user role is simplest. Full admin is only needed for the one-time **Step 0** account bootstrap (state bucket, lock table, OIDC provider); day-to-day challenge setup can run under a narrower scoped policy if your org has one. Local AWS CLI credentials must be configured (`~/.aws/credentials`, an SSO profile, or `AWS_PROFILE` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
- **Hugging Face:** A seat in the HF org that will own the Space, with rights to create Spaces in that namespace.
- **Discord** (optional, for submission notifications): admin on the server, to manage webhooks and channels.

### Local environment and tooling

- **AWS CLI** — installed and authenticated.
- **OpenTofu** — v1.6+ on your path.
- **envsubst** (from the `gettext` package) — used by `opentofu/policies/render.sh`. `brew install gettext` on macOS.
- **Python** — a `conda`/`mamba` env from `environment.yml` (`conda env create -f environment.yml`), which includes `pytest`.
- **Git LFS** — for the challenge logo. `brew install git-lfs`, then `git lfs install` once per machine.

Check everything is present:

```bash
aws --version
tofu --version
envsubst --version
python --version
git lfs --version
```

## Procedure

Read the whole SOP before starting. Several steps produce a value (an IAM role ARN, access keys) that a later step consumes. This SOP does not cover data cleaning or staging the ground-truth datasets.

### Create the challenge repo

The challenge repo stores all code and drives CI/CD. On pushes/merges to `main`, GitHub Actions deploys the HF frontend and the AWS backend.

1. Clone the template (or your fork) and rename the directory to the new challenge name:

   ```bash
   git clone https://github.com/OpenADMET/blind-challenge-template.git
   mv blind-challenge-template <your-challenge-name>
   cd <your-challenge-name>
   ```

2. Remove the template's history: `rm -rf .git`

3. Enable the deploy workflows. `.github/workflows/backend-tests.yml` already runs (on the template and on your copy). The three that actually deploy something ship under `.github/workflows.disabled/` so they don't fire against the template's placeholder infra; move them into place:

   ```bash
   mv .github/workflows.disabled/*.yml .github/workflows/
   rmdir .github/workflows.disabled
   ```

4. **Choose the challenge name (`challenge_name`).** e.g. `cyp-challenge`. Rules:
   - It must equal the GitHub repo name (step 11). The OIDC trust policy derives the allowed repo path from it.
   - If you set the `NAME_MUST_CONTAIN` Actions variable (see below), the name must contain that substring — OpenADMET sets it to `challenge` so a shared `*challenge*` IAM policy covers the resources. Leave the variable unset to skip that check.

5. **Update `hf_space/`** for your challenge — mainly `config.py` and `README.md` (endpoint definitions, links, page content). `leaderboards.py` may need minor tweaks. Add the logo as `hf_space/_static/challenge_logo.png` (not shipped in the template — Git LFS pointers break when the `.git` dir is reinitialised).

6. **Update `backend/`, `tests/`, and `README.md`** as needed. Work through every `TODO` (see the repo README for the Todo Tree tip), then run `pytest tests/backend/`.

7. **Set the challenge identity** in `opentofu/variables.tf`: `challenge_name` (your slug from step 4, must equal the repo name), `hf_owner` (HF org/user), `github_org` (the org that will own the repo). No other `opentofu/*.tf` file needs editing — every resource name derives from `challenge_name`. See `opentofu/README.md`'s "Checklist: copying this repo for a new challenge".

8. **Fill in `opentofu/backend.hcl`** — replace the `REPLACE_ME` placeholders with your state bucket + lock table (from AWS Step 0). Commit it along with the `variables.tf` edits; the names aren't secret and CI reads `backend.hcl` directly.

9. Complete the **AWS setup** (below).
10. Create the **Hugging Face Space** and **Discord channel** (below).
11. **Create the GitHub repo:**
    1. Name it to exactly match `challenge_name`.
    2. Settings → Secrets and variables → Actions. Add the three **secrets** — that's the whole GitHub setup:
       - `AWS_DEPLOY_ROLE_ARN` (from AWS Step 3)
       - `DISCORD_WEBHOOK_URL` (from the Discord step)
       - `HF_TOKEN` (from the Hugging Face token step)

       Optionally set **variables** `AWS_REGION` (deploy region, default `us-east-1`), `HF_INTERNAL_TESTING_SPACE` (`owner/space` for the `hf_testing` branch job), `NAME_MUST_CONTAIN` (substring the challenge name must contain). Everything else comes from `variables.tf` and `backend.hcl` in the repo.

12. **Deal with the template-project files.** Several files describe *the template project*, not your challenge. If you're not running your challenge repo as its own open-source project, delete them:

    ```bash
    rm SECURITY.md SUPPORT.md CHANGELOG.md CITATION.cff CONTRIBUTING.md CODE_OF_CONDUCT.md .github/PULL_REQUEST_TEMPLATE.md
    ```

    - `SECURITY.md` — **do not just leave it**: it routes vulnerability reports to OpenADMET, misdirecting anything about *your* deployment. Delete it or point it at your own channel.
    - `CODE_OF_CONDUCT.md` / `CONTRIBUTING.md` / `.github/PULL_REQUEST_TEMPLATE.md` — keep only if you take external contributions; then replace the enforcement contact and the template's dev-setup / DCO wording with your own.
    - `CITATION.cff` — replace with your project's citation metadata if you want a "Cite this repository" button, else delete.
    - `SUPPORT.md` / `CHANGELOG.md` — the template's; start your own or delete.
    - `LICENSE` — **keep.** Your repo stays under Apache-2.0; add your own copyright line if you like.

    Edit, don't delete:

    - `README.md` — rewrite for your challenge (step 6).
    - `docs/SETUP.md` — remove the "Internal (OpenADMET)" note near the top; the rest is a useful reference.
    - `environment.yml` — change `name: blind-challenge-template` to your challenge slug.
    - `.github/ISSUE_TEMPLATE/config.yml` — repoint the Discussions / security URLs at your repo (or delete for plain issue templates).

13. Re-initialise git:

   ```bash
   git init
   git add .
   git commit -m "initial commit"
   ```

14. Push to GitHub. Four workflows run: the naming guardrail, backend unit tests, OpenTofu apply (AWS), and the HF Space deploy (fails until the Space exists).

## Setting up the AWS backend

### Step 0: First-time bootstrap (once per AWS account)

Skip this whole step if you've already stood up a challenge in this AWS account — the state backend and OIDC provider are shared.

**Remote state backend.** Choose an S3 bucket name and a DynamoDB table name once for the account, then create them. Each challenge repo records these two names in its `opentofu/backend.hcl` (step 8 of the procedure above):

```bash
STATE_BUCKET=<your-tofu-state-bucket>
LOCK_TABLE=<your-tofu-lock-table>
AWS_REGION=us-east-1

aws s3api create-bucket --bucket "$STATE_BUCKET" --region "$AWS_REGION"
aws s3api put-bucket-versioning \
  --bucket "$STATE_BUCKET" \
  --versioning-configuration Status=Enabled

aws dynamodb create-table \
  --table-name "$LOCK_TABLE" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

**GitHub OIDC provider:**

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

**Optional — shared developer IAM group/policy.** If several people will set up challenges, give them a scoped IAM group instead of admin. `opentofu/policies/challenge-developers-policy.json.tmpl` is a ready-made policy document: it grants management of every `*challenge*` resource (S3, Lambda, EventBridge, ECR, Secrets Manager, the per-challenge IAM identities and deploy role) plus read/write on the shared state backend, and denies destructive changes to that backend. Render it with the same script Step 2 uses, then create the group once per account:

```bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text) \
CHALLENGE_NAME=placeholder GITHUB_ORG=placeholder \
STATE_BUCKET="$STATE_BUCKET" LOCK_TABLE="$LOCK_TABLE" \
  opentofu/policies/render.sh   # also renders the Step 2 templates; ignore those here

aws iam create-policy \
  --policy-name challenge-developers \
  --policy-document file://opentofu/policies/.rendered/challenge-developers-policy.json

aws iam create-group --group-name challenge-developers
aws iam attach-group-policy \
  --group-name challenge-developers \
  --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/challenge-developers"
# then: aws iam add-user-to-group --group-name challenge-developers --user-name <dev>
```

The policy keys off the literal substring `challenge` in resource names. Set the `NAME_MUST_CONTAIN=challenge` Actions variable so the naming guardrail enforces it; if you scope the policy to a different substring, match `NAME_MUST_CONTAIN` to it. `CHALLENGE_NAME` / `GITHUB_ORG` don't affect this policy (it globs `*challenge*`) but `render.sh` still requires them — any value works when running Step 0 standalone; run from a configured challenge repo and it reads them from `variables.tf` / `backend.hcl` automatically.

### Step 1: Create this challenge's IAM identities, get the HF Space credentials

From the repo root:

```bash
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT/opentofu"
CHALLENGE_NAME=<your-challenge-name>   # must match variables.tf's challenge_name

tofu init \
  -backend-config=backend.hcl \
  -backend-config="key=${CHALLENGE_NAME}/terraform.tfstate"
```

`backend.hcl` supplies the state `bucket` and lock `dynamodb_table`; `key` is passed separately because it's per-challenge. **Every challenge must use a different `key`**, or `tofu apply` will read/write another challenge's real state.

This first `tofu init` generates `opentofu/.terraform.lock.hcl`. The template ships without one; **commit the one you just generated to your challenge repo** — it pins the provider versions and records their checksums, so CI and every maintainer resolve the same ones.

Then a targeted `tofu apply` to create the HF Space IAM user (which also creates the S3 bucket, since its policy references the bucket) and the ECR repository (CI pushes the Lambda image before running `tofu apply`, so it must exist first). Run `tofu` commands from `opentofu/`:

```bash
tofu apply \
  -target=aws_iam_user.hf_space \
  -target=aws_iam_access_key.hf_space \
  -target=aws_iam_user_policy.hf_space \
  -target=aws_ecr_repository.lambda \
  -target=aws_ecr_repository_policy.lambda \
  -target=aws_ecr_lifecycle_policy.lambda

tofu output -raw hf_space_access_key_id && echo
tofu output -raw hf_space_secret_access_key && echo
tofu output -raw s3_bucket_name && echo
```

Copy the access key, secret, and bucket name into the Hugging Face Space secrets (below).

### Step 2: (Optional) Upload ground-truth files

A good point to push ground truth to the bucket. Skip if you don't have it yet (live scoring will fail until you do; previous challenges used dummy data for early testing).

```bash
BUCKET=$(tofu output -raw s3_bucket_name)
PREFIX="ground_truth"
aws s3 cp "activity-dataset.parquet"     "s3://${BUCKET}/${PREFIX}/activity-dataset.parquet"     --only-show-errors
aws s3 cp "activity-identifiers.parquet" "s3://${BUCKET}/${PREFIX}/activity-identifiers.parquet" --only-show-errors
aws s3 cp "structure-dataset.zip"        "s3://${BUCKET}/${PREFIX}/structure-dataset.zip"        --only-show-errors
aws s3 cp "structure-identifiers.parquet" "s3://${BUCKET}/${PREFIX}/structure-identifiers.parquet" --only-show-errors
```

### Step 3: Create the GitHub Actions deploy role, set GitHub secrets

Each challenge repo gets its **own** dedicated deploy role, scoped only to that challenge's resources. Do not share one role across challenges.

The policies are templates (`opentofu/policies/*.json.tmpl`) rendered per-challenge by `opentofu/policies/render.sh` — nothing is hand-edited or committed. `render.sh` reads `challenge_name` / `github_org` from `opentofu/variables.tf` and the state bucket / lock table from `opentofu/backend.hcl`, so from a configured challenge repo the only input is the account ID. Run **from the repo root**:

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export CHALLENGE_NAME=<your-challenge-name>   # for the aws iam commands below; render.sh would read it from variables.tf anyway
# export AWS_REGION=us-east-1   # optional, this is the default
# GITHUB_ORG / STATE_BUCKET / LOCK_TABLE are auto-read from variables.tf and
# backend.hcl; export any of them to override.

cd "$ROOT"
opentofu/policies/render.sh

aws iam create-role \
  --role-name "${CHALLENGE_NAME}-github-actions-deploy" \
  --assume-role-policy-document file://opentofu/policies/.rendered/github-actions-oidc-trust-policy.json

aws iam put-role-policy \
  --role-name "${CHALLENGE_NAME}-github-actions-deploy" \
  --policy-name "${CHALLENGE_NAME}-github-actions-deploy" \
  --policy-document file://opentofu/policies/.rendered/github-actions-deploy-policy.json

aws iam get-role --role-name "${CHALLENGE_NAME}-github-actions-deploy" --query Role.Arn --output text
```

- The last command returns the role ARN (`arn:aws:iam::<account>:role/<challenge>-github-actions-deploy`) — put it in the `AWS_DEPLOY_ROLE_ARN` GitHub secret.
- ⚠️ If the role already exists and you're only updating permissions, double-check `--role-name` before `put-role-policy` — it's not derived from a variable here, so a stale name silently overwrites a **different** challenge's deploy-role permissions.

## Create a Hugging Face Space

The Space is deployed automatically from GitHub Actions on pushes to `main` (`deploy-hf-space.yml`). Pushing the `hf_testing` branch deploys to the Space named by the `HF_INTERNAL_TESTING_SPACE` Actions variable, if set (a shared testing Space is cheaper than one per challenge, but only one challenge can use it at a time).

To create the Space:

1. Name it to match `challenge_name`. SDK: Docker. Template: blank. Hardware: CPU basic. Visibility: private (for initial testing).
2. Space → Settings → Variables and secrets → New secret. Add:
   - `S3_BUCKET` — from `tofu output -raw s3_bucket_name` (Step 1; equals `challenge_name`)
   - `AWS_DEFAULT_REGION` — usually `us-east-1`
   - `AWS_ACCESS_KEY_ID` — from Step 1
   - `AWS_SECRET_ACCESS_KEY` — from Step 1

## Create a Hugging Face token

Tokens are per account, not per org. HF account menu → Access Tokens → Create new token → `Write` → name it → create. Paste it into the `HF_TOKEN` GitHub secret.

## Create Discord channels and webhook

Two channels per challenge works well: one for discussion, one for submission feedback (kept separate so feedback doesn't drown the discussion). Create a discussion channel (private initially). The submissions channel can be reused across non-concurrent challenges.

1. Edit Channel → Integrations → Webhooks → (pick or create one) → Copy Webhook URL → put it in the `DISCORD_WEBHOOK_URL` GitHub secret.

To disable Discord notifications entirely, set `DISCORD_NOTIFICATIONS=0` in the backend environment.

## Troubleshooting

AWS deploys run automatically via CI, but you can also run OpenTofu locally — see `opentofu/README.md` → "Manual/local operations".

## Teardown

Use this when a challenge has fully concluded. You may only want part of it (e.g. drop the Lambdas but leave the Space and S3). This checklist removes **everything specific to one challenge** — its AWS resources including its own IAM identities (HF Space user, artifacts uploader, Lambda exec role, and the manually created deploy role), its HF Space, and its Discord channels — while leaving the **shared, account-wide bootstrap** (Step 0) untouched.

⚠️ Destructive and largely irreversible. Do step 1 first.

**Do not delete** (shared across every challenge in the account):

- The OpenTofu state bucket / lock table (from Step 0)
- The GitHub OIDC provider (`token.actions.githubusercontent.com`)
- Any shared developer IAM group/policy
- A shared testing HF Space or shared Discord submissions channel — unless no other challenge is using them

### 1. Archive anything worth keeping

```bash
CHALLENGE_NAME=<your-challenge-name>
aws s3 sync "s3://${CHALLENGE_NAME}" "./archive/${CHALLENGE_NAME}" --only-show-errors
```

Also note anything worth keeping from the GitHub repo (issues, discussions) and Discord.

### 2. Destroy the Tofu-managed AWS resources

Removes the S3 bucket, both IAM users and their keys, the Lambda exec role, all Lambda functions, the ECR repo, EventBridge rules, and both Secrets Manager secrets. See `opentofu/README.md` → "Teardown" for the full step-by-step (remove the `prevent_destroy` guard, empty the ECR repo first, then `tofu destroy -var="force_destroy=true"`).

### 3. Delete the GitHub Actions deploy role

Created directly via `aws iam` in Step 3 above — **not** in Tofu state, so `tofu destroy` won't touch it. Double-check `CHALLENGE_NAME`.

```bash
aws iam delete-role-policy \
  --role-name "${CHALLENGE_NAME}-github-actions-deploy" \
  --policy-name "${CHALLENGE_NAME}-github-actions-deploy"

aws iam delete-role --role-name "${CHALLENGE_NAME}-github-actions-deploy"
```

### 4. Delete this challenge's Tofu state file (optional)

The state file lives in the shared state bucket under a per-challenge key, so deleting it doesn't touch other challenges:

```bash
STATE_BUCKET=<your-tofu-state-bucket>
aws s3api list-object-versions \
  --bucket "$STATE_BUCKET" \
  --prefix "${CHALLENGE_NAME}/terraform.tfstate" \
  --output json \
  --query '[Versions[] || `[]`, DeleteMarkers[] || `[]`][]' \
| jq -r '.[] | "\(.Key)\t\(.VersionId)"' \
| while IFS=$'\t' read -r key version; do
    aws s3api delete-object --bucket "$STATE_BUCKET" --key "$key" --version-id "$version"
  done
```

Not required — an orphaned, already-destroyed state file is harmless — but tidy. Never delete the shared state bucket itself.

### 5. Delete the Hugging Face Space

Not managed by Tofu. From an account with owner/admin on the HF org: Space → Settings → "Delete this Space".

### 6. Delete the Discord channels

Delete the challenge-specific discussion channel. Delete the submissions channel too, unless another concurrent challenge still uses it.

### 7. Delete the GitHub repository

Repo Settings → General → Danger Zone → "Delete this repository". Irreversible — confirm anything worth keeping is preserved first.
