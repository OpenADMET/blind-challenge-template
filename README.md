# blind-challenge-template

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Backend Tests](https://github.com/OpenADMET/blind-challenge-template/actions/workflows/backend-tests.yml/badge.svg)](https://github.com/OpenADMET/blind-challenge-template/actions/workflows/backend-tests.yml)
[![HF Space Code Quality](https://github.com/OpenADMET/blind-challenge-template/actions/workflows/hf-space-checks.yml/badge.svg)](https://github.com/OpenADMET/blind-challenge-template/actions/workflows/hf-space-checks.yml)

Template backend and infrastructure for a **blind challenge** — copy this repo to stand up a new ML competition in minutes. Supports three independent submission tracks out of the box: regression, classification, and structure (protein–ligand pose) prediction, each scored and leaderboarded separately. Built and used by [OpenADMET](https://openadmet.org), open-sourced for anyone to reuse.

## Background

Read the announcement post for the rationale behind the architecture and how the scoring pipeline works: **[Open-sourcing our blind challenge platform](https://openadmet.ghost.io/open-sourcing-our-blind-challenge-platform/)**.

## Getting started

See [`docs/SETUP.md`](docs/SETUP.md) for the step-by-step checklist to stand up a new challenge — it's org-agnostic, so bring your own account-specific values (state bucket names, IAM group, Discord server).

---

## Customising this template for a new challenge

Everything a new challenge needs to fill in is marked with a `TODO` comment: endpoint names, dataset sizes, and deadlines in [`backend/config.py`](backend/config.py); endpoint names, external links, and the page-content markdown in [`hf_space/config.py`](hf_space/config.py); example counts/names in `docs/`; and the Space title in [`hf_space/README.md`](hf_space/README.md). The two `config.py` files are the source of truth — `hf_space/config.py` must mirror `backend/config.py`'s endpoint lists and dataset sizes.

The quickest way to see every fill-in point at once is the [**Todo Tree**](https://marketplace.visualstudio.com/items?itemName=Gruntfuggly.todo-tree) VS Code extension, which lists all `TODO` markers across the repo in a single tree view (markdown `TODO`s use `<!-- TODO: ... -->` comments so they don't render on the Space but are still picked up). Outside VS Code, `grep -rn "TODO" backend/config.py hf_space/config.py` or `grep -rn "TODO:" --include='*.py' --include='*.md' --include='*.yml' .` gives the same list. Work through them all, then run `pytest tests/backend/`. The test fixtures under `tests/backend/test_data/` follow whatever endpoint names and sizes `backend/config.py` declares — regenerate them after any such change with `python tests/backend/generate_test_data.py`. That data is fully synthetic (fixed-seed random values, textbook SMILES, stub PDBs); never commit real challenge data there.

**Community-health files** — `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `CHANGELOG.md`, and the `.github/` issue/PR templates describe *this template project*. When you copy the repo for a real challenge, delete or replace the ones that don't apply (see `docs/SETUP.md` step 12), and point `SECURITY.md` at your own disclosure channel. `LICENSE` stays (Apache-2.0).

---

## Repository Structure

| Path | Contents |
|------|----------|
| [`backend/`](backend/README.md) | Submission validation, bootstrapped scoring, leaderboard generation, and the AWS Lambda handlers |
| [`hf_space/`](hf_space/README.md) | The Gradio app (submission form, leaderboards) deployed to the HF Space |
| [`opentofu/`](opentofu/README.md) | OpenTofu config provisioning all AWS resources (S3, Lambda, EventBridge, IAM, ECR, Secrets Manager) |
| [`docs/`](docs/scoring-and-leaderboards.md) | Design docs — scoring/leaderboard pipeline diagrams, phase/stage disambiguation, etc. |
| `tests/backend/` | Backend unit tests (pytest) |
| `.github/workflows/` | CI: `backend-tests.yml` (lint/type-check/docstring-check/pytest for `backend/`) and `hf-space-checks.yml` (the same checks for `hf_space/`) run here; the deploy-on-push workflows for the backend/infra and the HF Space live in `.github/workflows.disabled/` until you stand up a challenge |

---

## Documentation

- [backend/README.md](backend/README.md) — Submission validation, scoring, manifest/leaderboard generation, and Lambda handler docs
- [opentofu/README.md](opentofu/README.md) — AWS infrastructure: resources created, IAM policy grants, setup walkthrough for a new challenge deployment, teardown
- [docs/scoring-and-leaderboards.md](docs/scoring-and-leaderboards.md) — How a regression or classification submission turns into bootstrapped per-endpoint metrics, a macro-averaged ("MA") score, and each track's per-endpoint + overall leaderboards
- [docs/structure-scoring.md](docs/structure-scoring.md) — How a structure submission's PDB files turn into per-compound pose-quality scores (LDDT-PLI/BiSyRMSD/LDDT-LP) and the structure leaderboard
- [docs/phases-and-stages.md](docs/phases-and-stages.md) — Disambiguates the three unrelated "phase"/"stage" concepts in this codebase (compound-set phase, leaderboard stage, HF Space `CURRENT_PHASE`)

---

## Local Development

```bash
# Create the conda environment (backend deps + dev tools)
conda env create -f environment.yml
conda activate blind-challenge-template

# Run backend tests
pytest tests/backend/

# Validate a submission file locally
python -m backend.cli --regression-predictions path/to/predictions.parquet
```

See [backend/README.md](backend/README.md) for validation rules, scoring behavior, and the full CLI reference.

---

## Deployment Flow (GitHub Actions → AWS)

```
Push to main
  │
  ├─ backend/** or opentofu/** changed?
  │    │
  │    ├─ backend/** changed?
  │    │    ├── Build Lambda image (docker buildx)
  │    │    └── Push to ECR: sha-{commit} + latest  [uses AWS_DEPLOY_ROLE_ARN OIDC]
  │    │
  │    ├─ tofu init
  │    ├─ tofu plan   [passes lambda_image_digest -var when backend changed]
  │    └─ tofu apply  [uses AWS_DEPLOY_ROLE_ARN OIDC]
  │         └── Create/update: S3, Lambda, EventBridge, IAM, ECR, Secrets Manager
  │
  ├─ Post-deploy (if tofu apply succeeded):
  │    ├── Sync DISCORD_WEBHOOK_URL ─► AWS Secrets Manager
  │    └── Sync HF_TOKEN ─► AWS Secrets Manager
  │
  └─ Deploy hf_space/ to production HF Space  [uses HF_TOKEN]

Push to hf_testing
  │
  └─ Deploy hf_space/ to the shared testing Space in the HF_INTERNAL_TESTING_SPACE
     repo variable  [uses HF_TOKEN; job skipped if the variable is unset]
```

Two workflows are active in the template itself, both on every push and PR to `main` (and `hf_testing`), plus manual dispatch: `.github/workflows/backend-tests.yml` (ruff/mypy/pydoclint, then pytest) on changes to `backend/`, `tests/`, `environment.yml`, `pytest.ini`, or `pyproject.toml`; and `.github/workflows/hf-space-checks.yml` (the same ruff/mypy/pydoclint checks, no test suite) on changes to `hf_space/` or `pyproject.toml`.

### Required GitHub Actions Secrets (`.github/workflows/deploy.yml`, `deploy-hf-space.yml`)

| Secret | Purpose | Scope |
|--------|---------|-------|
| `AWS_DEPLOY_ROLE_ARN` | OIDC role for ECR push + OpenTofu apply | AWS Lambda + ECR |
| `HF_TOKEN` | Hugging Face API token (write access) | HF Space deploy; synced to AWS Secrets Manager after `tofu apply` for the HF-restarter Lambda |
| `DISCORD_WEBHOOK_URL` | Discord notification webhook URL | Synced to AWS Secrets Manager after `tofu apply` |

### GitHub Actions Variables — all optional

Non-secret per-challenge values (the state bucket + lock table, the challenge name, the HF owner) live in two committed files, `opentofu/backend.hcl` and `opentofu/variables.tf`, and CI reads them straight from there. The only Actions Variables are optional overrides:

| Variable | Purpose | Default if unset |
|----------|---------|------------------|
| `AWS_REGION` | Deploy region | `us-east-1` |
| `HF_INTERNAL_TESTING_SPACE` | `owner/space` of a shared testing Space for the `hf_testing` branch | that job is skipped |
| `NAME_MUST_CONTAIN` | Substring `challenge_name` must contain (naming guardrail) | check skipped |

See [docs/SETUP.md](docs/SETUP.md) for the full walkthrough, and [opentofu/README.md — Where secrets live](opentofu/README.md#where-secrets-live) for the secrets/variables map.

---

## HF Space Deployment

`hf_space/` (the full app) is deployed automatically by `.github/workflows/deploy-hf-space.yml`, based on which branch was pushed — each job splits `hf_space/` into its own subtree and force-pushes it to the Space's `main` branch:

| Branch | Destination |
|--------|-------------|
| `main` | Production Space — `{HF_OWNER}/{challenge_name}` |
| `hf_testing` | The shared testing Space in the `HF_INTERNAL_TESTING_SPACE` repo variable (job skipped if unset) |

To stage changes before they go live, push or merge to `hf_testing` first, verify on the testing Space, then merge to `main` to promote to production.

Binary assets under `hf_space/` are tracked with Git LFS; the deploy workflow installs [Git Xet](https://github.com/huggingface/xet-core) purely as an LFS transfer agent, so the push to the Space moves that content through Hugging Face's faster Xet backend instead of plain LFS HTTP transfer.

### HF Space Configuration

The Space reads from the same S3 bucket as the backend. Set these secrets on the HF Space (and internal testing space):

| Secret | Value | Source |
|--------|-------|--------|
| `AWS_ACCESS_KEY_ID` | HF Space IAM user access key | `tofu output -raw hf_space_access_key_id` (from `opentofu/`) |
| `AWS_SECRET_ACCESS_KEY` | HF Space IAM user secret | `tofu output -raw hf_space_secret_access_key` (from `opentofu/`) |
| `S3_BUCKET` | S3 bucket name | `tofu output -raw s3_bucket_name` (from `opentofu/`) |
| `AWS_DEFAULT_REGION` | AWS region | `us-east-1` (default) |

---

## Contributing

Contributions to **the template itself** are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md), which covers the dev setup, the checks to run, the PR process, and the policy on AI-assisted contributions. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). To report a security issue, see [`SECURITY.md`](SECURITY.md).

Contributions here are for the shared template. Once you've copied it into a challenge repo, that repo is yours to run however you like.

## License

Licensed under the Apache License, Version 2.0 — see [`LICENSE`](LICENSE). Copyright © Open Molecular Software Foundation and contributors.

Unless you state otherwise, any contribution you submit for inclusion in this repository is licensed under Apache-2.0, and you certify the [Developer Certificate of Origin](https://developercertificate.org/) (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Citing

If this template supports work you publish, please cite it — see [`CITATION.cff`](CITATION.cff) or GitHub's "Cite this repository" button.

## Acknowledgements

We would like to thank our funders for their support of OpenADMET, in particular ARPAH, Radial (part of the Astera Institute (https://ror.org/00ydx1s47)), Schrödinger Inc, and the Gates Foundation.  We would also like to thank our partners Enamine, HuggingFace, OpenEye, CDD Vault, Discovery Life Sciences, and the beamline staff at NSLS-II for their support. 

This work is supported by the Advanced Research Projects Agency for Health (ARPA-H) under AVOID-OME, and Award Number 1AY1AX000035. The contents are those of the authors. They may not reflect the policies of the Department of Health and Human Services or the U.S. government. The content is solely the responsibility of the authors and does not necessarily represent the official views of the Advanced Research Projects Agency for Health.
