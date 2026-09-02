# Changelog

All notable changes to this template are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); tagged releases use
[Semantic Versioning](https://semver.org/).

A challenge repo copies the template at a point in time and doesn't track it
afterwards. This log is so you can see what changed if you copy it again for a
later challenge.

## [0.1.0] - 2026-09-02

Initial public release.

### Added

- Bootstrapped scoring backend for three independent submission tracks —
  regression, classification, and protein-ligand structure — with per-endpoint
  and macro-averaged leaderboards and pairwise significance testing.
- OpenTofu configuration for the full AWS backend (S3, Lambda via ECR,
  EventBridge, IAM, Secrets Manager), driven by a single `challenge_name`, with a
  per-challenge OIDC-scoped GitHub Actions deploy role.
- Hugging Face Spaces Gradio app: submission form, leaderboards, and per-track
  on/off toggles.
- `docs/SETUP.md` end-to-end walkthrough; deploy-role and shared-developer-group
  IAM policy templates rendered by `opentofu/policies/render.sh`.
- Fully synthetic backend test fixtures, regenerated from config by
  `tests/backend/generate_test_data.py`.
- CI: `backend-tests.yml` runs on push/PR; Dependabot for Actions and pip;
  GitHub Actions pinned to commit SHAs.

[0.1.0]: https://github.com/OpenADMET/blind-challenge-template/releases/tag/v0.1.0
