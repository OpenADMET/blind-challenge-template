# Changelog

All notable changes to this template are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); tagged releases use
[Semantic Versioning](https://semver.org/).

A challenge repo copies the template at a point in time and doesn't track it
afterwards. This log is so you can see what changed if you copy it again for a
later challenge.

## [0.1.1] - 2026-11-17

Security and bug-fix release. If you are running a challenge built from 0.1.0,
see "Action needed" below.

### Security

- **Leaderboard HTML injection.** Anonymous entrants' aliases and model report
  links were rendered as raw HTML on the HF Space leaderboards, so a crafted
  alias or link could run JavaScript in every visitor's browser. Names, aliases
  and links are now escaped. At submission, aliases are limited to 40 characters
  and can't contain `< > & " '` or backticks or control characters, and report
  links can't contain characters that aren't valid in a URL; the participant is
  told which rule their entry broke.
- **HF Space IAM user could read all submissions.** The Space's IAM user had
  `s3:GetObject` on `submissions/*`, which it never uses, so a leaked Space key
  exposed every participant's predictions. It can now read `leaderboard/*`
  only, with an explicit deny on `ground_truth/*`.
- **Structure submissions with several ligand poses.** A PDB file with more than
  one `LIG` residue was scored as the best of its poses. Validation now requires
  exactly one `LIG` residue in every file and rejects the whole submission
  otherwise, listing each offending file; the scorer also refuses such a file.
  A reference structure may still contain several ligand copies.

### Fixed

- Structure leaderboards failed to build ("Column 'coverage_mean' not found")
  because scores were labelled `Structure` while `STRUCTURE_ENDPOINTS` is
  `["structure"]`. Scores now use the configured label, and leaderboards match
  endpoints case-insensitively so existing score files still work (#25).
- A submission missing a leaderboard column (e.g. scored before the column was
  added) no longer stops the leaderboard for everyone: it gets NaN for that
  column. The HF Space now orders structure leaderboards by the backend's rank,
  so a leaderboard CSV without a given metric column still loads (#26).

### Added

- `tests/hf_space/`, the first tests for the HF Space app, run in CI by
  `hf-space-checks.yml`.

### Action needed for running challenges

- Run `tofu apply` to update the HF Space IAM policy.
- Redeploy the scoring Lambdas and the HF Space to pick up the fixes.
- Before computing final structure leaderboards, check already-accepted
  structure submissions for files with more than one `LIG` residue; they were
  scored as the best of their poses.
- Participants on the structure track should be told that a file without
  exactly one `LIG` residue now fails validation.

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

[0.1.1]: https://github.com/OpenADMET/blind-challenge-template/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/OpenADMET/blind-challenge-template/releases/tag/v0.1.0
