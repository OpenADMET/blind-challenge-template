# Changelog

All notable changes to this template are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); tagged releases use
[Semantic Versioning](https://semver.org/).

A challenge repo copies the template at a point in time and doesn't track it
afterwards. This log is so you can see what changed if you copy it again for a
later challenge.

## [0.2.0] - 2026-11-18

A large release for **new challenges only**: the S3 layout, identifiers files,
manifest columns and submission flow have changed, and there is no migration path
for a challenge already running on 0.1.x.

### Breaking changes

- **Compound sets replace phases.** Identifiers files have a `set` column (`live` /
  `hidden`) instead of `phase`. Each track's `TrackSpec.leaderboard_sets` maps every
  leaderboard stage to `live`, `hidden` or `all`. Scores are stored under
  `scores/{set}/{track}/`, and manifests have one `{set}_scores_uri` column per set
  (#16).
- **Manifests moved** from `submissions/manifest/` to `manifests/{track}/`, outside
  the delete-protected prefix, so entries can be removed without lifting it (#12).
- **Submissions need a Hugging Face login** (`hf_oauth: true`). The submitter is the
  logged-in account, and the typed username field is gone (#23).
- **`BOOTSTRAP_SAMPLES` is 10,000** (was 1,000), so scores from 0.1.x can't be
  compared with new ones (#11).
- Submission IDs are UUIDv7, so they sort by submission time; existing UUIDv4 IDs
  are still accepted (#13).

### Security

- **Login and account checks.** Accounts younger than `MIN_ACCOUNT_AGE_DAYS`
  (default 1) can't submit, and on a Space the account's ID must match the login.
  The account check, the submission cooldown and the new ban list all **fail
  closed**: if they can't be checked, the submission is refused (#23, #22).
- **Ban list** in the bucket (`admin/banned_users.csv`), read by the Space every 5
  minutes, matched on username or account ID, and edited with
  `scripts/manage_bans.py`. The Space can read it but not write it (#22).
- The live leaderboard rounds scores to 2 significant figures in everything the
  browser receives, including the CSV download, so full-precision live scores
  aren't exposed during the challenge.

### Added

- **Per-track config** (#16): each track is defined once per side in a `TrackSpec`
  (endpoints, metrics, ranking, dataset size, extra columns, compound sets), with a
  drift test checking `backend/` and `hf_space/` agree. `TrackVisibility` sets
  where each track appears on the Space. `REQUIRE_MODEL_REPORT` controls whether
  the final leaderboard requires a model report link.
- **Organiser scripts** (`scripts/`, #18): interim and final leaderboard notebooks
  that build, check (failed builds, cutoff, ranks, metric ranges, movement from
  live) and upload behind a guard, refusing partial uploads;
  `quarantine_submissions.py` to take entries off the leaderboard reversibly; and
  `manage_bans.py`. Guides: `docs/bans.md`, `docs/removing-entries.md` (#17).
- **Head-to-Head tab** (phase 4): compares two final-leaderboard entrants on every
  test compound. After the final deadline, the final notebook releases the files it
  reads; entrants are identified by rank, and the Space still can't read ground
  truth, submissions or scores.
- **Space refresh:** theme colours and font in `config.py` (neutral defaults); a
  status line and a final-week model report reminder that follow the date without
  a restart; announcement, dataset and Discord link buttons; consistent headings;
  a "Built with the OpenADMET blind challenge template" credit line
  (`SHOW_TEMPLATE_CREDIT`).
- **Leaderboard table:** scores show as "mean ± std" but sort numerically by the
  mean; 2 significant figures on live, 3 on interim and final; ↑/↓ headers and a
  bold primary metric; significance tiers shown and shaded on interim and final;
  an endpoint picker; name search, filters and a column picker.
- Ligand bonds for structure scoring come from the ground-truth SMILES (optional
  `SMILES` column in the structure identifiers), fixing unscoreable strained rings
  and false PoseBusters failures (#27).
- A test suite for the HF Space (`tests/hf_space/`), and tests for `scripts/`, run
  in CI (#7).
- RDKit and PoseBusters in `environment.yml`, so the ligand scoring tests run in CI.
- `access-analyzer:ValidatePolicy` in the developer policy template, for editing
  the bucket policy in the console (#19).

### Changed

- The Space runs on **Gradio 6** (pandas 3, pydantic 2.14), with its own leaderboard
  table in place of `gradio-leaderboard`. The CSV download now works in one click.
- Bootstrap p-values are `min(1, 2 (r + 1) / (B + 1))`, with ties counted for both
  sides, so they are never exactly 0 (#11).
- Structures are scored once per submission and re-aggregated for each compound set
  (#15).
- A failed upload is reported to the participant instead of being shown as a
  success.
- The final leaderboard no longer re-checks usernames (they come from the login).
- Both `config.py` files are reorganised into sections, and long docstrings and
  history notes are trimmed.

### Removed

- `gradio-leaderboard`, the typed username field, `validate_hf_username`, the
  hard-coded `BANNED_USERNAMES`, `utils.current_phase`,
  `TrackPaths.phase_2_entries`, `dummy_score_single_structure`,
  `score_structure_submission`, and case-insensitive endpoint matching (only needed
  for score files from before 0.1.1).

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

[0.2.0]: https://github.com/OpenADMET/blind-challenge-template/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/OpenADMET/blind-challenge-template/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/OpenADMET/blind-challenge-template/releases/tag/v0.1.0
