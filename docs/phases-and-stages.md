# Phases & Stages — Disambiguation

The word "phase" (and, separately, "stage") is used for **three unrelated concepts**
across this codebase, and they get confused easily because they all use small integers
or the words live/interim/final. This doc is a single place to look them up.

| Concept | Values | Lives in | Controls |
|---|---|---|---|
| Compound-set **phase** | `0`, `1` (`2` exists as a data tag, but is never scored on its own) | `backend/` — `TrackPaths.scores_paths`, `{track}-identifiers.parquet`'s `"phase"` column | Which compounds a submission's scores are computed against |
| Leaderboard **stage** | `"live"`, `"interim"`, `"final"` | `backend/aws_leaderboards.py` — `create_track_leaderboards(stage=...)` | Which S3 prefix a leaderboard is read/written under, the submission cutoff, and whether significance testing / invalid-submission filtering run |
| Challenge-timeline **`CURRENT_PHASE`** | `0`–`4` | `hf_space/config.py` | Which tabs/content are visible on the HF Space — set **manually**, not derived from dates |

## 1. Compound-set phase (backend scoring)

Every submission is scored **twice**, immediately on upload, against two different
compound subsets — see the `for phase in [0, 1]` loop in
`process_new_regression_submission` / `process_new_classification_submission` /
`process_new_structure_submission` (`backend/aws_submission_processing.py`):

| Phase | Meaning | Ground truth used | Written to |
|---|---|---|---|
| `0` | "all" — every compound in the test set | Full ground truth, unfiltered | `scores/all/{track}/...` |
| `1` | The subset of compounds unblinded early (during the challenge's phase 1 window) | Ground truth filtered to `identifiers["phase"] == 1` | `scores/phase_1/{track}/...` |

`TrackPaths.scores_paths` (`backend/config.py`) is the `{0: scores_all, 1: scores_phase_1}`
lookup used by this loop.

**A `"phase" == 2` tag also exists** in `{track}-identifiers.parquet` — compounds held
blinded until the very end of the challenge. It is **not** part of this scoring loop
(there is no `scores/phase_2/`); phase-2 compounds are simply included within phase 0
("all") once the interim/final leaderboards score against the complete set. Nothing in
the codebase reads the `"phase" == 2` tag directly — it's a data-side marker only.

`TrackPaths.phase_2_entries` (`backend/config.py`) — an S3 prefix property
(`submissions/manifest/phase_2_entries/{track}`) — is defined but **not currently
referenced anywhere else in the codebase**. It isn't wired into any Lambda or script
today.

## 2. Leaderboard stage

Covered in full in [scoring-and-leaderboards.md](scoring-and-leaderboards.md); summary:

| Stage | Scored from (phase) | Submission cutoff | Significance testing | Invalid-submission filtering |
|---|---|---|---|---|
| `live` | `1` (`scores/phase_1`) | none — always every submission so far | never | never |
| `interim` | `0` (`scores/all`) | `INTERIM_LEADERBOARD_DEADLINE` | master leaderboard only | no |
| `final` | `0` (`scores/all`) | `FINAL_LEADERBOARD_DEADLINE` | master leaderboard only | yes |

`live` is auto-generated on a schedule by the leaderboard Lambda; `interim`/`final` are
generated manually by calling `create_track_leaderboards(stage="interim" | "final")`.

## 3. Challenge-timeline `CURRENT_PHASE` (HF Space UI)

`hf_space/config.py`:

```python
CURRENT_PHASE = 0
```

The comment above it is explicit: **"This is updated manually to allow control over
what is visible on the HF Space."** Nothing in the codebase bumps this automatically
when `INTERIM_LEADERBOARD_DEADLINE` / `FINAL_LEADERBOARD_DEADLINE` pass — a human has
to edit this constant (and redeploy) at each transition. If it's left stale, the Space
will keep showing (or hiding) tabs that no longer match reality.

| `CURRENT_PHASE` | Meaning | Submit tab | Live Leaderboard | Interim Leaderboard | Final Leaderboard |
|---|---|---|---|---|---|
| `0` | Pre-challenge | "Coming Soon" placeholder | shown (placeholder) | hidden | hidden |
| `1` | Phase 1 — only live leaderboard | open | shown | hidden | hidden |
| `2` | Phase 2 — live + static interim leaderboard | open | shown | shown | hidden |
| `3` | Challenge closed, final leaderboard not yet released | "Closed" placeholder | shown | shown | shown (placeholder) |
| `4` | Challenge closed, final leaderboard released | "Closed" placeholder | shown | shown | shown |

Gating logic: `hf_space/app.py` (tab visibility) and `hf_space/submission.py`'s
`render_submission_tab(phase)` (submission portal copy). `CURRENT_PHASE == 2` is meant
to roughly line up with `INTERIM_LEADERBOARD_DEADLINE` passing, and `3`/`4` with
`FINAL_LEADERBOARD_DEADLINE` passing — but that alignment is a human's responsibility,
not code.

## Where this lives in the code

| Concept | File |
|---|---|
| Compound-set phase, `TrackPaths.scores_paths`, `phase_2_entries` | `backend/config.py` |
| Phase-based scoring loop | `backend/aws_submission_processing.py` |
| Leaderboard stage, cutoffs, significance/filtering rules | `backend/aws_leaderboards.py`, `backend/config.py` |
| `CURRENT_PHASE` and tab gating | `hf_space/config.py`, `hf_space/app.py`, `hf_space/submission.py` |
