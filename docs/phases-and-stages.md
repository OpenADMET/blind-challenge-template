# Compound Sets, Stages and the Challenge Timeline

Three separate ideas decide what a leaderboard shows. Only the third is called a
"phase".

| Concept | Values | Lives in | Controls |
|---|---|---|---|
| **Compound set** | `"live"`, `"hidden"`, `"all"` | `{dataset}-identifiers.parquet`'s `set` column; `TrackSpec.leaderboard_sets` (both `config.py` files) | Which compounds a submission is scored on |
| **Leaderboard stage** | `"live"`, `"interim"`, `"final"` | `backend/aws_leaderboards.py` (`create_track_leaderboards(stage=...)`) | Submission cutoff, significance testing, invalid-submission filtering, and (through `leaderboard_sets`) which compound set is ranked |
| **Challenge phase** | `CURRENT_PHASE`, `0` to `4` | `hf_space/config.py` | Which tabs the Space shows, set by hand |

## 1. Compound sets

Each compound in a dataset's identifiers file (`activity-identifiers.parquet`,
`structure-identifiers.parquet`) is in the `live` or the `hidden` set (`set`
column). `all` is both together.

Each track's `TrackSpec.leaderboard_sets` says which set each leaderboard stage
ranks on. The default is:

| Stage | Set |
|---|---|
| `live` | `live` |
| `interim` | `all` |
| `final` | `all` |

A track can rank differently, e.g. classification with `{"live": "live",
"interim": "hidden", "final": "hidden"}`, so its final leaderboard only uses
compounds no one has seen scores for.

When a submission arrives it is scored on every set its track's stages use
(`TrackSpec.scored_sets`), and each set's scores are saved under
`scores/{set}/{track}/{user}/{submission_id}/`. The manifest row records one
`{set}_scores_uri` per set, and the leaderboard builder picks the column for the
stage's set. Structure scores each compound once and re-aggregates per set; the
activity tracks recompute their metrics per set, since R², Spearman and Kendall are
whole-sample statistics.

## 2. Leaderboard stage

Covered in full in [scoring-and-leaderboards.md](scoring-and-leaderboards.md); summary:

| Stage | Submission cutoff | Significance testing | Invalid-submission filtering |
|---|---|---|---|
| `live` | none: every submission so far | never | never |
| `interim` | `INTERIM_LEADERBOARD_DEADLINE` | master leaderboard only | no |
| `final` | `FINAL_LEADERBOARD_DEADLINE` | master leaderboard only | yes |

`live` is generated on a schedule by the leaderboard Lambda; `interim`/`final` are
generated manually with `create_track_leaderboards(stage="interim" | "final")`.

## 3. Challenge timeline: `CURRENT_PHASE` (HF Space)

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
| Compound sets, `TrackSpec.leaderboard_sets`, `TrackPaths.scores` | `backend/config.py` (mirrored in `hf_space/config.py`) |
| Scoring each set, `compound_set_ids` | `backend/aws_submission_processing.py` |
| Leaderboard stage, cutoffs, significance/filtering rules | `backend/aws_leaderboards.py`, `backend/config.py` |
| `CURRENT_PHASE` and tab gating | `hf_space/config.py`, `hf_space/app.py`, `hf_space/submission.py` |
