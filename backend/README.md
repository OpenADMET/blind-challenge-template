# Backend

Backend services for a blind challenge.

This package currently provides:

- Submission validation (regression, classification, and structure formats)
- Regression and classification prediction scoring (bootstrapped metrics)
- S3 helpers for reading submissions and writing scores/leaderboards
- Lambda handlers for regression, classification, and structure evaluation,
  leaderboard generation, the full cross-track entries list, and HF Space restart

## Container Image Strategy

All image-based backend Lambdas share one container image:

- Dockerfile: `backend/Dockerfile.lambda`
- Built/pushed via `docker build`/`docker push` — see `opentofu/README.md`'s "Manual/local operations" for the exact commands, or let CI (`deploy.yml`) handle it on push to `main`
- Runtime handler selected per Lambda by `image_config.command` in `opentofu/lambda.tf`

Configured handlers:

- Regression evaluator: `backend.lambda_handler_regression.handler`
- Classification evaluator: `backend.lambda_handler_classification.handler`
- Structure evaluator: `backend.lambda_handler_structure.handler`
- Leaderboard generator: `backend.lambda_handler_leaderboard.handler`
- Entries list generator: `backend.lambda_handler_entries.handler`
- HF Space restarter: `backend.lambda_handler_hf_restart.handler`

This keeps dependencies and code identical across evaluators while preserving different entry points.


## Modules

- `config.py`: Challenge constants, dataset sizes, metrics, and **dataclass-based S3 path helpers**. Defines:
  - `TrackPaths`: Dataclass for S3 path management per track, with properties for all S3 prefixes (submissions, manifest, scores, leaderboards). `REGRESSION_PATHS`, `CLASSIFICATION_PATHS`, and `STRUCTURE_PATHS` are three fully independent tracks, each with its own submission-upload prefix, Lambda, and scoring/leaderboard pipeline. `ACTIVITY_PATHS` is no longer an upload track — it's kept only as the shared ground-truth/identifiers source (`activity-dataset.parquet`, `activity-identifiers.parquet`) that both regression and classification score against, since they cover the same underlying compound set. `CLASSIFICATION_ENDPOINTS` may be empty (e.g. before that track launches) — every place that consumes it degrades to a no-op rather than erroring.
  - `SubmissionKey`: Dataclass for parsing and validating S3 submission keys, with helpers for key/prefix validation and track/file matching.
  - All constants (dataset sizes, metrics, deadlines, etc.) and leaderboard sort keys (now strings, not lists).
- `leaderboard.py`: Modular leaderboard system using dataclasses:
  - `EntryMetric`, `LeaderboardEntry`, `EntryComparison`, `FinalLeaderboard`.
  - Supports pairwise statistical comparisons and Compact Letter Display (CLD) for significance grouping.
  - Generates leaderboard DataFrames sorted by primary metric and submission time, with optional CLD columns.
- `aws_submission_processing.py`: Validates and scores incoming submissions, saving score files to S3.
- `aws_manifest.py`: Aggregates validated submission metadata into a per-track manifest.
- `aws_leaderboards.py`: Builds and saves ranked leaderboards from manifest + score data, batch-loaded via DuckDB. `create_track_leaderboards` builds one leaderboard per endpoint, plus a macro-ranked master (`MACRO_ENDPOINT_LABEL`, "MA") when a track has more than one endpoint; significance testing and invalid-submission filtering apply only to that master, and only for "interim"/"final" stages. A track with zero endpoints (e.g. `CLASSIFICATION_ENDPOINTS = []`) returns `{}` immediately rather than building anything.
- `submission_validation.py`: Pandera validation for regression/classification files and zip checks for structure files.
- `evaluate_predictions.py`: Bootstrap scoring + aggregation helpers.
- `utils.py`: Phase determination + utility transforms/sampling.
- `discord_bot.py`: Posts submission results to Discord via webhook.
- `cli.py`: Local validation CLI (`python -m backend.cli ...`).
- `lambda_handler_regression.py`: S3-triggered handler for regression prediction evaluation.
- `lambda_handler_classification.py`: S3-triggered handler for classification prediction evaluation.
- `lambda_handler_structure.py`: S3-triggered handler for structure prediction evaluation.
- `lambda_handler_leaderboard.py`: Scheduled handler for leaderboard generation.
- `lambda_handler_entries.py`: Scheduled (daily) handler for the full cross-track entries list.
- `lambda_handler_hf_restart.py`: Scheduled handler to restart HuggingFace Space via `huggingface_hub`.

## Validation Behavior

### Regression submissions

Input: a dataframe loaded from `.parquet` or `.csv` with:

- Identifier columns: `SMILES`, `Molecule_Name`
- Regression endpoint columns (from `config.REGRESSION_ENDPOINTS`): `ENDPOINT_1`, `ENDPOINT_2`, `ENDPOINT_3`, `ENDPOINT_4` (template default — set to this challenge's real names)
- Exact row count: `ACTIVITY_DATASET_SIZE` — regression and classification submissions cover the same underlying compound set, so both share this constant

Validation checks:

- all identifier + endpoint columns must be present (non-strict schema — extra columns beyond these are allowed and ignored)
- identifier columns are non-null strings
- endpoint columns are non-null floats, and finite (`-inf`/`inf` rejected) — participants must predict every compound; `evaluate_predictions.score_activity_predictions` also raises if a compound with ground truth ends up with a missing prediction, as a backstop

Entry point: `validate_regression_submission()`.

### Classification submissions

Input: a dataframe loaded from `.parquet` or `.csv` with:

- Identifier columns: `SMILES`, `Molecule_Name`
- Classification endpoint columns (from `config.CLASSIFICATION_ENDPOINTS`): `ENDPOINT_5`, `ENDPOINT_6` (template default) — may be an empty list (e.g. before that track launches)
- Exact row count: `ACTIVITY_DATASET_SIZE`

Validation checks:

- all identifier + endpoint columns must be present (non-strict schema — extra columns beyond these are allowed and ignored)
- identifier columns are non-null strings
- endpoint columns are non-null and boolean-valued (`True`/`False` or `1`/`0`) — checked via `isin`, not dtype coercion, so a non-binary value like `0.37` is rejected rather than silently truthy-coerced

Entry point: `validate_classification_submission()`.

### Structure submissions

Input: `.zip` file.

Validation checks:

- filename suffix is `.zip`
- archive contains exactly `STRUCTURE_DATASET_SIZE` `.pdb` files (or, when `expected_ids` is supplied, exactly that set of IDs — no more, no less)

Entry point: `validate_structure_submission()`.

Note: the ligand residue in each PDB must be named `LIG` (used by OST — `evaluate_predictions.score_single_structure` — to identify the small molecule), but this is **not** checked at submission time; a wrong residue name only surfaces later as a `NaN` structure score, not a validation error. Deeper structure validation (ligand residue naming, ligand/protein chain correctness, alignment, binding site geometry, etc.) is not implemented yet. See [docs/structure-scoring.md](../docs/structure-scoring.md) for the full structure scoring pipeline.

## Scoring Behavior (Regression / Classification)

Implemented in `evaluate_predictions.py` and used by `aws_submission_processing.py`.
See [docs/scoring-and-leaderboards.md](../docs/scoring-and-leaderboards.md) for the
full diagram — this is a summary.

Regression and classification are **two independent submission tracks**, each with
its own upload, validation, Lambda, and metric set:

- Regression (from `config.ACTIVITY_METRICS`): `ST-RAE`, `MAE`, `R2`, `Spearman_R`, `Kendall_Tau`
- Classification (from `config.CLASSIFICATION_METRICS`): `MCC`, `Accuracy`, `Precision`, `Recall`, `F1`

Scoring flow (identical shape for both tracks, differing only in `endpoints`/`metrics`):

1. Merge predictions with ground truth on `Molecule_Name`.
2. For each of the track's endpoints, in one call to `score_activity_predictions(predictions, ground_truth, endpoints=...)`:
   - Exclude compounds with no ground-truth value for that endpoint (`y_true` is `NaN` — a compound not tested for that endpoint isn't scored on it; the exclusion mask uses `pd.isna`, not `np.isnan`, since a classification ground-truth column with any missing label upcasts to `object` dtype after the merge). A compound that *does* have a ground-truth value but a missing prediction raises `ValueError` — participants are expected to predict every compound; `submission_validation.py` is the primary check for this, `score_activity_predictions` is a backstop.
   - Dispatch to that endpoint's metric list (`_metrics_for_endpoint`) — regression metrics for a `REGRESSION_ENDPOINTS` entry, classification metrics for a `CLASSIFICATION_ENDPOINTS` entry.
   - For a `REGRESSION_ENDPOINTS` entry, also pull that endpoint's per-compound credible-interval bounds from ground truth (`f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_UPPER_SUFFIX}"` / `f"{endpoint}{REGRESSION_CREDIBLE_INTERVALS_LOWER_SUFFIX}"`, i.e. `..._conf_high`/`..._conf_low`) and thread them through to `bootstrap_metrics`.
   - Run bootstrap sampling (`BOOTSTRAP_SAMPLES = 1000`, indices shared across endpoints with the same eligible-compound pool size) and compute that endpoint's metrics per bootstrap sample. `bootstrap_metrics` inspects each metric function's signature (`_metric_needs_credible_interval_bounds`) and only passes the credible-interval bounds to metrics that accept them — currently just `ST-RAE` (`rae_soft_threshold_absolute_error`), which clips a prediction's error to the distance to the nearest credible-interval bound rather than the raw distance to the point estimate; predictions falling entirely inside the interval score zero error. A metric that fails to compute (including `ST-RAE` when bounds are missing), or returns a non-finite value, raises `RuntimeError` — it is **not** silently scored as `0` (for error metrics like MAE/ST-RAE, `0` means "perfect," so zero-filling a real failure would misrepresent it as a flawless score).
3. `add_macro_endpoint(...)` computes a macro-averaged score per bootstrap sample across the track's endpoints, stored as a synthetic `MACRO_ENDPOINT_LABEL` ("MA") pseudo-endpoint. Every metric is a plain arithmetic mean across endpoints. (`Spearman_R` previously used a Fisher z-transform — the standard treatment for combining several *estimates of the same* correlation, not for averaging across *different* endpoints, where it let one or two near-perfect endpoints dominate the macro score; see `docs/scoring-and-leaderboards.md` for the full explanation.)
4. Aggregate to mean/std by endpoint (including the macro pseudo-endpoint) for leaderboard use.

Both tracks score against the same shared ground truth (`load_activity_ground_truth()`,
`activity-dataset.parquet`) — they cover the same underlying compound set, just
different columns.

Output helpers:

- `score_activity_predictions(predictions, ground_truth, endpoints)` → per-sample, per-endpoint results for the given `endpoints` list (no macro row — callers append that separately)
- `add_macro_endpoint(...)` → appends that track's macro row
- `compute_macro_bootstrap_results(...)` → per-sample macro-averaged results across one track's endpoints (called by `add_macro_endpoint`)
- `average_bootstrap_results_by_endpoint(...)` → endpoint-level mean/std summary
- `pivot_endpoint_results_wide(...)` → flattens one track's per-endpoint summary into the single wide row saved as that track's `averaged-results.parquet`, with every endpoint's columns consistently prefixed (e.g. `ENDPOINT_1_MAE_mean`, `MA_ST-RAE_mean` for regression; `ENDPOINT_5_MCC_mean`, `MA_MCC_mean` for classification)

A leaderboard for any single endpoint (macro or a specific real endpoint) is built by narrowing back down to that endpoint's columns and stripping the prefix — see `aws_leaderboards._narrow_averaged_results_to_endpoint` — so `primary_metric` is always a bare metric name (e.g. `ST-RAE` or `MCC`) regardless of which endpoint a given leaderboard targets.


## S3, Manifest, and Leaderboard Workflow

All S3 path logic is managed by the `TrackPaths` dataclass in `config.py`. `ACTIVITY_PATHS` is used only for ground-truth/identifiers loading (one shared dataset covering both regression and classification's compound set). `REGRESSION_PATHS`, `CLASSIFICATION_PATHS`, and `STRUCTURE_PATHS` are three fully independent tracks — each with its own submission-upload prefix, and used for everything downstream of scoring — manifest, scores, and leaderboards. See [docs/phases-and-stages.md](../docs/phases-and-stages.md) if the compound-set "phase" (`0`/`1`, used below for `scores/`) vs. leaderboard "stage" (`live`/`interim`/`final`) distinction is unclear.

Submission S3 keys are parsed and validated using the `SubmissionKey` dataclass, which ensures correct file structure and track/file matching.

### Manifest and Leaderboard Generation

- **Manifest creation**: Uses `create_manifest(track, only_latest=True)` in `aws_manifest.py` to aggregate validated submission metadata for each track. Only the latest submission per user is included by default.
- **Score loading**: Batch loads all required `averaged-results.parquet` and `bootstrap-results.parquet` files for leaderboard generation using DuckDB and S3 `httpfs`.
- **Leaderboard generation**: Uses the modular `FinalLeaderboard` class in `leaderboard.py` to:
  - Sort by the primary metric (as a string, e.g. `ST-RAE` or `LDDT-PLI`) and submission time.
  - Optionally perform pairwise statistical comparisons and generate Compact Letter Display (CLD) columns for significance grouping.
  - Output a DataFrame with all metrics, user info, and optional CLD.
- **One leaderboard per endpoint, plus a master**: `aws_leaderboards.create_track_leaderboards(track_paths, stage, ...)` builds one leaderboard per `track_paths.endpoints` entry, plus an additional macro-ranked master (`endpoint=MACRO_ENDPOINT_LABEL`, "MA") whenever a track has more than one endpoint — a single-endpoint track like structure just gets its one leaderboard, which also doubles as the "master" for significance-testing purposes. Every variant is saved under its own `{endpoint_slug}_leaderboard_{version}.csv` filename via `save_leaderboard(..., endpoint_slug=...)`.
- **`stage` derives significance testing and filtering internally** (not passed by the caller):
  - `stage="live"`: no significance testing, no invalid-submission filtering — kept cheap since it's auto-generated on a schedule and always reflects every submission so far.
  - `stage="interim"`: significance testing (`significant_method`, default `"CLD"`) on the master leaderboard only; no invalid-submission filtering.
  - `stage="final"`: significance testing on the master leaderboard only; invalid submissions (bad HF username / model report link) are removed.
  - Significance testing never runs on the individual per-endpoint leaderboards of a multi-endpoint track — only on the master (or the one leaderboard of a single-endpoint track).
- **Live leaderboard**: Saved to S3 using `save_live_leaderboard()`/`save_leaderboard(..., stage="live")`, always overwriting `{endpoint_slug}_leaderboard_latest.csv`. Auto-generated by the leaderboard Lambda from phase 1 (`scores/phase_1`) scores.
- **Interim/final leaderboards**: Generated manually (not by a scheduled Lambda) via `create_track_leaderboards(stage="interim" | "final")`, both scored from `scores/all`, differing only in the submission cutoff (`INTERIM_LEADERBOARD_DEADLINE` vs. `FINAL_LEADERBOARD_DEADLINE`) and, for "final", invalid-submission removal (see above).

### Entries List (Every Track)

Separate from the leaderboard manifests above — a full, unfiltered roster of every
valid, scored submission across **every** track, for entrant/publication tracking
rather than ranking:

- **`create_all_entries_list()`**: Calls `create_manifest(track, only_latest=False, date_cutoff=None)` for `"regression"`, `"classification"`, and `"structure"`, adds a `"track"` column to each (if not already present), and concatenates them into one DataFrame. No latest-submission-only filtering and no date cutoff — every valid, scored submission from every user is included. A track with no scored submissions (or, for classification, no endpoints at all) simply contributes nothing.
- **`save_all_entries_list(entries_df)`**: Saves to S3, mirroring the leaderboard save pattern — a dated snapshot plus an overwritten `all_entries_latest.csv`.
- Run daily by the entries list Lambda (`lambda_handler_entries.py`), scheduled via `opentofu/eventbridge.tf` — see the OpenTofu README for the schedule.

### Example S3 Key Structure

- Submissions: `submissions/{track}/{user_id}/{submission_id}/{filename}` — `{track}` is `regression`, `classification`, or `structure`, each with its own independent upload
- Manifest: `submissions/manifest/{track}/{user_id}_{submission_id}.parquet` — `{track}` is `regression`, `classification`, or `structure`, one manifest row per submission
- Scores: `scores/all/{track}/{user_id}/{submission_id}/averaged-results.parquet`, `scores/phase_1/{track}/{user_id}/{submission_id}/averaged-results.parquet`
- Leaderboard: `leaderboard/{live,interim,final}/{track}/{endpoint_slug}_leaderboard_latest.csv` — `{endpoint_slug}` is `MA` for the master (multi-endpoint tracks only), `Structure` for the structure track's one leaderboard, one of the regression endpoints (e.g. `ENDPOINT_1`) for `track="regression"`, or one of the classification endpoints (e.g. `ENDPOINT_5`) for `track="classification"`
- Entries list (every track combined): `entries/all_entries_latest.csv`

All S3 paths are constructed using the `TrackPaths` dataclass helpers (including `.all_entries` for the entries list — the same prefix for every track, since the list spans all of them), not by string concatenation.


## Leaderboard Features

- **Pairwise significance testing**: The leaderboard system supports pairwise statistical comparisons between submissions using bootstrap samples, with Holm-Bonferroni correction for multiple testing.
- **Compact Letter Display (CLD)**: Optionally generates CLD columns to indicate statistically indistinguishable groups.
- **Dataclass-based entries**: All leaderboard rows are constructed from `LeaderboardEntry` dataclasses, which parse and store all metric and user info.

### `lambda_handler_regression.handler(event, context)`

Expected event detail:

- `event["detail"]["file_path"]` or `event["detail"]["object"]["key"]` path shaped like:
  - `submissions/regression/{user_id}/{submission_uuid}/predictions.parquet`
  - or `.csv`

Behavior:

- validates path structure
- extracts `user_id` and `submission_uuid`
- calls `process_new_regression_submission(...)`

### `lambda_handler_classification.handler(event, context)`

Expected event detail (same shape as regression):

- `event["detail"]["file_path"]` or `event["detail"]["object"]["key"]` path shaped like:
  - `submissions/classification/{user_id}/{submission_uuid}/predictions.parquet`
  - or `.csv`

Behavior:

- validates path structure
- extracts `user_id` and `submission_uuid`
- calls `process_new_classification_submission(...)`

### `lambda_handler_structure.handler(event, context)`

Expected event detail (same shape as regression/classification):

- `event["detail"]["file_path"]` or `event["detail"]["object"]["key"]` path shaped like:
  - `submissions/structure/{user_id}/{submission_uuid}/structures.zip`

Behavior:

- validates path structure
- extracts `user_id` and `submission_uuid`
- calls `process_new_structure_submission(...)`

### `lambda_handler_entries.handler(event, context)`

Scheduled (daily, see `opentofu/eventbridge.tf`). No expected event payload.

Behavior:

- calls `create_all_entries_list()` for every track combined
- calls `save_all_entries_list(...)` to write the dated snapshot + `all_entries_latest.csv`
- returns `{"statusCode": 200, "message": ..., "rows": <int>}`

### `lambda_handler_hf_restart.handler(event, context)`

Restarts HF Space using env vars:

- `HF_OWNER`
- `HF_SPACE_NAME`
- `HF_TOKEN`

## CLI Usage

From repository root:

```bash
# Regression only
python -m backend.cli --regression-predictions path/to/predictions.parquet

# Classification only
python -m backend.cli --classification-predictions path/to/predictions.parquet

# Structure only
python -m backend.cli --structure-predictions path/to/structures.zip

# All three
python -m backend.cli \
  --regression-predictions path/to/regression_predictions.csv \
  --classification-predictions path/to/classification_predictions.csv \
  --structure-predictions path/to/structures.zip
```

Rules:

- At least one of the three options must be provided.
- Regression and classification files must be `.parquet` or `.csv`.
- Structure files are validated as `.zip`.


## Configuration

- All constants (dataset sizes, metrics, deadlines, etc.) are defined in `config.py`.
- `S3_BUCKET` is configured via the `S3_BUCKET` environment variable (default: `"blind-challenge-template"`).
- Discord notifications are controlled by the `DISCORD_NOTIFICATIONS` environment variable (default: enabled).
- Leaderboard sort keys are now strings (e.g. `ST-RAE`, `LDDT-PLI`), not lists.


## Known TODOs

- Deeper structure validation rules beyond file count and PDB format (e.g., ligand residue naming, ligand/protein chain correctness, binding site geometry) — see the note under Structure submissions above
- Implement structure-scoring quality checks (currently OST score is trusted as-is)
