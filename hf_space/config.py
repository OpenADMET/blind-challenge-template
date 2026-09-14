"""Configuration constants for the blind challenge HuggingFace Space.

Everything a challenge author needs to customise on the Space side lives here:
the endpoint lists / dataset sizes (which must mirror ``backend/config.py``), the
external links, and the page-content markdown blocks. Work through every ``TODO``
below when standing up a new challenge.
"""

import os

# -------------------------------- Global Params ---------------------------------------

# This is updated manually to allow control over what is visible on the HF Space.
# Possible phases:
# 0: Pre-challenge
# 1: Phase 1 (only live leaderboard)
# 2: Phase 2 (live leaderboard + static interim leaderboard)
# 3: Challenge closed, final leaderboard not yet released
# 4: Challenge closed, final leaderboard released
CURRENT_PHASE = 0
HOURS_BETWEEN_SUBMISSIONS = 12
# Minimum standard deviation for regression predictions to be considered valid.
# Constant columns (with or without slight noise added) can be used to game the
# leaderboard metric.
MIN_PREDICTION_STD = 0.01

# Leaderboard and S3 configuration.
# Regression, classification, and structure are three fully independent backend
# tracks (each with its own submissions/scores/leaderboard S3 prefix, Lambda,
# per-track macro-average, etc.). Must mirror backend.config.TrackPaths.track.
REGRESSION_TRACK = "regression"
CLASSIFICATION_TRACK = "classification"
STRUCTURE_TRACK = "structure"

# {track} is one of REGRESSION_TRACK / CLASSIFICATION_TRACK / STRUCTURE_TRACK.
# {endpoint_slug} is MACRO_ENDPOINT_LABEL ("MA") for a track's master (macro-ranked)
# leaderboard (only present once a track has more than one endpoint), a
# REGRESSION_ENDPOINTS/CLASSIFICATION_ENDPOINTS entry (e.g.
# "ENDPOINT_1") for that endpoint's own leaderboard, or
# STRUCTURE_ENDPOINTS[0] ("Structure") for the structure track's one leaderboard —
# see leaderboards.format_leaderboard_uri.
# {version} is "latest" or an ISO timestamp.
# Every leaderboard file (all tracks, all stages) follows this one convention — must
# mirror backend.aws_leaderboards._leaderboard_file_path.
LEADERBOARD_URI_FORMAT = (
    "leaderboard/{leaderboard_type}/{track}/{endpoint_slug}_leaderboard_{version}.csv"
)
S3_BUCKET: str = os.environ.get("S3_BUCKET", "")
AWS_DEFAULT_REGION: str = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
# TODO: link to this challenge's announcement blog post.
CHALLENGE_ANNOUCEMENT_LINK = "https://<ORG-WEBSITE>/<challenge-announcement-post>"
# TODO: link to the HuggingFace dataset holding this challenge's train/test data.
DATASET_DOWNLOAD_LINK = (
    "https://huggingface.co/datasets/<ORG>/<challenge-train-test-dataset>"
    if CURRENT_PHASE >= 1
    else False
)
# TODO: link to this challenge's tutorial / baseline-notebook repository.
TUTORIAL_LINK = "https://github.com/<ORG>/<Challenge-Tutorial>"

# Activity bootstrap-results.parquet files hold one row per (Sample, Endpoint) pair,
# including this synthetic pseudo-endpoint holding the multi-endpoint macro-averaged
# scores for that bootstrap sample. Must mirror backend.config.MACRO_ENDPOINT_LABEL.
MACRO_ENDPOINT_LABEL = "MA"

# -------------------------------- Activity Track Params -------------------------------

# Dataset sizes and required columns
# TODO: must equal backend.config.ACTIVITY_DATASET_SIZE.
ACTIVITY_DATASET_SIZE = 750
IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]

# Drives one leaderboard tab per activity endpoint. Must mirror
# backend.config.ACTIVITY_ENDPOINTS (same names, same order).
# TODO: rename *_NAME to this challenge's track labels, and the endpoint lists to
# match backend/config.py exactly.
REGRESSION_NAME = "Regression Prediction"
REGRESSION_ENDPOINTS = [
    "ENDPOINT_1",
    "ENDPOINT_2",
    "ENDPOINT_3",
    "ENDPOINT_4",
]
CLASSIFICATION_NAME = "Classification Prediction"
CLASSIFICATION_ENDPOINTS = [
    "ENDPOINT_5",
    "ENDPOINT_6",
]

# Show or hide each activity track on the Submit tab: its schema block and its
# entry in the "Track" selector (see render_submission_tab). These are a
# submission switch only — they do NOT touch the leaderboards. A track's
# leaderboard tab appears whenever its endpoint list (REGRESSION_ENDPOINTS /
# CLASSIFICATION_ENDPOINTS) is non-empty; that's the knob for leaderboard
# visibility. For a track the challenge doesn't run at all, set its endpoint list
# to []. When you flip one of these, edit the page markdown further down so the
# copy participants read matches what's actually open.
REGRESSION_ACTIVE = True
CLASSIFICATION_ACTIVE = True

# Regression and classification are independent submission tracks (see
# render_submission_tab / Submission.track) — REQUIRED_REGRESSION_COLUMNS and
# REQUIRED_CLASSIFICATION_COLUMNS drive each upload's own validation.
# ACTIVITY_DATASET_SIZE is the shared row-count both tracks validate against (same
# compounds, different columns) — not a submission-upload concept.
REQUIRED_REGRESSION_COLUMNS = IDENTIFIER_COLUMNS + REGRESSION_ENDPOINTS
REQUIRED_CLASSIFICATION_COLUMNS = IDENTIFIER_COLUMNS + CLASSIFICATION_ENDPOINTS

# Metric names as they appear (unprefixed, once narrowed to a single endpoint) in a
# leaderboard's columns, and how each should be labelled in the UI, in display order
# (primary/sort metric RAE first). Must mirror backend.config.ACTIVITY_METRICS.
ACTIVITY_METRIC_DISPLAY_NAMES = {
    "ST-RAE": "ST-RAE",
    "MAE": "MAE",
    "R2": "R²",
    "Spearman_R": "Spearman's ρ",
    "Kendall_Tau": "Kendall's τ",
}

# Same idea, for the classification track's leaderboard columns. MCC is the
# primary/sort metric, listed first. Must mirror backend.config.CLASSIFICATION_METRICS.
CLASSIFICATION_METRIC_DISPLAY_NAMES = {
    "MCC": "MCC",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1": "F1 Score",
}

# -------------------------------- Structure Track Params -------------------------------

# Turn the structure track on or off. Like the activity flags above this shows or
# hides the track's Submit-tab option; unlike them it also shows/hides the
# structure leaderboard tab (which additionally needs STRUCTURE_ENDPOINTS to be
# non-empty — see leaderboards.py). Edit the page markdown to match when you flip
# it.
STRUCTURE_TRACK_LIVE = False
# TODO: must equal backend.config.STRUCTURE_DATASET_SIZE (0 / unused if this
# challenge has no structure track).
STRUCTURE_DATASET_SIZE = 184

# Structure track's single leaderboard slug. Must mirror
# backend.config.STRUCTURE_ENDPOINTS.
STRUCTURE_NAME = "Structure Prediction"
STRUCTURE_ENDPOINTS = ["structure"]

# ------------------------------- Useful functions -------------------------------------


def _submission_schema_row(column: str, dtype: str, description: str) -> str:
    return f"| `{column}` | {dtype} | {description} |"


def _endpoint_schema_row(endpoint: str, kind: str) -> str:
    """Build a submission schema table row for one regression/classification endpoint."""
    if kind == "regression":
        return _submission_schema_row(
            endpoint, "float", f"Predicted value for `{endpoint}`"
        )
    return _submission_schema_row(
        endpoint,
        "bool",
        f"Predicted class label for `{endpoint}` (`True`/`False` or `1`/`0`)",
    )


# Single source of truth for each track's submission file column schema — shown on
# both the Submit tab (submission.py) and in the FAQ, generated from
# REGRESSION_ENDPOINTS / CLASSIFICATION_ENDPOINTS so the two can't drift out of sync
# with each other or with the actual validation logic (which checks
# REQUIRED_REGRESSION_COLUMNS/REQUIRED_CLASSIFICATION_COLUMNS).
def _schema_table_md(rows: list[str]) -> str:
    return "\n".join(
        [
            "| Column Name | Type | Description |",
            "| :--- | :--- | :--- |",
            _submission_schema_row(
                "SMILES", "string", "SMILES string for the compound"
            ),
            _submission_schema_row(
                "Molecule_Name", "string", "Unique compound identifier"
            ),
            *rows,
        ]
    )


SUBMISSION_SCHEMA_TABLE_REGRESSION_MD = _schema_table_md(
    [_endpoint_schema_row(e, "regression") for e in REGRESSION_ENDPOINTS]
)
SUBMISSION_SCHEMA_TABLE_CLASSIFICATION_MD = _schema_table_md(
    [_endpoint_schema_row(e, "classification") for e in CLASSIFICATION_ENDPOINTS]
)


def _example_row(identifiers: tuple[str, str], values: list) -> str:
    return ",".join([*identifiers, *[str(v) for v in values]])


def _csv_example_md(endpoint_columns: list[str], example_rows: list[list]) -> str:
    header = ",".join(IDENTIFIER_COLUMNS + endpoint_columns)
    rows = "\n".join(
        _example_row(identifiers, values) for identifiers, values in example_rows
    )
    return f"""```
{header}
{rows}
...
```"""


SUBMISSION_CSV_EXAMPLE_REGRESSION_MD = _csv_example_md(
    REGRESSION_ENDPOINTS,
    [
        (("CCO", "OADMET-00000"), [6.23, 5.94, 6.78, 5.41]),
        (("c1ccccc1", "OADMET-00001"), [5.87, 6.10, 5.55, 6.02]),
    ],
)
SUBMISSION_CSV_EXAMPLE_CLASSIFICATION_MD = _csv_example_md(
    CLASSIFICATION_ENDPOINTS,
    [
        (("CCO", "OADMET-00000"), [False, True]),
        (("c1ccccc1", "OADMET-00001"), [True, False]),
    ],
)

# ------------------------------- Challenge Page Content -------------------------------

# Challenge page content
#
# The four markdown blocks below are all the challenge-specific prose on the
# Space. They are placeholder skeletons — replace the <ANGLE_BRACKET> text and
# work through every "TODO" comment. Where each block renders:
#   PAGE_TITLE           -> browser tab / Gradio app title (app.py)
#   HEADER_MARKDOWN       -> top of the landing / Leaderboard tab (app.py)
#   HOW_TO_PARTICIPATE_MD -> spliced into ABOUT_MD during phases 1-2
#   ABOUT_MD             -> the "About" tab
#   FAQ_MD              -> the "About / FAQ" tab
# Keep everything challenge-specific here; app.py / leaderboards.py / submission.py
# only import these names.

# TODO: this challenge's title (short — shows in the browser tab).
PAGE_TITLE = "Blind Challenge: <Challenge Title>"

# TODO: fill in. Keep to a one-line title + 2-4 sentences on what participants
# predict and why it matters, then the "go to Leaderboard / Submit" pointer, then
# the timeline bullets. Rendered at the top of the landing / Leaderboard tab.
HEADER_MARKDOWN = """## Blind Challenge: <Challenge Title> 🧬
<One or two sentences: what participants are predicting (the endpoint / property)
and why accurate prediction matters for drug discovery.>

<Optional second paragraph with more scientific background.>

Go to the **Leaderboard** to see current standings. To participate, head to the **Submit** tab to upload your predictions.

<!-- TODO: replace the timeline below with this challenge's real dates. -->
**Challenge Timeline**:
* Submissions open <DATE>.
* Intermediate submission deadline is <DATE>.
* An intermediate leaderboard will be released <DATE>.
* Submissions will close on <DATE>.
"""

# TODO: fill in. Rendered inside ABOUT_MD while CURRENT_PHASE is 1 or 2.
HOW_TO_PARTICIPATE_MD = f"""## ✅ How to Participate

1. **Register:** Create an account with Hugging Face.
2. **Get the Data:** Download the training and test sets from the [challenge dataset page]({DATASET_DOWNLOAD_LINK}).
3. **Walk through the Tutorial:** Check out the baseline notebooks and validation scripts in the [challenge tutorial repository]({TUTORIAL_LINK}).
4. **Disclosures:** Indicate whether your submission contributes open-source code and disclose any use of **proprietary training data**.
5. **Join the Community:** Get support, ask questions, and collaborate in the `#<your-challenge>` channel on our [Discord](<Discord invite URL>).
6. **Submit:** Upload your predictions via the **Submit** tab.

---"""

# TODO: fill in every <...> section below. Suggested sections (delete any that do
# not apply to this challenge): Background (why the endpoint matters) · The
# Dataset (how train/test were built, sizes) · Developing the Assay (optional) ·
# The Challenge Tracks (one paragraph per track + its scoring) · Timeline ·
# Acknowledgements. Rendered on the "About" tab.
ABOUT_MD = f"""
# <Challenge Title> Blind Challenge

## Background: Why This Matters

<Why does predicting this endpoint / property matter for drug discovery? What
makes it hard to model? Keep it to a few short paragraphs.>

---

## The Dataset

<How were the training and test sets generated? Assay(s) used, compound sources,
approximate sizes. The test set has {ACTIVITY_DATASET_SIZE} compounds.>

<!-- TODO: delete this section if the challenge has no wet-lab assay to describe. -->
## Developing the Assay

<Assay format, readout, controls, anything participants should know about the
measurement noise model.>

---

## 🧪 The Challenge Tracks

<Describe each active track. Endpoint names and the submission schema are
generated automatically from config further down this tab.>

### Regression Track
<What is predicted, over how many endpoints, and the primary + secondary metrics.>

### Classification Track
<What is predicted and the primary + secondary metrics. Delete this section if
CLASSIFICATION_ENDPOINTS is empty.>

### Structure Track
<Pose-prediction task description. Delete this section if there is no structure
track.>

---

{HOW_TO_PARTICIPATE_MD if CURRENT_PHASE in [1, 2] else ""}

## 📅 Timeline

*All submission deadlines are 23:59 UTC.*

<!-- TODO: replace with this challenge's real dates. -->
| Date | Action |
|:--- |:--- |
| **<DATE>** | Challenge announced |
| **<DATE>** | Training/Test sets released; submissions open 🚀 |
| **<DATE> (23:59 UTC)** | Deadline for intermediate leaderboard submissions |
| **<DATE>** | Intermediate leaderboard released |
| **<DATE> (23:59 UTC)** | Submissions close 🏁 |
| **From <DATE>** | Final leaderboard released, webinars, blog posts, and wrap-up |

---

## Acknowledgements
<!-- TODO: name the experimentalists / institutions / funding that contributed to this challenge. -->
"""

# TODO: fill in the challenge-specific answers (goal, structure, scoring, label
# definitions). The rate-limit / teams / proprietary-data / open-source / file-
# format answers below are challenge-agnostic and pull details from config —
# leave them unless this challenge's rules differ. Rendered on the "About" tab.
FAQ_MD = f"""
### Frequently Asked Questions

**What is the goal of this challenge?**
<One or two sentences: what participants predict, over how many endpoints, and for
how many test-set compounds ({ACTIVITY_DATASET_SIZE}).>

**How is this challenge structured?**
<Single continuous stage, or sequential unblinding phases? When (if ever) is the
test set partially revealed? See docs/phases-and-stages.md.>

**Can I compete in one or more tracks?**
Yes! Each track is an independent submission (its own file), scored independently and ranked on its own leaderboard. A strong entry in one track is ranked regardless of whether you submit to the others.

**I want to participate with colleagues. Should we submit together or separately?**
Please submit under a single Hugging Face account representing your lab or team. Multiple submissions from the same team or lab (i.e., individuals who cooperate intensively to prepare entries) are not permitted.

**Are there limits on submission frequency?**
Yes, submissions are rate-limited to at most once every {HOURS_BETWEEN_SUBMISSIONS} hours per track. Only your latest valid submission counts toward the live leaderboard standings. Please run the validation script before uploading!

**Where do I ask technical questions?**
For technical questions regarding dataset schema, infrastructure, or submission mechanics, use the `#<your-challenge>` [channel on the Discord](<Discord invite URL>).

---

#### Evaluation & Scoring

**How are predictions evaluated in the regression track?**
<Primary metric (and why it suits this data), then the secondary metrics. All
metrics are bootstrapped over {{n}} resamples.>

**How are predictions evaluated in the classification track?**
<Primary metric, label definition, secondary metrics. Delete if no classification
track.>

**How exactly are the classification labels assigned?**
<Spell out the positive / negative label rules, including any handling of
low-confidence or below-detection compounds. Delete if no classification track.>

---

#### Data, Rules & Submissions

**Can I use external or proprietary data to train my models?**
Yes! External data and pretrained models are fully permitted. However, you must disclose whether you used proprietary training data by checking the *"I used proprietary data"* box during submission. This flag will be publicly displayed on the leaderboard.

**Am I required to open-source my code?**
Open-sourcing code is strongly encouraged (and required if you want to win kudos from the community or qualify for novel machine learning recognition), but not strictly mandatory for standard leaderboard ranking. When submitting, tick the *"Open Source Code"* checkbox and provide a link to your public repository. This will need to be a publicly accessible link at submission time, and will be displayed on the leaderboard.

**Will there be a summary preprint paper?**
No. To maximize open-science velocity and release datasets as quickly as possible, we publish detailed post-challenge writeups, deep-dive benchmarks, and analysis as **blog posts** rather than traditional preprints. Participants are actively encouraged to publish their own methods and findings independently!

---

#### Technical & Submission File Formats

**What file formats are required for submission?**

Each track's predictions are submitted as an independent file: a `.parquet` (preferred) or `.csv` file containing exactly {ACTIVITY_DATASET_SIZE} rows (one per test set compound).

**Submission Schema Details**

*Regression track:*

{SUBMISSION_SCHEMA_TABLE_REGRESSION_MD}

*Classification track:*

{SUBMISSION_SCHEMA_TABLE_CLASSIFICATION_MD}

*Rules:*
1. Each file must contain exactly {ACTIVITY_DATASET_SIZE} rows—no more, no fewer.
2. Column names are case-sensitive and must match exactly.
3. All prediction columns in a given file are required and must be fully populated for every row: numeric predictions must be finite floats (no `NaN`, `inf`, or `-inf`) and class labels must be valid booleans.
4. Each track's predictions are scored independently and ranked on separate leaderboards (see "Evaluation & Scoring" above).

**Where can I find the submission validation script?**
Validation scripts will be released on GitHub. Always run the validation script locally before uploading to avoid failed submission attempts.

**My entry hasn't appeared on the leaderboard. What should I do?**
First, check the `#<your-challenge>-submissions` channel on Discord for automated receipts and error logs. If your submission failed validation, re-run the local validation script against your file. If no receipt appears after 2 hours, post a message in `#<your-challenge>` for help.
"""
