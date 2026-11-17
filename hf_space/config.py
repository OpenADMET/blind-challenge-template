"""Configuration for the blind challenge Hugging Face Space.

Everything a challenge author customises on the Space side: the schedule, submission
rules, links, tracks (which must mirror ``backend/config.py``; the drift test checks),
what each leaderboard shows, appearance and the page text. Work through every
``TODO`` when standing up a new challenge.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

# ------------------------------- Phase and schedule ------------------------------------

# Set by hand to control what the Space shows:
# 0: before the challenge
# 1: open, live leaderboard only
# 2: open, live leaderboard and the interim leaderboard
# 3: closed, final leaderboard not yet released
# 4: closed, final leaderboard (and Head-to-Head tab) released
CURRENT_PHASE = 0

# TODO: this challenge's submission deadlines, as in backend/config.py (UTC, format
# "%Y-%m-%dT%H:%M:%SZ"; the drift test checks they match). The Space uses them for
# the status line under the header and the model report reminder.
INTERIM_LEADERBOARD_DEADLINE = "2099-01-01T23:59:59Z"
FINAL_LEADERBOARD_DEADLINE = "2099-12-31T23:59:59Z"
# With REQUIRE_MODEL_REPORT, a banner reminds participants to add a model report link
# this many days before FINAL_LEADERBOARD_DEADLINE.
MODEL_REPORT_REMINDER_DAYS = 7

# --------------------------------- Submission rules ------------------------------------

HOURS_BETWEEN_SUBMISSIONS = 12
# Only Hugging Face accounts at least this old can submit, which makes throwaway
# accounts for dodging the cooldown less useful. 0 turns the check off.
MIN_ACCOUNT_AGE_DAYS = 1
_MIN_ACCOUNT_AGE = (
    f"{MIN_ACCOUNT_AGE_DAYS} day{'s' if MIN_ACCOUNT_AGE_DAYS != 1 else ''}"
)
# Minimum standard deviation of each regression prediction column. Constant columns
# (with or without slight noise added) can be used to game the leaderboard metric.
MIN_PREDICTION_STD = 0.01
# Entries on the final leaderboard must have a reachable model report link. Must
# mirror backend.config.REQUIRE_MODEL_REPORT.
REQUIRE_MODEL_REPORT = True

# -------------------------------------- Links ------------------------------------------

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
# TODO: invite link to this challenge's Discord server, or "" for no Discord button.
DISCORD_LINK = "https://discord.gg/<invite-code>"

# -------------------------------------- Tracks -----------------------------------------

# Track names, as in S3 prefixes. Must mirror backend.config's TrackSpec.track values.
REGRESSION_TRACK = "regression"
CLASSIFICATION_TRACK = "classification"
STRUCTURE_TRACK = "structure"

# TODO: rename to this challenge's track labels (shown on the Submit tab and as
# leaderboard tab titles).
REGRESSION_NAME = "Regression Prediction"
CLASSIFICATION_NAME = "Classification Prediction"
STRUCTURE_NAME = "Structure Prediction"
TRACK_NAMES: dict[str, str] = {
    REGRESSION_TRACK: REGRESSION_NAME,
    CLASSIFICATION_TRACK: CLASSIFICATION_NAME,
    STRUCTURE_TRACK: STRUCTURE_NAME,
}

IDENTIFIER_COLUMNS = ["SMILES", "Molecule_Name"]
# TODO: must equal backend.config.ACTIVITY_DATASET_SIZE / STRUCTURE_DATASET_SIZE
# (the drift test checks). Regression and classification share the activity dataset.
ACTIVITY_DATASET_SIZE = 750
STRUCTURE_DATASET_SIZE = 184
# The macro-averaged pseudo-endpoint of a multi-endpoint track: its overall
# leaderboard. Must mirror backend.config.MACRO_ENDPOINT_LABEL.
MACRO_ENDPOINT_LABEL = "MA"

# Compound sets and leaderboard stages. Must mirror backend/config.py.
COMPOUND_SETS = ("all", "live", "hidden")
LEADERBOARD_STAGES = ("live", "interim", "final")
LeaderboardStage = Literal["live", "interim", "final"]
DEFAULT_LEADERBOARD_SETS = {"live": "live", "interim": "all", "final": "all"}


@dataclass(frozen=True)
class TrackSpec:
    """What a competition track is. Mirrors ``backend.config.TrackSpec``.

    ``tests/backend/test_config_drift.py`` checks every field matches the backend's.
    See the backend class for what each field means.

    Attributes
    ----------
    track : str
        Track name, as in S3 prefixes.
    dataset : str
        Ground-truth dataset the track scores against.
    endpoints : list[str]
        Endpoint names, in display order; one leaderboard tab each.
    metrics : list[str]
        Metric names, in display order.
    primary_metric : str
        The metric leaderboards are ranked by.
    higher_is_better : bool
        Direction of ``primary_metric``.
    dataset_size : int
        Number of compounds a submission must cover.
    extra_columns : tuple[str, ...]
        Additional averaged-results columns carried onto the leaderboard.
    leaderboard_sets : dict[str, str]
        The compound set ("live", "hidden" or "all") each leaderboard stage ranks
        on.

    """

    track: str
    dataset: str
    endpoints: list[str]
    metrics: list[str]
    primary_metric: str
    higher_is_better: bool
    dataset_size: int
    extra_columns: tuple[str, ...] = field(default=())
    leaderboard_sets: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LEADERBOARD_SETS)
    )

    def __post_init__(self) -> None:
        """Reject a spec that can't be ranked or displayed consistently.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``primary_metric`` isn't one of ``metrics``, a metric or endpoint is
            listed twice, ``dataset_size`` is negative, or ``leaderboard_sets``
            doesn't map every stage to a known compound set.

        """
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"{self.track}: primary_metric {self.primary_metric!r} is not one of "
                f"its metrics {self.metrics}"
            )
        if set(self.leaderboard_sets) != set(LEADERBOARD_STAGES) or not set(
            self.leaderboard_sets.values()
        ) <= set(COMPOUND_SETS):
            raise ValueError(
                f"{self.track}: leaderboard_sets must map each of {LEADERBOARD_STAGES} "
                f"to one of {COMPOUND_SETS}, got {self.leaderboard_sets}"
            )
        for name, values in (("metrics", self.metrics), ("endpoints", self.endpoints)):
            duplicates = sorted({v for v in values if values.count(v) > 1})
            if duplicates:
                raise ValueError(f"{self.track}: duplicate {name} {duplicates}")
        if self.dataset_size < 0:
            raise ValueError(f"{self.track}: dataset_size must not be negative")


# TODO: must match backend/config.py's specs exactly (the drift test checks).
REGRESSION_SPEC = TrackSpec(
    track=REGRESSION_TRACK,
    dataset="activity",
    endpoints=["ENDPOINT_1", "ENDPOINT_2", "ENDPOINT_3", "ENDPOINT_4"],
    metrics=["ST-RAE", "MAE", "R2", "Spearman_R", "Kendall_Tau"],
    primary_metric="ST-RAE",
    higher_is_better=False,
    dataset_size=ACTIVITY_DATASET_SIZE,
)
CLASSIFICATION_SPEC = TrackSpec(
    track=CLASSIFICATION_TRACK,
    dataset="activity",
    endpoints=["ENDPOINT_5", "ENDPOINT_6"],
    metrics=["MCC", "Accuracy", "Precision", "Recall", "F1"],
    primary_metric="MCC",
    higher_is_better=True,
    dataset_size=ACTIVITY_DATASET_SIZE,
)
STRUCTURE_SPEC = TrackSpec(
    track=STRUCTURE_TRACK,
    dataset="structure",
    endpoints=["structure"],
    metrics=["LDDT-PLI", "BiSyRMSD", "LDDT-LP"],
    primary_metric="LDDT-PLI",
    higher_is_better=True,
    dataset_size=STRUCTURE_DATASET_SIZE,
    extra_columns=("coverage_mean",),
)
TRACK_SPECS: dict[str, TrackSpec] = {
    spec.track: spec for spec in (REGRESSION_SPEC, CLASSIFICATION_SPEC, STRUCTURE_SPEC)
}

REQUIRED_REGRESSION_COLUMNS = IDENTIFIER_COLUMNS + REGRESSION_SPEC.endpoints
REQUIRED_CLASSIFICATION_COLUMNS = IDENTIFIER_COLUMNS + CLASSIFICATION_SPEC.endpoints

# --------------------------------- Track visibility ------------------------------------


@dataclass(frozen=True)
class TrackVisibility:
    """Where a track is shown on the Space.

    Attributes
    ----------
    submissions_open : bool
        Show the track's schema and its option in the Submit tab's track selector.
    live : bool
        Show the track's tab on the live leaderboards.
    interim : bool
        Show the track's tab on the interim leaderboards.
    final : bool
        Show the track's tab on the final leaderboards.

    """

    submissions_open: bool
    live: bool = True
    interim: bool = True
    final: bool = True

    def on_leaderboard(self, stage: LeaderboardStage) -> bool:
        """Whether the track has a tab on the given leaderboard stage.

        Parameters
        ----------
        stage : LeaderboardStage
            "live", "interim" or "final".

        Returns
        -------
        bool
            True if the track is shown on that stage's leaderboards.

        """
        return bool(getattr(self, stage))


# TODO: set for this challenge, and edit the page markdown below to match. A track
# with an empty endpoint list (e.g. CLASSIFICATION_SPEC.endpoints = []) is never shown,
# whatever its visibility. CURRENT_PHASE separately decides which leaderboard stages
# exist at all.
TRACK_VISIBILITY: dict[str, TrackVisibility] = {
    REGRESSION_TRACK: TrackVisibility(submissions_open=True),
    CLASSIFICATION_TRACK: TrackVisibility(submissions_open=True),
    STRUCTURE_TRACK: TrackVisibility(
        submissions_open=False, live=False, interim=False, final=False
    ),
}

# ------------------------------- Leaderboard display -----------------------------------

# How each metric (or extra column, without its "_mean") is labelled on the
# leaderboards; one not listed is shown under its own name.
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "R2": "R²",
    "Spearman_R": "Spearman's ρ",
    "Kendall_Tau": "Kendall's τ",
    "F1": "F1 Score",
    "coverage": "Coverage",
}
# Whether a higher value of each metric (or extra column) is better, for the ↑/↓
# in the leaderboard headers. A metric not listed gets no arrow; a track's primary
# metric uses its TrackSpec.higher_is_better.
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "ST-RAE": False,
    "MAE": False,
    "R2": True,
    "Spearman_R": True,
    "Kendall_Tau": True,
    "MCC": True,
    "Accuracy": True,
    "Precision": True,
    "Recall": True,
    "F1": True,
    "LDDT-PLI": True,
    "BiSyRMSD": False,
    "LDDT-LP": True,
    "coverage": True,
}


def metric_display_names(spec: TrackSpec) -> dict[str, str]:
    """Map a track's metric names to their leaderboard labels, in display order.

    Parameters
    ----------
    spec : TrackSpec
        The track.

    Returns
    -------
    dict[str, str]
        Metric name to label, in ``spec.metrics`` order.

    """
    return {m: METRIC_DISPLAY_NAMES.get(m, m) for m in spec.metrics}


# Significant figures for scores on each leaderboard stage. The live leaderboard shows
# fewer, so it gives away less about the test set while the challenge is running.
LEADERBOARD_SIG_FIGS: dict[LeaderboardStage, int] = {
    "live": 2,
    "interim": 3,
    "final": 3,
}
# Stages whose scores are also rounded in the data the Space sends (the values the
# table sorts by, and the CSV download), so full-precision scores never reach the
# browser. On other stages only the text shown is rounded.
ROUNDED_STAGES: tuple[LeaderboardStage, ...] = ("live",)

# ---------------------------------- Head-to-head ---------------------------------------

# The Head-to-Head tab (phase 4 only) compares two final-leaderboard entrants on every
# test compound, from release files written by scripts/final_leaderboard.ipynb
# (backend/head_to_head_release.py). Set to False to leave the tab out.
HEAD_TO_HEAD_ENABLED = True
# Which per-compound views each track gets: "regression" (predicted vs measured, and
# who is closer), "classification" (agreement grid and confusion matrices) or
# "structure" (per-compound score bars). A track not listed has no head-to-head.
HEAD_TO_HEAD_KINDS: dict[str, Literal["regression", "classification", "structure"]] = {
    REGRESSION_TRACK: "regression",
    CLASSIFICATION_TRACK: "classification",
    STRUCTURE_TRACK: "structure",
}
# Where the release files are ({file}: ground_truth, entries, pairwise, bootstrap,
# predictions or per_compound). Must mirror backend.config.TrackPaths.head_to_head.
HEAD_TO_HEAD_URI_FORMAT = "leaderboard/final/{track}/head_to_head/{file}.parquet"
# Development only: read the release files from {H2H_LOCAL_DIR}/{track}/{file}.parquet.
H2H_LOCAL_DIR: str = os.environ.get("H2H_LOCAL_DIR", "")

# ----------------------------------- Appearance ----------------------------------------

# TODO (optional): the Space's colours. Each is a Gradio colour name ("blue", "sky",
# "emerald", "violet", "slate", "zinc", ...) or a hex colour such as "#1f6feb", from
# which lighter and darker shades are made. Primary is buttons and highlights,
# secondary is accents, neutral is backgrounds, borders and text.
THEME_PRIMARY_COLOUR = "blue"
THEME_SECONDARY_COLOUR = "sky"
THEME_NEUTRAL_COLOUR = "slate"
# TODO (optional): a Google Fonts family name, e.g. "Inter", or None for Gradio's.
THEME_FONT: str | None = None
# TODO: add the challenge logo here (not shipped in the template). The header shows it
# when the file exists. Relative to hf_space/.
LOGO_PATH = "_static/challenge_logo.png"
# A small "Built with the blind challenge template" line under every page, linking to
# the template and its citation. Set to False to leave it out.
SHOW_TEMPLATE_CREDIT = True
TEMPLATE_CREDIT_MD = (
    "Built with the OpenADMET [blind challenge template]"
    "(https://github.com/OpenADMET/blind-challenge-template) "
    "([cite](https://doi.org/10.5281/zenodo.22780110))"
)

# --------------------------------------- S3 --------------------------------------------

S3_BUCKET: str = os.environ.get("S3_BUCKET", "")
AWS_DEFAULT_REGION: str = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
# Where the backend writes each leaderboard: {leaderboard_type} is the stage, {track}
# the track, {endpoint_slug} an endpoint or MACRO_ENDPOINT_LABEL, and {version}
# "latest" or a timestamp. Must mirror backend.aws_leaderboards._leaderboard_file_path
# (the drift test checks).
LEADERBOARD_URI_FORMAT = (
    "leaderboard/{leaderboard_type}/{track}/{endpoint_slug}_leaderboard_{version}.csv"
)

# ------------------------- Submission file formats (page text) -------------------------


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
# REGRESSION_SPEC.endpoints / CLASSIFICATION_SPEC.endpoints so the two can't drift out of sync
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
    [_endpoint_schema_row(e, "regression") for e in REGRESSION_SPEC.endpoints]
)
SUBMISSION_SCHEMA_TABLE_CLASSIFICATION_MD = _schema_table_md(
    [_endpoint_schema_row(e, "classification") for e in CLASSIFICATION_SPEC.endpoints]
)


def _example_row(identifiers: tuple[str, str], values: list) -> str:
    return ",".join([*identifiers, *[str(v) for v in values]])


def _csv_example_md(
    endpoint_columns: list[str], example_rows: list[tuple[tuple[str, str], list]]
) -> str:
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
    REGRESSION_SPEC.endpoints,
    [
        (("CCO", "OADMET-00000"), [6.23, 5.94, 6.78, 5.41]),
        (("c1ccccc1", "OADMET-00001"), [5.87, 6.10, 5.55, 6.02]),
    ],
)
SUBMISSION_CSV_EXAMPLE_CLASSIFICATION_MD = _csv_example_md(
    CLASSIFICATION_SPEC.endpoints,
    [
        (("CCO", "OADMET-00000"), [False, True]),
        (("c1ccccc1", "OADMET-00001"), [True, False]),
    ],
)

# ------------------------------- Challenge Page Content -------------------------------

# Challenge page content
#
# The markdown blocks below are all the challenge-specific prose on the
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
HEADER_MARKDOWN = """# Blind Challenge: <Challenge Title> 🧬
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
HOW_TO_PARTICIPATE_MD = f"""### ✅ How to Participate

1. **Register:** Create a personal Hugging Face account{f" (it can submit once it's {_MIN_ACCOUNT_AGE} old)" if MIN_ACCOUNT_AGE_DAYS else ""}.
2. **Get the Data:** Download the training and test sets from the [challenge dataset page]({DATASET_DOWNLOAD_LINK}).
3. **Walk through the Tutorial:** Check out the baseline notebooks and validation scripts in the [challenge tutorial repository]({TUTORIAL_LINK}).
4. **Disclosures:** Indicate whether your submission contributes open-source code and disclose any use of **proprietary training data**.
5. **Join the Community:** Get support, ask questions, and collaborate in the `#<your-challenge>` channel on our [Discord]({DISCORD_LINK}).
6. **Submit:** Log in with Hugging Face on the **Submit** tab and upload your predictions.

---"""

# TODO: fill in every <...> section below. Suggested sections (delete any that do
# not apply to this challenge): Background (why the endpoint matters) · The
# Dataset (how train/test were built, sizes) · Developing the Assay (optional) ·
# The Challenge Tracks (one paragraph per track + its scoring) · Timeline ·
# Acknowledgements. Rendered on the "About" tab.
ABOUT_MD = f"""
## About the Challenge

### Background: Why This Matters

<Why does predicting this endpoint / property matter for drug discovery? What
makes it hard to model? Keep it to a few short paragraphs.>

---

### The Dataset

<How were the training and test sets generated? Assay(s) used, compound sources,
approximate sizes. The test set has {ACTIVITY_DATASET_SIZE} compounds.>

<!-- TODO: delete this section if the challenge has no wet-lab assay to describe. -->
### Developing the Assay

<Assay format, readout, controls, anything participants should know about the
measurement noise model.>

---

### 🧪 The Challenge Tracks

<Describe each active track. Endpoint names and the submission schema are
generated automatically from config further down this tab.>

#### Regression Track
<What is predicted, over how many endpoints, and the primary + secondary metrics.>

#### Classification Track
<What is predicted and the primary + secondary metrics. Delete this section if
CLASSIFICATION_SPEC.endpoints is empty.>

#### Structure Track
<Pose-prediction task description. Delete this section if there is no structure
track.>

---

{HOW_TO_PARTICIPATE_MD if CURRENT_PHASE in [1, 2] else ""}

### 📅 Timeline

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

### Acknowledgements
<!-- TODO: name the experimentalists / institutions / funding that contributed to this challenge. -->
"""

# TODO: fill in the challenge-specific answers (goal, structure, scoring, label
# definitions). The rate-limit / teams / proprietary-data / open-source / file-
# format answers below are challenge-agnostic and pull details from config —
# leave them unless this challenge's rules differ. Rendered on the "About" tab.
FAQ_MD = f"""
## Frequently Asked Questions

**What is the goal of this challenge?**
<One or two sentences: what participants predict, over how many endpoints, and for
how many test-set compounds ({ACTIVITY_DATASET_SIZE}).>

**How is this challenge structured?**
<Single continuous stage, or sequential unblinding phases? When (if ever) is the
test set partially revealed? See docs/phases-and-stages.md.>

**Can I compete in one or more tracks?**
Yes! Each track is an independent submission (its own file), scored independently and ranked on its own leaderboard. A strong entry in one track is ranked regardless of whether you submit to the others.

**I want to participate with colleagues. Should we submit together or separately?**
Please submit from one team member's personal Hugging Face account, using your team or lab name as the alias. (Organisation accounts can't log in.) Multiple submissions from the same team or lab (i.e., individuals who cooperate intensively to prepare entries) are not permitted.

**Are there limits on submission frequency?**
Yes, submissions are rate-limited to at most once every {HOURS_BETWEEN_SUBMISSIONS} hours per track, per Hugging Face account. Only your latest valid submission counts toward the live leaderboard standings. Please run the validation script before uploading! Using more than one account to get around the limit breaks the rules and can get your accounts banned.

**Can I submit through the API or a script?**
No. Submissions are made on the **Submit** tab while logged in with Hugging Face, so every entry belongs to a real account.{f" New accounts can submit once they're {_MIN_ACCOUNT_AGE} old." if MIN_ACCOUNT_AGE_DAYS else ""}

**Where do I ask technical questions?**
For technical questions regarding dataset schema, infrastructure, or submission mechanics, use the `#<your-challenge>` [channel on the Discord]({DISCORD_LINK}).

---

### Evaluation & Scoring

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

### Data, Rules & Submissions

**Can I use external or proprietary data to train my models?**
Yes! External data and pretrained models are fully permitted. However, you must disclose whether you used proprietary training data by checking the *"I used proprietary data"* box during submission. This flag will be publicly displayed on the leaderboard.

**Am I required to open-source my code?**
Open-sourcing code is strongly encouraged (and required if you want to win kudos from the community or qualify for novel machine learning recognition), but not strictly mandatory for standard leaderboard ranking. When submitting, tick the *"Open Source Code"* checkbox and provide a link to your public repository. This will need to be a publicly accessible link at submission time, and will be displayed on the leaderboard.

**Will there be a summary preprint paper?**
No. To maximize open-science velocity and release datasets as quickly as possible, we publish detailed post-challenge writeups, deep-dive benchmarks, and analysis as **blog posts** rather than traditional preprints. Participants are actively encouraged to publish their own methods and findings independently!

---

### Technical & Submission File Formats

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
