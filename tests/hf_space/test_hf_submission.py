"""Tests for the HF Space submission form: validation, cooldown and upload."""

import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
import submission
from config import (
    ACTIVITY_DATASET_SIZE,
    CLASSIFICATION_ENDPOINTS,
    HOURS_BETWEEN_SUBMISSIONS,
    REGRESSION_ENDPOINTS,
    REQUIRED_REGRESSION_COLUMNS,
    STRUCTURE_DATASET_SIZE,
)
from models import Submission
from submission import MAX_ALIAS_LENGTH, alias_error, model_report_link_error


@pytest.mark.parametrize(
    "alias", ["Chemprop Wizard", "Team-42 (v2)", "Équipe Café", "a" * MAX_ALIAS_LENGTH]
)
def test_alias_error_accepts_normal_aliases(alias):
    assert alias_error(alias) is None


def test_alias_error_too_long_says_how_long():
    message = alias_error("a" * (MAX_ALIAS_LENGTH + 5))
    assert message is not None
    assert f"{MAX_ALIAS_LENGTH + 5} characters" in message
    assert f"maximum is {MAX_ALIAS_LENGTH}" in message


@pytest.mark.parametrize(
    ("alias", "named"),
    [
        ("<img src=x onerror=alert(1)>", "< >"),
        ("Fish & Chips", "&"),
        ('say "hi"', '"'),
        ("it's", "'"),
        ("back`tick", "`"),
    ],
)
def test_alias_error_names_the_forbidden_characters(alias, named):
    message = alias_error(alias)
    assert message is not None
    assert f"contains {named}," in message


def test_alias_error_control_character():
    message = alias_error("two\tparts")
    assert message is not None
    assert "control character" in message


@pytest.mark.parametrize(
    "link", ["https://example.org/report", "https://doi.org/10.5281/zenodo.123?x=1&y=2"]
)
def test_model_report_link_error_accepts_normal_links(link):
    assert model_report_link_error(link) is None


@pytest.mark.parametrize(
    "link",
    [
        'https://example.org/" onmouseover="alert(1)',
        "https://example.org/<script>",
        "https://example.org/a b",
    ],
)
def test_model_report_link_error_rejects_unsafe_links(link):
    assert model_report_link_error(link) is not None


def _submit(**overrides):
    args = dict(
        username="alice",
        user_alias="",
        anon_checkbox=False,
        participant_name="Alice",
        discord_username="",
        email="",
        affiliation="",
        model_tag="",
        proprietary_data_checkbox=False,
        open_code_checkbox=False,
        track_select="Regression Prediction",
        file_input=None,
    )
    args.update(overrides)
    return submission.submit_predictions(**args)


def test_submit_rejects_bad_alias_with_reason_before_upload(monkeypatch):
    """The participant sees why their alias was refused, and nothing is uploaded,
    so the submission cooldown isn't used up."""
    monkeypatch.setattr(submission, "validate_hf_username", lambda name: True)
    monkeypatch.setattr(
        submission,
        "upload_submission",
        lambda *a, **k: pytest.fail("upload must not be attempted"),
    )
    result = _submit(user_alias="<b>me</b>", anon_checkbox=True)
    assert result["visible"] is True
    assert result["value"] == alias_error("<b>me</b>")


def test_submit_rejects_unsafe_model_report_link(monkeypatch):
    monkeypatch.setattr(submission, "validate_hf_username", lambda name: True)
    result = _submit(model_tag='https://example.org/" onmouseover="x')
    assert "Method Report Link" in result["value"]


# --- Test data -----------------------------------------------------------------


def _identifiers(n: int = ACTIVITY_DATASET_SIZE) -> dict[str, list[str]]:
    return {
        "SMILES": ["C"] * n,
        "Molecule_Name": [f"MOL-{i:04d}" for i in range(n)],
    }


def _regression_df(n: int = ACTIVITY_DATASET_SIZE) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {**_identifiers(n), **{c: rng.normal(5, 1, n) for c in REGRESSION_ENDPOINTS}}
    )


def _classification_df(n: int = ACTIVITY_DATASET_SIZE) -> pd.DataFrame:
    pattern = [True, False] * (n // 2) + [True] * (n % 2)
    return pd.DataFrame(
        {**_identifiers(n), **{c: pattern for c in CLASSIFICATION_ENDPOINTS}}
    )


def _write(df: pd.DataFrame, path) -> str:
    if str(path).endswith(".parquet"):
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
    return str(path)


def _structure_zip(path, n_files: int = STRUCTURE_DATASET_SIZE) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n_files):
            zf.writestr(f"STRUCT-{i:04d}.pdb", "END\n")
    return str(path)


@pytest.fixture
def known_user(monkeypatch):
    """Every Hugging Face username "exists", without a network call."""
    monkeypatch.setattr(submission, "validate_hf_username", lambda name: True)


@pytest.fixture
def no_upload(monkeypatch):
    """Fail the test if anything is uploaded."""
    monkeypatch.setattr(
        submission,
        "upload_submission",
        lambda *a, **k: pytest.fail("upload must not be attempted"),
    )


@pytest.fixture
def uploads(monkeypatch):
    """Capture ``upload_submission`` calls instead of uploading."""
    calls: list[tuple[Submission, object]] = []

    def _capture(sub, file_path):
        calls.append((sub, file_path))
        return sub

    monkeypatch.setattr(submission, "upload_submission", _capture)
    return calls


@pytest.fixture
def no_cooldown(monkeypatch):
    """No previous submission, so the cooldown never blocks."""
    monkeypatch.setattr(submission, "_fetch_last_submission_date", lambda *a: None)


# --- _read_tabular_submission --------------------------------------------------


@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".CSV"])
def test_read_tabular_submission_reads_parquet_and_csv(tmp_path, suffix):
    path = _write(_regression_df(), tmp_path / f"preds{suffix}")
    df, error = submission._read_tabular_submission(
        submission.Path(path), REQUIRED_REGRESSION_COLUMNS, "Regression"
    )
    assert error is None
    assert len(df) == ACTIVITY_DATASET_SIZE


def test_read_tabular_submission_rejects_other_file_types(tmp_path):
    path = tmp_path / "preds.xlsx"
    path.write_text("x")
    df, error = submission._read_tabular_submission(
        path, REQUIRED_REGRESSION_COLUMNS, "Regression"
    )
    assert df is None
    assert error == "Error: Regression submissions must be a .parquet or .csv file."


def test_read_tabular_submission_reports_unreadable_parquet(tmp_path):
    path = tmp_path / "preds.parquet"
    path.write_text("not parquet")
    df, error = submission._read_tabular_submission(
        path, REQUIRED_REGRESSION_COLUMNS, "Regression"
    )
    assert df is None
    assert error.startswith("Error: Could not read parquet file")


def test_read_tabular_submission_checks_row_count(tmp_path):
    path = _write(_regression_df(n=10), tmp_path / "preds.csv")
    _, error = submission._read_tabular_submission(
        submission.Path(path), REQUIRED_REGRESSION_COLUMNS, "Regression"
    )
    assert error == f"Error: Expected {ACTIVITY_DATASET_SIZE} rows, got 10."


def test_read_tabular_submission_names_missing_columns(tmp_path):
    path = _write(
        _regression_df().drop(columns=[REGRESSION_ENDPOINTS[0]]), tmp_path / "p.csv"
    )
    _, error = submission._read_tabular_submission(
        submission.Path(path), REQUIRED_REGRESSION_COLUMNS, "Regression"
    )
    assert error == f"Error: Missing required columns: { {REGRESSION_ENDPOINTS[0]} }"


# --- _format_submission_time_message --------------------------------------------


def test_format_submission_time_message_shows_time_left(utc_now):
    last = utc_now - timedelta(hours=HOURS_BETWEEN_SUBMISSIONS) + timedelta(minutes=90)
    message = submission._format_submission_time_message(last, track="regression")
    assert message.startswith("Error: You submitted a regression prediction on ")
    assert last.strftime("%Y-%m-%d %H:%M:%S (UTC)") in message
    # About 1h30m left; allow for the seconds that pass while the test runs.
    assert "Please wait 01:29:" in message or "Please wait 01:30:00" in message


def test_format_submission_time_message_never_negative(utc_now):
    last = utc_now - timedelta(hours=HOURS_BETWEEN_SUBMISSIONS + 5)
    message = submission._format_submission_time_message(last, track="structure")
    assert "Please wait 00:00:00" in message


# --- _fetch_last_submission_date -----------------------------------------------


def test_fetch_last_submission_date_without_bucket_is_none(monkeypatch):
    monkeypatch.setattr(submission, "S3_BUCKET", "")
    monkeypatch.setattr(
        submission.boto3, "client", lambda *a, **k: pytest.fail("no S3 call expected")
    )
    assert submission._fetch_last_submission_date("regression", "alice") is None


def test_fetch_last_submission_date_no_submissions(fake_s3):
    assert submission._fetch_last_submission_date("regression", "alice") is None
    assert fake_s3.list_prefixes == ["submissions/regression/alice/"]


def test_fetch_last_submission_date_latest_metadata_only(fake_s3, utc_now):
    old, new = utc_now - timedelta(days=2), utc_now - timedelta(hours=1)
    fake_s3.add_metadata("submissions/regression/alice/a/metadata.json", old)
    fake_s3.add_metadata("submissions/regression/alice/b/metadata.json", new)
    # A newer prediction file must not count: only metadata.json marks a submission.
    fake_s3.add_metadata("submissions/regression/alice/c/predictions.csv", utc_now)
    # Another user's and another track's submissions are not this user's.
    fake_s3.add_metadata("submissions/regression/alice2/d/metadata.json", utc_now)
    fake_s3.add_metadata("submissions/structure/alice/e/metadata.json", utc_now)
    assert submission._fetch_last_submission_date("regression", "alice") == new


def test_fetch_last_submission_date_s3_error_is_none(fake_s3):
    # Current behaviour: an S3 failure looks like "no previous submission", so the
    # cooldown is skipped (fails open). #7 tracks making this an explicit choice.
    fake_s3.fail_with = RuntimeError("S3 down")
    assert submission._fetch_last_submission_date("regression", "alice") is None


# --- upload_submission --------------------------------------------------------


def _sub(track: str = "Regression Prediction", filename: str = "p.csv") -> Submission:
    return Submission(username="Alice", track=track, filename=filename)


def test_upload_submission_without_bucket_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(submission, "S3_BUCKET", "")
    monkeypatch.setattr(
        submission.boto3, "client", lambda *a, **k: pytest.fail("no S3 call expected")
    )
    sub = _sub()
    assert submission.upload_submission(sub, tmp_path / "p.csv") is sub


@pytest.mark.parametrize(
    ("track", "filename", "canonical"),
    [
        ("Regression Prediction", "my preds.csv", "predictions.csv"),
        ("Classification Prediction", "x.parquet", "predictions.parquet"),
        ("Structure Prediction", "poses.zip", "structures.zip"),
    ],
)
def test_upload_submission_keys(fake_s3, tmp_path, track, filename, canonical):
    sub = _sub(track, filename)
    uploaded = submission.upload_submission(sub, tmp_path / filename)

    file_key = f"{sub.s3_prefix}/{canonical}"
    assert uploaded.s3_key == file_key
    assert fake_s3.uploads == [(str(tmp_path / filename), "test-bucket", file_key)]
    [put] = fake_s3.puts
    assert put["Key"] == f"{sub.s3_prefix}/metadata.json"
    assert put["ContentType"] == "application/json"
    stored = Submission.model_validate_json(put["Body"])
    assert stored.submission_id == sub.submission_id
    assert stored.s3_key == file_key


def test_upload_submission_s3_error_is_swallowed(fake_s3, tmp_path):
    # Current behaviour: the error is logged, not raised, and submit_predictions
    # still reports success. #7 tracks surfacing this to the participant.
    fake_s3.fail_with = RuntimeError("S3 down")
    uploaded = submission.upload_submission(_sub(), tmp_path / "p.csv")
    assert uploaded.s3_key is not None


# --- submit_predictions: identity and form fields -------------------------------


@pytest.mark.parametrize("username", ["", "   "])
def test_submit_requires_username(username, no_upload):
    result = _submit(username=username)
    assert result["value"] == "Error: Hugging Face username is required."


def test_submit_rejects_unknown_username(monkeypatch, no_upload):
    monkeypatch.setattr(submission, "validate_hf_username", lambda name: False)
    result = _submit(username=" ghost ")
    assert "'ghost' could not be found" in result["value"]


def test_submit_blocks_banned_username_case_insensitively(
    monkeypatch, known_user, no_upload
):
    monkeypatch.setitem(submission.BANNED_USERNAMES, "mallory", "Multiple accounts")
    result = _submit(username="Mallory")
    assert "has been blocked" in result["value"]
    assert "Multiple accounts" in result["value"]


def test_submit_anonymous_needs_alias(known_user, no_upload):
    result = _submit(anon_checkbox=True, user_alias="  ")
    assert result["value"].startswith("Error: An alias is required")


def test_submit_rejects_invalid_email(known_user, no_upload):
    result = _submit(email="not-an-email")
    assert result["value"] == "Error: Please enter a valid email address."


def test_submit_open_code_needs_reachable_report(monkeypatch, known_user, no_upload):
    monkeypatch.setattr(
        submission, "validate_model_details", lambda tag: "Invalid link"
    )
    result = _submit(open_code_checkbox=True, model_tag="https://example.org/gone")
    assert result["value"].startswith("Error: Could not open the Method Report Link")


def test_submit_requires_track(known_user, no_upload):
    result = _submit(track_select=None)
    assert result["value"] == "Error: Please select a track."


def test_submit_requires_file(known_user, no_upload):
    result = _submit(file_input=None)
    assert result["value"] == "Error: Please upload a submission file."


# --- submit_predictions: file contents ------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.nan, "contains NaN values"),
        (np.inf, "contains infinite values"),
    ],
)
def test_submit_regression_rejects_bad_values(
    tmp_path, known_user, no_upload, no_cooldown, value, expected
):
    df = _regression_df()
    df.loc[3, REGRESSION_ENDPOINTS[1]] = value
    result = _submit(file_input=_write(df, tmp_path / "p.parquet"))
    assert result["value"] == f"Error: {REGRESSION_ENDPOINTS[1]} column {expected}."


def test_submit_regression_rejects_constant_predictions(
    tmp_path, known_user, no_upload, no_cooldown
):
    df = _regression_df()
    df[REGRESSION_ENDPOINTS[0]] = 5.0
    result = _submit(file_input=_write(df, tmp_path / "p.csv"))
    assert "constant or near-constant" in result["value"]


def test_submit_classification_rejects_non_binary(
    tmp_path, known_user, no_upload, no_cooldown
):
    df = _classification_df()
    df[CLASSIFICATION_ENDPOINTS[0]] = df[CLASSIFICATION_ENDPOINTS[0]].astype(int)
    df.loc[0, CLASSIFICATION_ENDPOINTS[0]] = 2
    result = _submit(
        track_select="Classification Prediction",
        file_input=_write(df, tmp_path / "p.csv"),
    )
    assert "non-binary values" in result["value"]


def test_submit_classification_rejects_single_class(
    tmp_path, known_user, no_upload, no_cooldown
):
    df = _classification_df()
    df[CLASSIFICATION_ENDPOINTS[1]] = False
    result = _submit(
        track_select="Classification Prediction",
        file_input=_write(df, tmp_path / "p.csv"),
    )
    assert "all one class" in result["value"]


def test_submit_structure_must_be_zip(tmp_path, known_user, no_upload):
    path = tmp_path / "poses.tar"
    path.write_text("x")
    result = _submit(track_select="Structure Prediction", file_input=str(path))
    assert result["value"] == "Error: Structure submissions must be a .zip file."


def test_submit_structure_unreadable_zip(tmp_path, known_user, no_upload):
    path = tmp_path / "poses.zip"
    path.write_text("not a zip")
    result = _submit(track_select="Structure Prediction", file_input=str(path))
    assert result["value"].startswith("Error: Could not read zip file")


def test_submit_structure_wrong_file_count(tmp_path, known_user, no_upload):
    path = _structure_zip(tmp_path / "poses.zip", n_files=3)
    result = _submit(track_select="Structure Prediction", file_input=path)
    assert (
        result["value"]
        == f"Error: Expected {STRUCTURE_DATASET_SIZE} files in zip, got 3."
    )


# --- submit_predictions: cooldown and success -----------------------------------


def _valid_file(track: str, tmp_path) -> str:
    if track == "Regression Prediction":
        return _write(_regression_df(), tmp_path / "p.csv")
    if track == "Classification Prediction":
        return _write(_classification_df(), tmp_path / "p.csv")
    return _structure_zip(tmp_path / "poses.zip")


TRACKS = [
    ("Regression Prediction", "regression"),
    ("Classification Prediction", "classification"),
    ("Structure Prediction", "structure"),
]


@pytest.mark.parametrize(("track", "slug"), TRACKS)
def test_submit_cooldown_blocks_recent_submitter(
    monkeypatch, tmp_path, utc_now, known_user, no_upload, track, slug
):
    seen = []

    def _last(track_slug, user_id):
        seen.append((track_slug, user_id))
        return utc_now - timedelta(hours=1)

    monkeypatch.setattr(submission, "_fetch_last_submission_date", _last)
    result = _submit(
        username="Org/Alice",
        track_select=track,
        file_input=_valid_file(track, tmp_path),
    )
    assert result["value"].startswith(f"Error: You submitted a {slug} prediction")
    # The cooldown is looked up under the S3-safe, lower-cased username.
    assert seen == [(slug, "org_alice")]


@pytest.mark.parametrize(("track", "slug"), TRACKS)
def test_submit_success_after_cooldown(
    monkeypatch, tmp_path, utc_now, known_user, uploads, track, slug
):
    last = utc_now - timedelta(hours=HOURS_BETWEEN_SUBMISSIONS, minutes=1)
    monkeypatch.setattr(submission, "_fetch_last_submission_date", lambda *a: last)
    path = _valid_file(track, tmp_path)
    result = _submit(
        username=" alice ",
        user_alias=" Wizard ",
        anon_checkbox=True,
        email=" a@example.org ",
        track_select=track,
        file_input=path,
    )

    assert result == {
        "__type__": "update",
        "value": f"Submission received from 'Wizard' for the {track} track. Your "
        "predictions are being processed and should appear on the leaderboard "
        "within 2 hours.",
        "visible": True,
    }
    [(sub, file_path)] = uploads
    assert (sub.username, sub.user_alias, sub.email) == (
        "alice",
        "Wizard",
        "a@example.org",
    )
    assert sub.anonymous is True
    assert sub.track == track
    assert sub.model_report_link == "Not submitted"
    assert str(file_path) == path


def test_submit_without_open_code_keeps_report_link_unchecked(
    monkeypatch, tmp_path, known_user, uploads, no_cooldown
):
    monkeypatch.setattr(
        submission,
        "validate_model_details",
        lambda tag: pytest.fail("link is only checked for open-code submissions"),
    )
    _submit(
        model_tag=" https://example.org/report ",
        file_input=_valid_file("Regression Prediction", tmp_path),
    )
    [(sub, _)] = uploads
    assert sub.model_report_link == "https://example.org/report"
