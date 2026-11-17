"""Tests for the HF Space submission-form validation."""

import pytest
import submission
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
