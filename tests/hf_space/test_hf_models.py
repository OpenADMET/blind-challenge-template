"""Tests for the HF Space submission model."""

import time
import uuid

from models import Submission, new_submission_id


def _submission(**overrides) -> Submission:
    fields = {
        "username": "alice",
        "track": "Regression Prediction",
        "filename": "predictions.csv",
    }
    return Submission(**{**fields, **overrides})


def test_new_submission_id_is_a_uuid7_string():
    sid = new_submission_id()
    parsed = uuid.UUID(sid)
    assert parsed.version == 7
    assert str(parsed) == sid


def test_submission_ids_sort_in_creation_order():
    """IDs created later sort later, so S3 listings come out in submission order."""
    ids = []
    for _ in range(5):
        ids.append(new_submission_id())
        time.sleep(0.002)  # cross a millisecond boundary each time
    assert ids == sorted(ids)


def test_submission_gets_a_fresh_uuid7_by_default():
    a = _submission()
    b = _submission()
    assert a.submission_id != b.submission_id
    assert uuid.UUID(a.submission_id).version == 7


def test_existing_uuid4_ids_are_still_accepted():
    """Submissions recorded before the switch keep their UUIDv4 IDs."""
    old = str(uuid.uuid4())
    assert _submission(submission_id=old).submission_id == old
