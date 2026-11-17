"""Tests for ``scripts/quarantine_submissions.py`` (with a fake S3)."""

import pandas as pd
import pytest
import quarantine_submissions as quarantine
from botocore.exceptions import ClientError

SID = "0190a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b"
LIVE = f"manifests/regression/alice_{SID}.parquet"
QUARANTINED = f"quarantine/{LIVE}"


class FakeS3:
    def __init__(self, keys=()):
        self.keys = set(keys)

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.keys:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def copy_object(self, Bucket, Key, CopySource):  # noqa: N803
        self.keys.add(Key)

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.keys.discard(Key)


def _targets(tmp_path, rows) -> str:
    path = tmp_path / "targets.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _run(tmp_path, s3, *args, rows=None):
    rows = rows or [
        {"submission_id": SID, "safe_username": "Alice", "track": "regression"}
    ]
    log = tmp_path / "log.csv"
    code = quarantine.main(
        [_targets(tmp_path, rows), "--bucket", "b", "--log", str(log), *args], s3=s3
    )
    return code, pd.read_csv(log)


@pytest.mark.parametrize(
    "label", ["regression", "Regression Prediction", " REGRESSION "]
)
def test_normalise_track(label):
    assert quarantine.normalise_track(label) == "regression"


def test_unknown_track_is_refused():
    with pytest.raises(ValueError, match="unknown track"):
        quarantine.normalise_track("docking")


def test_load_targets_normalises_and_dedupes(tmp_path):
    path = _targets(
        tmp_path,
        [
            {
                "submission_id": SID,
                "safe_username": "Org/Alice",
                "track": "Regression Prediction",
                "extra": 1,
            },
            {
                "submission_id": SID,
                "safe_username": "org_alice",
                "track": "regression",
                "extra": 2,
            },
            {
                "submission_id": "",
                "safe_username": "bob",
                "track": "regression",
                "extra": 3,
            },
        ],
    )
    df = quarantine.load_targets(path)
    assert df.to_dict("records") == [
        {"submission_id": SID, "safe_username": "org_alice", "track": "regression"}
    ]


def test_manifest_key():
    assert quarantine.manifest_key("regression", "alice", SID) == LIVE


def test_dry_run_moves_nothing(tmp_path):
    s3 = FakeS3([LIVE])
    code, log = _run(tmp_path, s3)
    assert code == 0
    assert s3.keys == {LIVE}
    assert list(log["status"]) == ["dry-run"]
    assert list(log["action"]) == ["quarantine"]


def test_execute_moves_the_manifest_and_back(tmp_path):
    s3 = FakeS3([LIVE])
    code, log = _run(tmp_path, s3, "--execute")
    assert code == 0
    assert s3.keys == {QUARANTINED}
    assert (log.loc[0, "source_key"], log.loc[0, "dest_key"]) == (LIVE, QUARANTINED)
    code, log = _run(tmp_path, s3, "--execute", "--unquarantine")
    assert code == 0
    assert s3.keys == {LIVE}
    assert list(log["action"]) == ["unquarantine"]


def test_missing_manifest_is_reported(tmp_path):
    code, log = _run(tmp_path, FakeS3(), "--execute")
    assert code == 1
    assert list(log["status"]) == ["not-found"]


def test_existing_destination_is_kept_unless_overwrite(tmp_path):
    s3 = FakeS3([LIVE, QUARANTINED])
    code, log = _run(tmp_path, s3, "--execute")
    assert code == 1
    assert list(log["status"]) == ["destination-exists"]
    assert LIVE in s3.keys
    code, log = _run(tmp_path, s3, "--execute", "--overwrite")
    assert code == 0
    assert s3.keys == {QUARANTINED}


def test_bucket_is_required(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    path = _targets(
        tmp_path, [{"submission_id": SID, "safe_username": "a", "track": "regression"}]
    )
    assert quarantine.main([path], s3=FakeS3()) == 2
