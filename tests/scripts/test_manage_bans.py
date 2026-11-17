"""Tests for ``scripts/manage_bans.py``."""

import hashlib
import io

import manage_bans
import pytest
from botocore.exceptions import ClientError

import bans

HEADER = ",".join(bans.COLUMNS) + "\n"


class FakeS3:
    """Holds the ban list file and honours ``IfMatch`` like S3 does."""

    def __init__(self, text: str = HEADER) -> None:
        self.text = text
        self.writes: list[str] = []

    @property
    def etag(self) -> str:
        return '"' + hashlib.md5(self.text.encode()).hexdigest() + '"'

    def get_object(self, Bucket, Key):  # noqa: N803
        assert (Bucket, Key) == ("test-bucket", bans.BAN_LIST_KEY)
        return {"Body": io.BytesIO(self.text.encode()), "ETag": self.etag}

    def put_object(self, Bucket, Key, Body, ContentType, IfMatch):  # noqa: N803
        if IfMatch != self.etag:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "changed"}},
                "PutObject",
            )
        self.text = Body.decode()
        self.writes.append(self.text)


def _run(s3, *args):
    return manage_bans.main(["--bucket", "test-bucket", *args], s3=s3)


ADD = ["add", "Mallory", "--reason-code", "multiple_accounts", "--note", "evidence"]


def test_add_is_a_dry_run_by_default(capsys):
    s3 = FakeS3()
    assert _run(s3, *ADD, "--by", "org") == 0
    assert s3.writes == []
    assert "Dry run" in capsys.readouterr().out


def test_add_with_execute_writes_a_valid_file():
    s3 = FakeS3()
    assert _run(s3, *ADD, "--by", "org", "--hf-user-id", "id-1", "--execute") == 0
    df = bans.parse_ban_list(s3.text)
    assert list(df["username"]) == ["mallory"]
    assert df.loc[0, "hf_user_id"] == "id-1"


def test_reinstate_with_execute():
    s3 = FakeS3()
    _run(s3, *ADD, "--by", "org", "--execute")
    assert _run(s3, "reinstate", "mallory", "--note", "apologised", "--execute") == 0
    assert bans.find_ban(bans.parse_ban_list(s3.text), "mallory") is None


def test_errors_are_reported_not_written(capsys):
    s3 = FakeS3()
    assert _run(s3, "reinstate", "nobody", "--note", "x", "--execute") == 1
    assert "no active ban" in capsys.readouterr().err
    assert s3.writes == []


def test_a_concurrent_edit_is_not_overwritten(monkeypatch, capsys):
    s3 = FakeS3()
    real_read = manage_bans.read_ban_list

    def _read_then_someone_else_writes(client, bucket):
        df, etag = real_read(client, bucket)
        client.text = HEADER + "eve,,rule_violation,,,,,\n"
        return df, etag

    monkeypatch.setattr(manage_bans, "read_ban_list", _read_then_someone_else_writes)
    assert _run(s3, *ADD, "--by", "org", "--execute") == 1
    assert "changed while you were editing" in capsys.readouterr().err
    assert "eve" in s3.text and "mallory" not in s3.text


def test_list_shows_active_bans_unless_all(capsys):
    s3 = FakeS3(
        HEADER
        + "mallory,,multiple_accounts,,,,,\n"
        + "eve,,rule_violation,,,,,2026-11-02T00:00:00Z\n"
    )
    _run(s3, "list")
    out = capsys.readouterr().out
    assert "mallory" in out and "eve" not in out
    _run(s3, "list", "--all")
    assert "eve" in capsys.readouterr().out


def test_bucket_is_required(monkeypatch, capsys):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    assert manage_bans.main(["list"], s3=FakeS3()) == 2
    assert "S3_BUCKET" in capsys.readouterr().err


def test_unknown_reason_codes_are_refused_by_the_parser():
    with pytest.raises(SystemExit):
        manage_bans.build_parser().parse_args(
            ["add", "x", "--reason-code", "nope", "--note", "n", "--by", "o"]
        )
