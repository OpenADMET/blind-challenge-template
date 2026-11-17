"""Shared setup for the HF Space tests.

The Space's modules import each other as top-level modules (``from config import
...``), the way they run inside the Space container, so ``hf_space/`` has to be on
``sys.path`` rather than imported as a package.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

HF_SPACE_DIR = Path(__file__).resolve().parents[2] / "hf_space"
if str(HF_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(HF_SPACE_DIR))


class FakeS3Client:
    """Records the calls the Space makes to S3 and answers ``list_objects_v2``.

    Attributes
    ----------
    objects : list[dict]
        What ``list_objects_v2`` returns as ``Contents`` (``Key`` and
        ``LastModified`` per object).
    uploads : list[tuple[str, str, str]]
        ``(local_path, bucket, key)`` for every ``upload_file`` call.
    puts : list[dict]
        The keyword arguments of every ``put_object`` call.
    list_prefixes : list[str]
        The ``Prefix`` of every ``list_objects_v2`` call.
    fail_with : Exception | None
        If set, every call raises it.

    """

    def __init__(self) -> None:
        self.objects: list[dict] = []
        self.uploads: list[tuple[str, str, str]] = []
        self.puts: list[dict] = []
        self.list_prefixes: list[str] = []
        self.fail_with: Exception | None = None

    def _maybe_fail(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def list_objects_v2(self, Bucket: str, Prefix: str) -> dict:  # noqa: N803
        self._maybe_fail()
        self.list_prefixes.append(Prefix)
        matching = [o for o in self.objects if o["Key"].startswith(Prefix)]
        response: dict = {"IsTruncated": False}
        if matching:
            response["Contents"] = matching
        return response

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self._maybe_fail()
        self.uploads.append((filename, bucket, key))

    def put_object(self, **kwargs) -> None:
        self._maybe_fail()
        self.puts.append(kwargs)

    def add_metadata(self, key: str, last_modified: datetime) -> None:
        """Add an object to the listing."""
        self.objects.append({"Key": key, "LastModified": last_modified})


@pytest.fixture
def fake_s3(monkeypatch):
    """Point ``submission``'s boto3 client and bucket at a ``FakeS3Client``."""
    import submission

    client = FakeS3Client()
    monkeypatch.setattr(submission, "S3_BUCKET", "test-bucket")
    monkeypatch.setattr(submission.boto3, "client", lambda *a, **k: client)
    return client


@pytest.fixture
def utc_now():
    """The current time, timezone-aware in UTC."""
    return datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def fake_local_hf_login(monkeypatch):
    """Let pages with a Hugging Face login button build without an HF token.

    Outside a Space, Gradio fakes the login with the machine's own HF account and
    fails if there's no token, as on CI. The tests never log in through it.
    """
    import gradio.oauth

    monkeypatch.setattr(
        gradio.oauth,
        "_get_mocked_oauth_info",
        lambda: {
            "access_token": "test-token",
            "token_type": "bearer",
            "expires_in": 3600,
            "id_token": "test",
            "scope": "openid profile",
            "expires_at": 0,
            "userinfo": {
                "sub": "test-sub",
                "name": "Test User",
                "preferred_username": "test-user",
                "profile": "https://huggingface.co/test-user",
                "picture": "",
            },
        },
    )
