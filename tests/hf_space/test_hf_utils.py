"""Tests for the HF Space helpers in ``hf_space/utils.py``."""

import socket

import pytest
import requests
import utils


class _Response:
    def __init__(self, status_code: int, location: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {"Location": location} if location else {}


@pytest.fixture
def no_sleep(monkeypatch):
    """Record the waits instead of sleeping."""
    waits: list[float] = []
    monkeypatch.setattr(utils.time, "sleep", waits.append)
    return waits


@pytest.fixture
def public_dns(monkeypatch):
    """Every hostname resolves to a public address, except ``internal.test``."""

    def _resolve(host, port):
        ip = "10.0.0.5" if host == "internal.test" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(utils.socket, "getaddrinfo", _resolve)


def _serve(monkeypatch, responses: dict[str, list[_Response]]) -> list[str]:
    """Answer ``requests.get`` from ``responses`` (popped in order per URL)."""
    requested: list[str] = []

    def _get(url, timeout, allow_redirects):
        assert allow_redirects is False  # every hop must be checked by hand
        requested.append(url)
        return responses[url].pop(0)

    monkeypatch.setattr(utils.requests, "get", _get)
    return requested


# --- _safeify_username ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("Alice", "alice"),
        ("  Alice  ", "alice"),
        ("Org/Alice", "org_alice"),
        ("first last", "first_last"),
    ],
)
def test_safeify_username(raw, safe):
    assert utils._safeify_username(raw) == safe


# --- _is_safe_public_url -------------------------------------------------------


@pytest.mark.parametrize(
    ("ip", "safe"),
    [
        ("93.184.216.34", True),
        ("10.0.0.5", False),  # private
        ("127.0.0.1", False),  # loopback
        ("169.254.169.254", False),  # link-local (cloud metadata endpoint)
        ("::1", False),
        ("0.0.0.0", False),
    ],
)
def test_is_safe_public_url_by_resolved_address(monkeypatch, ip, safe):
    monkeypatch.setattr(
        utils.socket, "getaddrinfo", lambda host, port: [(0, 0, 0, "", (ip, 0))]
    )
    assert utils._is_safe_public_url("https://example.org/report") is safe


def test_is_safe_public_url_unresolvable_or_hostless(monkeypatch):
    def _fail(host, port):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(utils.socket, "getaddrinfo", _fail)
    assert utils._is_safe_public_url("https://nowhere.invalid") is False
    assert utils._is_safe_public_url("not a url") is False


# --- check_page_exists --------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "exists"), [(200, True), (404, False), (500, False)]
)
def test_check_page_exists_status(monkeypatch, public_dns, no_sleep, status, exists):
    _serve(monkeypatch, {"https://example.org/r": [_Response(status)]})
    assert utils.check_page_exists("https://example.org/r") is exists


def test_check_page_exists_adds_https(monkeypatch, public_dns, no_sleep):
    requested = _serve(monkeypatch, {"https://example.org/r": [_Response(200)]})
    assert utils.check_page_exists("  example.org/r ") is True
    assert requested == ["https://example.org/r"]


def test_check_page_exists_refuses_internal_host(monkeypatch, public_dns, no_sleep):
    requested = _serve(monkeypatch, {})
    assert utils.check_page_exists("https://internal.test/admin") is False
    assert requested == []


def test_check_page_exists_follows_safe_redirects(monkeypatch, public_dns, no_sleep):
    requested = _serve(
        monkeypatch,
        {
            "https://example.org/a": [_Response(301, "/b")],
            "https://example.org/b": [_Response(200)],
        },
    )
    assert utils.check_page_exists("https://example.org/a") is True
    assert requested == ["https://example.org/a", "https://example.org/b"]


def test_check_page_exists_refuses_redirect_to_internal_host(
    monkeypatch, public_dns, no_sleep
):
    """A public page can't bounce the Space to an internal address (SSRF)."""
    requested = _serve(
        monkeypatch,
        {"https://example.org/a": [_Response(302, "https://internal.test/admin")]},
    )
    assert utils.check_page_exists("https://example.org/a") is False
    assert requested == ["https://example.org/a"]


def test_check_page_exists_gives_up_on_redirect_loops(
    monkeypatch, public_dns, no_sleep
):
    _serve(
        monkeypatch,
        {"https://example.org/a": [_Response(302, "/a")] * (utils._MAX_REDIRECTS + 1)},
    )
    assert utils.check_page_exists("https://example.org/a") is False


def test_check_page_exists_retries_rate_limits(monkeypatch, public_dns, no_sleep):
    _serve(
        monkeypatch,
        {"https://example.org/r": [_Response(429), _Response(429), _Response(200)]},
    )
    assert utils.check_page_exists("https://example.org/r", delay=0) is True
    # Exponential back-off between attempts (the zero delays are the per-request sleep).
    assert [w for w in no_sleep if w] == [5, 10]


def test_check_page_exists_stops_after_max_retries(monkeypatch, public_dns, no_sleep):
    _serve(monkeypatch, {"https://example.org/r": [_Response(429)] * 3})
    assert utils.check_page_exists("https://example.org/r", max_retries=2) is False


def test_check_page_exists_network_error(monkeypatch, public_dns, no_sleep):
    def _get(url, timeout, allow_redirects):
        raise requests.exceptions.ConnectionError("unreachable")

    monkeypatch.setattr(utils.requests, "get", _get)
    assert utils.check_page_exists("https://example.org/r") is False


# --- validate_hf_username / validate_model_details ------------------------------


@pytest.mark.parametrize(
    ("username", "url"),
    [
        ("alice", "https://huggingface.co/alice"),
        (" alice ", "https://huggingface.co/alice"),
        ("https://huggingface.co/alice", "https://huggingface.co/alice"),
    ],
)
def test_validate_hf_username_checks_profile_page(monkeypatch, username, url):
    calls = []

    def _check(u, **kwargs):
        calls.append((u, kwargs["restrict_to_public"]))
        return True

    monkeypatch.setattr(utils, "check_page_exists", _check)
    assert utils.validate_hf_username(username) is True
    # huggingface.co is a fixed, trusted host, so the SSRF guard is off for it.
    assert calls == [(url, False)]


@pytest.mark.parametrize(
    ("tag", "exists", "expected"),
    [
        (None, True, "Not submitted"),
        ("  ", True, "Not submitted"),
        ("http://example.org/r", True, "Invalid link"),
        ("https://example.org/r", False, "Invalid link"),
        (" https://example.org/r ", True, "https://example.org/r"),
    ],
)
def test_validate_model_details(monkeypatch, tag, exists, expected):
    monkeypatch.setattr(utils, "check_page_exists", lambda url, delay: exists)
    assert utils.validate_model_details(tag) == expected
