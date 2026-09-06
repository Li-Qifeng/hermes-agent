"""Short-window throttle cooldown: a 429 whose body names a rolling per-minute /
per-token window (SenseNova ``tpm/rpm exhausted``, ``Allocated quota exceeded``)
reopens in ~12 seconds, so the default 1-hour bench is ~300x the real recovery
time. In a multi-key pool that collapses rotation after each key hits one 429
("no available entries (all exhausted or empty)"). These tests pin the shortened
cooldown — and, critically, that a genuine billing verdict still keeps the full
bench regardless of the wording.

Regression for the NAS 429-cooldown mismatch (see snc-429-cooldown-report).
"""

from __future__ import annotations

import json
import time


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _entry(
    error_code: int,
    *,
    age_seconds: float,
    cred_id: str = "cred-1",
    priority: int = 0,
    failure_reason: str | None = None,
    error_message: str | None = None,
) -> dict:
    entry = {
        "id": cred_id,
        "label": cred_id,
        "auth_type": "api_key",
        "priority": priority,
        "source": "manual",
        "access_token": f"sk-test-{cred_id}",
        "base_url": "https://token.sensenova.cn/v1",
        "last_status": "exhausted",
        "last_status_at": time.time() - age_seconds,
        "last_error_code": error_code,
    }
    if failure_reason is not None:
        entry["failure_reason"] = failure_reason
    if error_message is not None:
        entry["last_error_message"] = error_message
    return entry


def _load(tmp_path, monkeypatch, entries: list[dict]):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, {"version": 1, "credential_pool": {"st": entries}})
    from agent.credential_pool import load_pool

    return load_pool("st")


# --- pure TTL sizing (no pool I/O) -----------------------------------------

def test_short_window_phrases_compress_bench_to_60s():
    from agent.credential_pool import _exhausted_ttl

    for msg in (
        "inference tpm exhausted, please try again later",
        "rpm exhausted",
        "Allocated quota exceeded, please increase your quota limit.",
    ):
        # sole_credential=False: this is the multi-key collapse the report describes.
        assert _exhausted_ttl(429, error_message=msg) == 60, msg


def test_billing_verdict_keeps_full_bench_despite_window_wording():
    from agent.credential_pool import _exhausted_ttl

    # A classified billing failure must NOT be shortened even if the body happens
    # to contain a short-window token — retrying a spent account every 60s just
    # re-fails forever.
    assert _exhausted_ttl(429, failure_reason="billing",
                          error_message="Allocated quota exceeded") == 3600
    # 402 is billing by definition.
    assert _exhausted_ttl(402, error_message="tpm exhausted") == 3600


def test_plain_429_without_window_wording_keeps_full_bench():
    from agent.credential_pool import _exhausted_ttl

    assert _exhausted_ttl(429, error_message="Too many requests") == 3600
    assert _exhausted_ttl(429, error_message=None) == 3600
    assert _exhausted_ttl(429, error_message="") == 3600


def test_window_match_is_case_insensitive():
    from agent.credential_pool import _exhausted_ttl

    assert _exhausted_ttl(429, error_message="Inference TPM EXHAUSTED") == 60


# --- end-to-end pool rotation (the actual failure mode) --------------------

def test_two_key_pool_recovers_after_short_window_429(tmp_path, monkeypatch):
    """Both keys hit a SenseNova TPM 429 ~90s ago. Under the old 1h bench the
    pool is empty ('no available entries'); with the short-window fix both keys
    are back in rotation after 60s."""
    entries = [
        _entry(429, age_seconds=90, cred_id="k1",
               error_message="inference tpm exhausted, please try again later"),
        _entry(429, age_seconds=90, cred_id="k2",
               error_message="Allocated quota exceeded, please increase your quota limit."),
    ]
    pool = _load(tmp_path, monkeypatch, entries)
    assert pool.has_available() is True
    got = pool.select()
    assert got is not None
    assert got.last_status == "ok"


def test_two_key_pool_billing_429_stays_benched(tmp_path, monkeypatch):
    """A classified billing 429 keeps the full bench even with window wording —
    the pool stays empty rather than retrying a spent account every minute."""
    entries = [
        _entry(429, age_seconds=90, cred_id="k1", failure_reason="billing",
               error_message="Allocated quota exceeded"),
    ]
    pool = _load(tmp_path, monkeypatch, entries)
    assert pool.has_available() is False
    assert pool.select() is None


def test_short_window_survives_reload(tmp_path, monkeypatch):
    """last_error_message persists to auth.json, so a restart can't downgrade a
    short-window throttle back to a 1h bench."""
    from agent.credential_pool import _exhausted_ttl

    entry = _entry(429, age_seconds=90, cred_id="k1",
                   error_message="rpm exhausted")
    pool = _load(tmp_path, monkeypatch, [entry])
    # Re-read the persisted store and confirm the message round-tripped.
    stored = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    persisted = stored["credential_pool"]["st"][0]
    assert persisted["last_error_message"] == "rpm exhausted"
    # And that the reloaded entry still sizes to 60s.
    reloaded = pool._entries[0]
    assert _exhausted_ttl(
        reloaded.last_error_code,
        sole_credential=pool._is_sole_credential(),
        failure_reason=reloaded.failure_reason,
        error_message=reloaded.last_error_message,
    ) == 60
