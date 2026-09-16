import os

import pytest

import dmc2014.capacity as capacity
from dmc2014.capacity import resolve_workers


def _clear_overrides(monkeypatch):
    monkeypatch.delenv("DMC2014_WORKERS", raising=False)
    monkeypatch.delenv("CHATGPT_WORKERS", raising=False)
    monkeypatch.delenv("CHATGPT_VPS_CAPACITY_CLI", raising=False)


def test_dmc_override_wins(monkeypatch):
    monkeypatch.setenv("DMC2014_WORKERS", "9")
    monkeypatch.setenv("CHATGPT_WORKERS", "5")
    assert resolve_workers("batch") == 9


def test_chatgpt_worker_env_is_second_priority(monkeypatch):
    monkeypatch.delenv("DMC2014_WORKERS", raising=False)
    monkeypatch.setenv("CHATGPT_WORKERS", "6")
    assert resolve_workers("batch") == 6


def test_invalid_override_is_rejected(monkeypatch):
    monkeypatch.setenv("DMC2014_WORKERS", "0")
    with pytest.raises(ValueError, match="positive"):
        resolve_workers("batch")


def test_shared_capacity_cli_is_used_when_configured(monkeypatch, tmp_path):
    _clear_overrides(monkeypatch)
    script = tmp_path / "capacity_cli.py"
    script.write_text(
        "import json; print(json.dumps({'workers': 11}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHATGPT_VPS_CAPACITY_CLI", str(script))
    assert resolve_workers("batch") == 11


def test_local_batch_fallback_uses_most_but_not_all_affinity_when_idle(monkeypatch):
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(capacity, "_local_memory_available_bytes", lambda: 32 * 1024**3)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 14


def test_local_default_fallback_keeps_more_headroom_when_idle(monkeypatch):
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(capacity, "_local_memory_available_bytes", lambda: 32 * 1024**3)
    assert resolve_workers("default", shared_cli_candidates=[]) == 12


def test_local_batch_fallback_keeps_quarter_cpu_floor_under_high_load(monkeypatch):
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(os, "getloadavg", lambda: (13.2, 10.0, 8.0))
    monkeypatch.setattr(capacity, "_local_memory_available_bytes", lambda: 32 * 1024**3)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 4


def test_local_default_fallback_can_back_off_below_quarter_under_high_load(monkeypatch):
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(os, "getloadavg", lambda: (13.2, 10.0, 8.0))
    monkeypatch.setattr(capacity, "_local_memory_available_bytes", lambda: 32 * 1024**3)
    assert resolve_workers("default", shared_cli_candidates=[]) == 1


def test_local_fallback_backs_off_when_memory_headroom_is_low(monkeypatch):
    _clear_overrides(monkeypatch)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(capacity, "_local_memory_available_bytes", lambda: 2 * 1024**3)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 1
