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


def _mock_local_host(monkeypatch, *, load, busy, available_gib=32):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    monkeypatch.setattr(os, "getloadavg", lambda: (load, load, load))
    monkeypatch.setattr(capacity, "_local_cpu_busy_fraction", lambda: busy)
    monkeypatch.setattr(
        capacity,
        "_local_memory_available_bytes",
        lambda: available_gib * 1024**3,
    )


def test_local_batch_fallback_uses_all_cores_when_idle(monkeypatch):
    _clear_overrides(monkeypatch)
    _mock_local_host(monkeypatch, load=0.0, busy=0.0)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 16


def test_local_batch_fallback_fills_utilization_gap(monkeypatch):
    _clear_overrides(monkeypatch)
    _mock_local_host(monkeypatch, load=8.0, busy=0.50)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 8


def test_local_batch_fallback_soft_brakes_at_load_28(monkeypatch):
    _clear_overrides(monkeypatch)
    _mock_local_host(monkeypatch, load=28.0, busy=0.0)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 10


def test_local_batch_fallback_hard_brakes_at_load_32(monkeypatch):
    _clear_overrides(monkeypatch)
    _mock_local_host(monkeypatch, load=32.0, busy=0.0)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 4


def test_local_batch_fallback_clamps_at_emergency_load(monkeypatch):
    _clear_overrides(monkeypatch)
    _mock_local_host(monkeypatch, load=48.0, busy=0.0)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 1


def test_local_default_fallback_remains_conservative(monkeypatch):
    _clear_overrides(monkeypatch)
    _mock_local_host(monkeypatch, load=13.2, busy=0.0)
    assert resolve_workers("default", shared_cli_candidates=[]) == 1


def test_local_fallback_backs_off_when_memory_headroom_is_low(monkeypatch):
    _clear_overrides(monkeypatch)
    _mock_local_host(monkeypatch, load=0.0, busy=0.0, available_gib=2)
    assert resolve_workers("batch", shared_cli_candidates=[]) == 1
