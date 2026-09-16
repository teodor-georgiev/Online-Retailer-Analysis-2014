import json
import os
from pathlib import Path
import sys

import pytest

from dmc2014.capacity import resolve_workers


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
    monkeypatch.delenv("DMC2014_WORKERS", raising=False)
    monkeypatch.delenv("CHATGPT_WORKERS", raising=False)
    script = tmp_path / "capacity_cli.py"
    script.write_text(
        "import json; print(json.dumps({'workers': 11}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHATGPT_VPS_CAPACITY_CLI", str(script))
    assert resolve_workers("batch") == 11


def test_local_batch_fallback_uses_most_but_not_all_affinity(monkeypatch):
    monkeypatch.delenv("DMC2014_WORKERS", raising=False)
    monkeypatch.delenv("CHATGPT_WORKERS", raising=False)
    monkeypatch.delenv("CHATGPT_VPS_CAPACITY_CLI", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    assert resolve_workers("batch", shared_cli_candidates=[]) == 14


def test_local_default_fallback_keeps_more_headroom(monkeypatch):
    monkeypatch.delenv("DMC2014_WORKERS", raising=False)
    monkeypatch.delenv("CHATGPT_WORKERS", raising=False)
    monkeypatch.delenv("CHATGPT_VPS_CAPACITY_CLI", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)))
    assert resolve_workers("default", shared_cli_candidates=[]) == 12
