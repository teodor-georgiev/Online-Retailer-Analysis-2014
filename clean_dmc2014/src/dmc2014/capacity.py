from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


DEFAULT_SHARED_CLI_CANDIDATES = (
    Path("/srv/projects/chatgpt-compute-chatgpt-edit/admin/vps_capacity.py"),
    Path("/srv/sentinelx-agents/lane-1/chatgpt-compute/admin/vps_capacity.py"),
)


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _local_cpu_count() -> int:
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        affinity = None
    if affinity:
        return max(1, len(affinity))
    return max(1, int(os.cpu_count() or 1))


def _local_fallback(profile: str) -> int:
    cpus = _local_cpu_count()
    if profile == "light":
        return min(2, cpus)
    if profile == "default":
        reserve = max(2, math.ceil(cpus * 0.25))
        return max(1, cpus - reserve)
    if profile == "batch":
        reserve = max(1, math.ceil(cpus * 0.125))
        return max(1, cpus - reserve)
    raise ValueError(f"unknown capacity profile: {profile}")


def _workers_from_cli(path: Path, profile: str) -> int | None:
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    try:
        completed = subprocess.run(
            [sys.executable, str(path), "--profile", profile, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        payload = json.loads(completed.stdout)
        workers = int(payload["workers"])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return workers if workers >= 1 else None


def resolve_workers(
    profile: str = "batch",
    *,
    shared_cli_candidates: Iterable[Path] | None = None,
) -> int:
    """Resolve the DMC thread budget without hard-coding the VPS core count."""
    override = os.environ.get("DMC2014_WORKERS")
    if override is not None:
        return _positive_int(override, "DMC2014_WORKERS")

    shared_env = os.environ.get("CHATGPT_WORKERS")
    if shared_env is not None:
        return _positive_int(shared_env, "CHATGPT_WORKERS")

    configured_cli = os.environ.get("CHATGPT_VPS_CAPACITY_CLI")
    if configured_cli:
        workers = _workers_from_cli(Path(configured_cli), profile)
        if workers is not None:
            return workers

    candidates = DEFAULT_SHARED_CLI_CANDIDATES if shared_cli_candidates is None else shared_cli_candidates
    for candidate in candidates:
        workers = _workers_from_cli(Path(candidate), profile)
        if workers is not None:
            return workers

    return _local_fallback(profile)
