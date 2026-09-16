from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


GIB = 1024 ** 3
BATCH_MIN_CPU_FRACTION = 0.25
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


def _local_memory_available_bytes(path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, remainder = line.partition(":")
        if key != "MemAvailable" or not sep:
            continue
        fields = remainder.strip().split()
        if not fields:
            return None
        try:
            value = int(fields[0])
        except ValueError:
            return None
        multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
        return max(0, value * multiplier)
    return None


def _local_load_average() -> float:
    try:
        return max(0.0, float(os.getloadavg()[0]))
    except (AttributeError, OSError):
        return 0.0


def _local_fallback(profile: str) -> int:
    cpus = _local_cpu_count()
    if profile == "light":
        reserve = max(2, math.ceil(cpus * 0.25))
        max_workers = 2
        min_cpu_floor = 1
        memory_headroom = 3 * GIB
    elif profile == "default":
        reserve = max(2, math.ceil(cpus * 0.25))
        max_workers = cpus
        min_cpu_floor = 1
        memory_headroom = 4 * GIB
    elif profile == "batch":
        reserve = max(1, math.ceil(cpus * 0.125))
        max_workers = cpus
        min_cpu_floor = max(1, math.ceil(cpus * BATCH_MIN_CPU_FRACTION))
        memory_headroom = 3 * GIB
    else:
        raise ValueError(f"unknown capacity profile: {profile}")

    load = min(float(cpus), _local_load_average())
    pressure_budget = math.floor(cpus - reserve - load)
    cpu_budget = max(min_cpu_floor, pressure_budget)
    available_memory = _local_memory_available_bytes()
    if available_memory is not None and available_memory < memory_headroom:
        return 1
    return max(1, min(cpus, max_workers, cpu_budget))


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
