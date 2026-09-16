from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dmc_points(y_true: Sequence[float], prediction: Sequence[float]) -> float:
    """Return the DMC 2014 point total: sum of absolute errors."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {pred.shape}")
    return float(np.abs(truth - pred).sum())


def best_threshold(
    y_true: Sequence[int],
    probability: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Select the hard-classification threshold with minimum DMC points.

    Ties are resolved deterministically in favour of the threshold closest to
    0.5, then the numerically smaller threshold.
    """
    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {prob.shape}")
    if thresholds is None:
        thresholds = np.linspace(0.20, 0.80, 241)
    candidates = [float(value) for value in thresholds]
    if not candidates:
        raise ValueError("thresholds must not be empty")

    scored: list[tuple[float, float]] = []
    for threshold in candidates:
        prediction = (prob >= threshold).astype(int)
        scored.append((dmc_points(truth, prediction), threshold))

    points, threshold = min(
        scored,
        key=lambda item: (item[0], abs(item[1] - 0.5), item[1]),
    )
    return float(threshold), float(points)
