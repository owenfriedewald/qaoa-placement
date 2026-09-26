"""Objective functions for exact-state QAOA placement simulations."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def expected_cost(probs: np.ndarray, costs: np.ndarray) -> float:
    return float(np.dot(probs, costs))


def cvar_cost(probs: np.ndarray, costs: np.ndarray, alpha: float) -> float:
    """Return lower-tail CVaR for minimization.

    The value is the expected cost of the best alpha probability mass. Smaller
    values are better. This uses exact probabilities, not finite-shot samples.
    """

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    order = np.argsort(costs)
    remaining = alpha
    weighted = 0.0
    for idx in order:
        if remaining <= 0:
            break
        mass = min(float(probs[idx]), remaining)
        weighted += mass * float(costs[idx])
        remaining -= mass
    if remaining > 1e-12:
        return float(costs[order[-1]])
    return weighted / alpha


def probability_below_threshold(probs: np.ndarray, costs: np.ndarray, threshold: float) -> float:
    return float(np.sum(probs[costs <= threshold]))


def objective_value(
    probs: np.ndarray,
    costs: np.ndarray,
    objective: str,
    objective_params: Mapping[str, float],
) -> float:
    if objective == "expected":
        return expected_cost(probs, costs)
    if objective == "cvar":
        return cvar_cost(probs, costs, alpha=float(objective_params["alpha"]))
    if objective == "beat_initial":
        # Minimize negative success probability.
        return -probability_below_threshold(probs, costs, float(objective_params["initial_cost"]) - 1e-12)
    if objective == "ratio_threshold":
        return -probability_below_threshold(probs, costs, float(objective_params["threshold_cost"]))
    raise ValueError(f"unsupported objective: {objective}")


def objective_label(objective: str, objective_params: Mapping[str, float]) -> str:
    if objective == "cvar":
        return f"cvar_{objective_params['alpha']:.2f}"
    if objective == "ratio_threshold":
        return f"ratio_threshold_{objective_params['ratio']:.2f}"
    return objective
