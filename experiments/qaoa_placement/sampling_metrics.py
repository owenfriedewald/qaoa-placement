"""Practical sampling metrics for placement QAOA experiments."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

import numpy as np


def probability_within_ratio(probs: np.ndarray, hpwls: np.ndarray, exact_hpwl: float, ratio: float, feasible_mask: np.ndarray) -> float:
    threshold = exact_hpwl * ratio
    return float(np.sum(probs[(hpwls <= threshold) & feasible_mask]))


def expected_shots(probability: float) -> Optional[float]:
    if probability <= 0:
        return None
    return 1.0 / probability


def best_of_k_expected_ratio(
    probs: np.ndarray,
    hpwls: np.ndarray,
    exact_hpwl: float,
    feasible_mask: np.ndarray,
    k: int,
) -> Optional[float]:
    """Expected best feasible HPWL ratio after k shots.

    Infeasible outcomes are retained as wasted shots. If no feasible sample has
    appeared after k draws, the contribution is treated as infinity and the
    returned value is None.
    """

    finite = np.isfinite(hpwls) & feasible_mask
    if not np.any(finite):
        return None
    values = sorted(set(float(value) for value in hpwls[finite]))
    expected = 0.0
    previous_cdf = 0.0
    for value in values:
        success_prob = float(np.sum(probs[(hpwls <= value) & feasible_mask]))
        cdf = 1.0 - (1.0 - success_prob) ** k
        mass = max(cdf - previous_cdf, 0.0)
        expected += value * mass
        previous_cdf = cdf
    if previous_cdf < 1.0 - 1e-9:
        return None
    return expected / exact_hpwl if exact_hpwl else 1.0


def finite_shot_best_ratio(sample_sequence: Sequence[int], hpwls: np.ndarray, exact_hpwl: float, feasible_mask: np.ndarray, k: int) -> Optional[float]:
    """Return best feasible ratio in the first k ordered sampled outcomes."""

    prefix = list(sample_sequence[:k])
    feasible_values = [float(hpwls[idx]) for idx in prefix if feasible_mask[idx] and math.isfinite(float(hpwls[idx]))]
    if not feasible_values:
        return None
    return min(feasible_values) / exact_hpwl if exact_hpwl else 1.0


def practical_sampling_metrics(
    probs: np.ndarray,
    counts: Mapping[int, int],
    sample_sequence: Sequence[int],
    hpwls: np.ndarray,
    exact_hpwl: float,
    initial_hpwl: float,
    feasible_mask: np.ndarray,
    optimal_mask: np.ndarray,
    shot_budget: int,
) -> Dict[str, object]:
    near = {}
    expected = {}
    for ratio in (1.01, 1.05, 1.10):
        probability = probability_within_ratio(probs, hpwls, exact_hpwl, ratio, feasible_mask)
        label = f"within_{int(round((ratio - 1.0) * 100))}pct"
        near[f"probability_{label}"] = probability
        near[f"expected_shots_{label}"] = expected_shots(probability)

    optimal_probability = float(np.sum(probs[optimal_mask & feasible_mask]))
    beat_initial_probability = float(np.sum(probs[(hpwls < initial_hpwl - 1e-12) & feasible_mask]))
    useful_probability = float(np.sum(probs[(hpwls < initial_hpwl - 1e-12) & feasible_mask]))
    for k in (10, 100, 1000, shot_budget):
        expected[f"best_of_{k}_expected_ratio"] = best_of_k_expected_ratio(probs, hpwls, exact_hpwl, feasible_mask, k)
        expected[f"best_of_{k}_sampled_ratio"] = finite_shot_best_ratio(sample_sequence, hpwls, exact_hpwl, feasible_mask, k)

    feasible_counts = sum(count for idx, count in counts.items() if feasible_mask[idx])
    useful_counts = sum(count for idx, count in counts.items() if feasible_mask[idx] and hpwls[idx] < initial_hpwl - 1e-12)
    return {
        "probability_beating_initial": beat_initial_probability,
        "expected_shots_to_optimum": expected_shots(optimal_probability),
        "useful_feasible_samples_per_1000": 1000.0 * useful_counts / max(sum(counts.values()), 1),
        "feasible_samples_per_1000": 1000.0 * feasible_counts / max(sum(counts.values()), 1),
        "useful_probability": useful_probability,
        **near,
        **expected,
    }
