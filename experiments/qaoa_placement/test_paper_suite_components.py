"""Focused tests for paper-suite mechanics."""

from __future__ import annotations

import math
import unittest

import numpy as np

from benchmark_suite import manifest_payload
from objective_functions import cvar_cost, expected_cost, objective_value
from ordering_control import make_palindromic_schedule
from run_paper_suite import apply_schedule_mixer
from sampling_metrics import finite_shot_best_ratio, best_of_k_expected_ratio


class ObjectiveFunctionTests(unittest.TestCase):
    def test_expected_cost(self) -> None:
        probs = np.asarray([0.5, 0.25, 0.25])
        costs = np.asarray([1.0, 3.0, 9.0])
        self.assertAlmostEqual(expected_cost(probs, costs), 3.5)

    def test_cvar_uses_lower_cost_tail(self) -> None:
        probs = np.asarray([0.5, 0.25, 0.25])
        costs = np.asarray([1.0, 3.0, 9.0])
        self.assertAlmostEqual(cvar_cost(probs, costs, alpha=0.5), 1.0)
        self.assertAlmostEqual(cvar_cost(probs, costs, alpha=0.75), (0.5 * 1.0 + 0.25 * 3.0) / 0.75)

    def test_threshold_objective_is_negative_probability(self) -> None:
        probs = np.asarray([0.2, 0.3, 0.5])
        costs = np.asarray([1.0, 2.0, 5.0])
        value = objective_value(probs, costs, "beat_initial", {"initial_cost": 5.0})
        self.assertAlmostEqual(value, -0.5)


class SamplingMetricTests(unittest.TestCase):
    def test_finite_shot_best_uses_ordered_sequence(self) -> None:
        hpwls = np.asarray([1.0, 5.0, 10.0])
        feasible = np.asarray([True, True, True])
        sequence = [2, 1, 0]
        self.assertAlmostEqual(finite_shot_best_ratio(sequence, hpwls, 1.0, feasible, 1), 10.0)
        self.assertAlmostEqual(finite_shot_best_ratio(sequence, hpwls, 1.0, feasible, 2), 5.0)
        self.assertAlmostEqual(finite_shot_best_ratio(sequence, hpwls, 1.0, feasible, 3), 1.0)

    def test_expected_best_of_k_accounts_for_wasted_infeasible_probability(self) -> None:
        probs = np.asarray([0.5, 0.5])
        hpwls = np.asarray([1.0, math.inf])
        feasible = np.asarray([True, False])
        self.assertIsNone(best_of_k_expected_ratio(probs, hpwls, 1.0, feasible, 1))


class PalindromicScheduleTests(unittest.TestCase):
    def test_palindromic_schedule_layers_are_forward_then_reverse(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3)]
        schedule = make_palindromic_schedule(edges)
        self.assertEqual(schedule.name, "palindromic_fr")
        self.assertEqual(schedule.edge_layers[0], tuple(edges))
        self.assertEqual(schedule.edge_layers[1], tuple(reversed(edges)))

    def test_palindromic_mixer_preserves_norm(self) -> None:
        schedule = make_palindromic_schedule([(0, 1), (1, 2)])
        state = np.zeros(3, dtype=np.complex128)
        state[0] = 1.0
        apply_schedule_mixer(state, beta=0.3, schedule=schedule, layer_idx=0, reps=1)
        self.assertAlmostEqual(float(np.sum(np.abs(state) ** 2)), 1.0)


class ManifestTests(unittest.TestCase):
    def test_manifest_hash_is_stable(self) -> None:
        rows = [{"instance_id": "a", "seed": 1}, {"instance_id": "b", "seed": 2}]
        first = manifest_payload(rows)
        second = manifest_payload(rows)
        self.assertEqual(first["schema_version"], "qaoa-placement-benchmark-v1")
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
