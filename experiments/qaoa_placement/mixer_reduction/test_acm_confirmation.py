"""Independent endpoint and classical budget controls for the new campaign."""
import itertools
import unittest

import numpy as np

from placement_core import PlacementProblem, hpwl
from run_acm_confirmation import expected_best, greedy


class ConfirmationEndpointTests(unittest.TestCase):
    def test_order_statistic_against_explicit_draw_sequences(self):
        probs = np.array([.2, .3, .5])
        costs = np.array([1., 3., -100.])
        legal = np.array([True, True, False])
        for draws in (1, 2, 4):
            exact = 0.
            for sequence in itertools.product(range(3), repeat=draws):
                weight = np.prod(probs[list(sequence)])
                selected = min([4.] + [costs[i] for i in sequence if legal[i]])
                exact += weight * selected
            self.assertAlmostEqual(expected_best(probs, costs, legal, 4., draws), exact, places=12)

    def test_initial_optimum_is_never_lost(self):
        value = expected_best(np.array([.3, .7]), np.array([2., -10.]),
                              np.array([True, False]), 1., 128)
        self.assertAlmostEqual(value, 1.)

    def test_greedy_one_call_returns_initial(self):
        problem = PlacementProblem(("A", "B"), ((0, 0), (1, 0), (5, 0)), (("A", "B", 1.),), 5.)
        initial = {"A": 0, "B": 2}
        cost, calls = greedy(problem, initial, 1)
        self.assertEqual(calls, 1)
        self.assertEqual(cost, hpwl(problem, initial))

    def test_greedy_respects_cap_and_reaches_known_adjacent_optimum(self):
        problem = PlacementProblem(("A", "B"), ((0, 0), (1, 0), (5, 0)), (("A", "B", 1.),), 5.)
        initial = {"A": 0, "B": 2}
        for budget in range(1, 20):
            cost, calls = greedy(problem, initial, budget)
            self.assertLessEqual(calls, budget)
            self.assertGreaterEqual(cost, 1.)
            self.assertLessEqual(cost, 5.)
        self.assertEqual(greedy(problem, initial, 20)[0], 1.)


if __name__ == "__main__":
    unittest.main()
