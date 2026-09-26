"""Regression checks for the review-driven null and scaling controls."""

from __future__ import annotations

import unittest

from eda_bridge.analyze_uniform_feasible_null import grouped_rows, hit_probability
from mixer_reduction.run_token_resource_scaling_audit import representative_problem
from token_permutation_encoding import token_data_qubits, token_register_width


class UniformNullTests(unittest.TestCase):
    def test_token_degeneracy_reproduces_review_probability(self) -> None:
        probability = hit_probability(2.0 / 720.0, 4096)
        self.assertGreater(probability, 0.99998)
        self.assertLess(probability, 1.0)

    def test_four_distinct_optimizer_seeds_are_required(self) -> None:
        valid = [
            {"window_id": "w0", "seed": str(seed)} for seed in (11, 17, 23, 29)
        ]
        self.assertEqual(len(grouped_rows(valid)["w0"]), 4)
        with self.assertRaises(ValueError):
            grouped_rows(valid[:-1] + [valid[0]])


class ScalingConstructionTests(unittest.TestCase):
    def test_representative_graph_and_token_register_sizes(self) -> None:
        expected_nets = {(4, 6): 6, (5, 8): 10, (6, 9): 12, (8, 12): 16}
        for (cells, sites), nets in expected_nets.items():
            width = 3 if sites in (6, 9) else 4
            height = 2 if sites in (6, 8) else 3
            problem = representative_problem(cells, sites, width, height)
            self.assertEqual(len(problem.nets), nets)
            self.assertEqual(token_register_width(problem), (sites - 1).bit_length())
            self.assertEqual(token_data_qubits(problem), sites * (sites - 1).bit_length())


if __name__ == "__main__":
    unittest.main()
