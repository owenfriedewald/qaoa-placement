"""Tests for token/permutation p=3 schedule variants."""

from __future__ import annotations

import unittest

from placement_core import PlacementProblem
from run_token_permutation_p3_quality_uplift import SCHEDULES, reduced_probs_scheduled, scheduled_token_edges
from run_token_permutation_quality_validation import build_reduced_model
from token_permutation_encoding import token_graph_edges


def tiny_problem() -> PlacementProblem:
    return PlacementProblem(
        cells=("A", "B"),
        sites=((0, 0), (1, 0), (0, 1)),
        nets=(("A", "B", 2.0),),
        penalty=20.0,
    )


class TokenPermutationP3ScheduleTests(unittest.TestCase):
    def test_schedules_preserve_edge_multiset(self) -> None:
        problem = tiny_problem()
        for graph in ("line", "ring"):
            expected = sorted(token_graph_edges(problem, graph))
            for schedule in SCHEDULES:
                with self.subTest(graph=graph, schedule=schedule):
                    self.assertEqual(sorted(scheduled_token_edges(problem, graph, schedule, 1)), expected)

    def test_scheduled_reduced_probabilities_are_normalized(self) -> None:
        problem = tiny_problem()
        model = build_reduced_model(problem)
        assignment = {"A": 0, "B": 2}
        for schedule in SCHEDULES:
            with self.subTest(schedule=schedule):
                probs = reduced_probs_scheduled(model, assignment, "line", schedule, 3, [0.17, 0.23] * 3)
                self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
