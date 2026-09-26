"""Tests for reduced token/permutation quality validation utilities."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from placement_core import PlacementProblem
from run_token_permutation_quality_validation import (
    aggregate_token_probs,
    build_reduced_model,
    evaluate_distribution,
    reduced_token_probabilities,
)
from token_permutation_encoding import (
    aggregate_probabilities_by_assignment,
    token_qaoa_circuit,
    valid_permutation_probability,
)


def tiny_problem() -> PlacementProblem:
    return PlacementProblem(
        cells=("A", "B"),
        sites=((0, 0), (1, 0), (0, 1)),
        nets=(("A", "B", 2.0),),
        penalty=20.0,
    )


class TokenPermutationQualityValidationTests(unittest.TestCase):
    def test_reduced_simulator_matches_qiskit_aggregation(self) -> None:
        problem = tiny_problem()
        assignment = {"A": 0, "B": 2}
        theta = [0.17, 0.23]
        model = build_reduced_model(problem)
        reduced = reduced_token_probabilities(model, assignment, "line", 1, theta)
        reduced_aggregated = aggregate_token_probs(model, reduced)

        circuit = token_qaoa_circuit(problem, theta[0], theta[1], "line", include_initial_state=True, initial_assignment=assignment)
        state = Statevector.from_instruction(circuit)
        self.assertAlmostEqual(valid_permutation_probability(problem, state), 1.0, places=10)
        qiskit_aggregated = aggregate_probabilities_by_assignment(problem, state)

        reduced_support = {key for key, prob in reduced_aggregated.items() if prob > 1e-10}
        self.assertEqual(reduced_support, set(qiskit_aggregated))
        for key in reduced_support:
            self.assertLess(abs(reduced_aggregated[key] - qiskit_aggregated[key]), 1e-10)

    def test_empty_aggregation_metrics_are_normalized(self) -> None:
        problem = tiny_problem()
        assignment = {"A": 0, "B": 2}
        model = build_reduced_model(problem)
        probs = reduced_token_probabilities(model, assignment, "complete", 2, [0.17, 0.23, 0.09, -0.31])
        metrics = evaluate_distribution(model, probs, assignment)
        self.assertAlmostEqual(float(metrics["aggregated_real_probability_mass"]), 1.0, places=10)
        self.assertTrue(metrics["empty_aggregation_valid"])
        self.assertTrue(np.isfinite(float(metrics["cvar_0.25"])))
        self.assertGreaterEqual(float(metrics["optimal_probability"]), 0.0)
        self.assertLessEqual(float(metrics["optimal_probability"]), 1.0)


if __name__ == "__main__":
    unittest.main()
