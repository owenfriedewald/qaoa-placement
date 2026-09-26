"""Tests for explicit-EMPTY token/permutation encoding."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.quantum_info import Statevector

from placement_core import PlacementProblem, hpwl
from token_permutation_encoding import (
    aggregate_probabilities_by_assignment,
    assignment_to_token_sites,
    bits_to_assignment,
    expected_empty_degeneracy,
    prepare_token_permutation_state,
    token_data_qubits,
    token_graph_edges,
    token_permutation_graph_metrics,
    token_phase_circuit,
    token_qaoa_circuit,
    token_sites_to_bits,
    token_sites_to_assignment,
    valid_permutation_probability,
)


def tiny_problem() -> PlacementProblem:
    return PlacementProblem(
        cells=("A", "B"),
        sites=((0, 0), (1, 0), (0, 1)),
        nets=(("A", "B", 2.0),),
        penalty=20.0,
    )


class TokenPermutationEncodingTests(unittest.TestCase):
    def test_encode_decode_assignment_with_empty_token(self) -> None:
        problem = tiny_problem()
        assignment = {"A": 0, "B": 2}
        sites = assignment_to_token_sites(problem, assignment)
        self.assertEqual(sites, (0, 2, 1))
        decoded = token_sites_to_assignment(problem, sites)
        self.assertEqual(decoded, assignment)
        bits = token_sites_to_bits(problem, sites)
        self.assertEqual(bits_to_assignment(problem, bits), assignment)
        self.assertEqual(expected_empty_degeneracy(problem), 1)

    def test_connected_token_swap_graphs(self) -> None:
        problem = tiny_problem()
        for graph in ("complete", "line", "ring", "empty_star", "real_empty_priority"):
            with self.subTest(graph=graph):
                self.assertGreater(len(token_graph_edges(problem, graph)), 0)
                metrics = token_permutation_graph_metrics(problem, graph)
                self.assertTrue(metrics["connected"])

    def test_phase_matches_hpwl_on_legal_basis(self) -> None:
        problem = tiny_problem()
        gamma = 0.17
        circuit = token_phase_circuit(problem, gamma)
        for assignment in ({"A": 0, "B": 1}, {"A": 2, "B": 0}):
            bits = token_sites_to_bits(problem, assignment_to_token_sites(problem, assignment))
            index = sum(bit << idx for idx, bit in enumerate(bits))
            evolved = Statevector.from_int(index, dims=2**circuit.num_qubits).evolve(circuit)
            expected = np.exp(-1j * gamma * hpwl(problem, assignment))
            self.assertLess(abs(evolved.data[index] - expected), 1e-10)

    def test_qaoa_preserves_permutation_subspace(self) -> None:
        problem = tiny_problem()
        circuit = token_qaoa_circuit(problem, 0.17, 0.23, "line", include_initial_state=True)
        state = Statevector.from_instruction(circuit)
        self.assertAlmostEqual(valid_permutation_probability(problem, state), 1.0, places=10)
        probs = aggregate_probabilities_by_assignment(problem, state)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
