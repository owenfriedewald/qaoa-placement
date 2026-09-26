"""Independent statevector and archived-kernel checks for input-aware circuits."""
import itertools
import unittest
from dataclasses import replace

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.linalg import hadamard

from invariant_placement import (ReducedPlacement, circuit_hash, collision_only_qubo,
                                 distance_coefficients, placement_circuit, token_phase)
from placement_core import PlacementProblem, assignment_to_bits, build_qubo, hpwl
from run_paper_suite import row_xy_probs
from run_qaoa_simulation import assignment_key, constrained_cost_energies
from run_token_permutation_p3_quality_uplift import reduced_probs_scheduled, scheduled_token_edges
from run_token_permutation_quality_validation import build_reduced_model
from token_permutation_encoding import (assignment_to_token_sites, token_sites_to_bits,
                                       token_phase_circuit)


class InvariantPlacementTests(unittest.TestCase):
    def setUp(self):
        self.problem = PlacementProblem(("A", "B"), ((0, 0), (1, 0), (0, 2), (1, 2)),
                                        (("A", "B", 1.7),), 5.0)
        self.initial = {"A": 3, "B": 0}
        self.theta = [0.23, 0.17, -0.31, 0.09, 0.41]

    def test_both_methods_have_same_parameter_and_preparation_contract(self):
        for method in ("token", "penalty_row_xy"):
            circuit = placement_circuit(self.problem, method, initial=self.initial)
            self.assertEqual(circuit.num_parameters, 5)
            self.assertEqual(circuit.count_ops().get("measure", 0), 0)
            self.assertGreater(circuit.count_ops().get("x", 0), 0)
            self.assertEqual(circuit_hash(circuit), circuit_hash(
                placement_circuit(self.problem, method, initial=self.initial)))

    def test_vectorized_token_matches_archive_with_arbitrary_first_gamma(self):
        legacy = build_reduced_model(self.problem)
        for graph, schedule in itertools.product(("line", "ring"), ("rotating", "reversed", "empty_prioritized")):
            model = ReducedPlacement(self.problem, "token", graph, schedule)
            expected = reduced_probs_scheduled(legacy, self.initial, graph, schedule, 3,
                                              [1.91] + self.theta)
            np.testing.assert_allclose(model.probabilities(self.theta, self.initial), expected, atol=1e-12)

    def test_vectorized_penalty_matches_archive_with_removed_row_penalty(self):
        model = build_qubo(self.problem)
        assignments, _, energies, _ = constrained_cost_energies(self.problem, model)
        indices = {assignment_key(self.problem, a): i for i, a in enumerate(assignments)}
        start = indices[assignment_key(self.problem, self.initial)]
        expected = row_xy_probs([1.91] + self.theta, energies, 3, self.problem, assignments, indices, start)
        reduced = ReducedPlacement(self.problem, "penalty_row_xy")
        actual = reduced.probabilities(self.theta, self.initial)
        for a, value in zip(assignments, expected):
            self.assertAlmostEqual(actual[reduced.index[assignment_key(self.problem, a)]], value, places=11)

    def test_circuits_match_reduced_distributions_and_clean_ancilla(self):
        for method, reps in itertools.product(("token", "penalty_row_xy"), (1, 3)):
            theta = self.theta[:2 * reps - 1]
            circuit = placement_circuit(self.problem, method, reps, initial=self.initial, theta=theta)
            full = Statevector.from_instruction(circuit).probabilities()
            model = ReducedPlacement(self.problem, method)
            expected = model.probabilities(theta, self.initial, reps)
            support = []
            for row in model.states:
                if method == "token":
                    bits = token_sites_to_bits(self.problem, row)
                else:
                    bits = assignment_to_bits(self.problem, dict(zip(self.problem.cells, row)))
                support.append(sum(int(bit) << i for i, bit in enumerate(bits)))
            np.testing.assert_allclose(full[support], expected, atol=1e-11)
            self.assertAlmostEqual(float(full[support].sum()), 1.0, places=11)

    def test_symbolic_phase_matches_archive_on_arbitrary_input(self):
        from qiskit.circuit import Parameter
        gamma = Parameter("gamma")
        new = token_phase(self.problem, gamma).assign_parameters({gamma: 0.37})
        old = token_phase_circuit(self.problem, 0.37)
        state = Statevector(np.ones(2**new.num_qubits) / np.sqrt(2**new.num_qubits))
        np.testing.assert_allclose(state.copy().evolve(new).data, state.copy().evolve(old).data, atol=1e-11)

    def test_phase_completion_preserves_all_valid_pairs(self):
        for sites in (self.problem.sites[:3], ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))):
            m = len(sites)
            width = (m - 1).bit_length()
            for policy in ("zero", "l1"):
                coeff = distance_coefficients(sites, policy)
                values = hadamard(len(coeff)) @ coeff
                for a, b in itertools.product(range(m), repeat=2):
                    expected = abs(sites[a][0] - sites[b][0]) + abs(sites[a][1] - sites[b][1])
                    self.assertAlmostEqual(values[a | (b << width)], expected, places=8)

    def test_completed_phase_circuit_preserves_legal_superposition(self):
        problem = replace(self.problem, sites=self.problem.sites[:3])
        # Compare complete prepared p=3 algorithms at two parameter settings.
        for scale in (1.0, -0.7):
            theta = np.asarray(self.theta) * scale
            states = [Statevector.from_instruction(placement_circuit(problem, "token", theta=theta,
                      completion=policy)) for policy in ("zero", "l1")]
            np.testing.assert_allclose(states[0].data, states[1].data, atol=1e-9)

    def test_exact_schedule_sequences(self):
        problem = replace(self.problem, sites=tuple((i, 0) for i in range(6)))
        self.assertEqual(scheduled_token_edges(problem, "line", "reversed", 1),
                         ((4, 5), (3, 4), (2, 3), (1, 2), (0, 1)))
        self.assertEqual(scheduled_token_edges(problem, "line", "rotating", 1),
                         ((1, 2), (2, 3), (3, 4), (4, 5), (0, 1)))


if __name__ == "__main__":
    unittest.main()
