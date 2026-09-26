"""Correctness tests for structured mixer candidates."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from placement_core import PlacementProblem, build_qubo, hpwl
from run_qaoa_simulation import assignment_to_bits, qubo_energy

from occupant_circuit import (
    append_site_pair_exchange,
    assignment_to_occupant_bits,
    build_sparse_occupant_mixer_circuit,
    circuit_ordered_sparse_edges,
    exchange_primitive_circuit,
    invalid_probability,
    occupant_bits_to_assignment,
    occupant_label_count,
    occupant_qubit,
    prepare_occupant_basis_state,
    probabilities_by_assignment,
)
from structured_mixers import (
    apply_ordered_edges,
    complete_reference_counts,
    feasible_basis,
    graph_metrics,
    make_sparse_candidate,
    occupant_register_resource_estimate,
)
from qiskit.circuit import QuantumCircuit
from compact_occupant_circuit import (
    assignment_to_binary_bits,
    binary_bits_to_assignment,
    binary_invalid_probability,
    binary_label_bits,
    binary_probabilities_by_assignment,
    binary_site_edge_exchange_gate,
    append_binary_site_edge_exchange,
    build_sparse_binary_mixer_circuit,
    prepare_binary_basis_state,
)
from register_partial_swap import (
    build_structured_binary_sparse_mixer_circuit,
    effective_data_unitary_from_clean_ancilla,
    register_partial_swap_circuit,
    register_partial_swap_matrix,
    structured_binary_ancilla_one_probability,
    structured_binary_probabilities_by_assignment,
)
from binary_phase_separator import (
    append_binary_placement_phase_separator,
    binary_basis_index,
    binary_data_qubits,
    binary_phase_term_count,
    build_complete_structured_binary_qaoa_circuit,
    complete_binary_qaoa_invalid_probability,
    complete_binary_qaoa_probabilities_by_assignment,
)
from phase_separator_optimization import (
    coordinate_arithmetic_phase_polynomial_circuit,
    distance_lookup_phase_polynomial_flag_reuse_circuit,
    distance_lookup_phase_polynomial_circuit,
    gray_site_codes,
    min_weight_site_codes,
    order_parity_terms,
    parity_order_metrics,
    parity_term_records,
    shared_location_phase_circuit,
)
from cell_site_encoding import (
    assignment_to_cell_site_bits,
    bits_to_cell_site_assignment,
    cell_site_data_qubits,
    cell_site_phase_circuit,
    graph_metrics as cell_site_graph_metrics,
    legal_assignments as cell_site_legal_assignments,
    move_edges as cell_site_move_edges,
    swap_only_edges as cell_site_swap_only_edges,
)


class SparseLegalMixerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = PlacementProblem(
            cells=("A", "B"),
            sites=((0, 0), (1, 0), (2, 0)),
            nets=(("A", "B", 1.0),),
            penalty=20.0,
        )
        self.assignments, self.index = feasible_basis(self.problem)

    def test_sparse_line_graph_is_connected(self) -> None:
        candidate = make_sparse_candidate(self.problem, self.assignments, self.index, "line")
        metrics = graph_metrics(len(self.assignments), candidate.feasible_edges)
        self.assertEqual(metrics["connected_components"], 1)
        self.assertGreater(metrics["diameter"], 0)

    def test_sparse_edges_preserve_legality(self) -> None:
        candidate = make_sparse_candidate(self.problem, self.assignments, self.index, "line")
        for left, right in candidate.feasible_edges:
            self.assertEqual(len(set(self.assignments[left].values())), len(self.problem.cells))
            self.assertEqual(len(set(self.assignments[right].values())), len(self.problem.cells))

    def test_ordered_edge_update_preserves_norm(self) -> None:
        candidate = make_sparse_candidate(self.problem, self.assignments, self.index, "line")
        state = np.zeros(len(self.assignments), dtype=np.complex128)
        state[0] = 1.0
        apply_ordered_edges(state, beta=0.31, edges=candidate.feasible_edges)
        self.assertAlmostEqual(float(np.sum(np.abs(state) ** 2)), 1.0)

    def test_resource_estimate_is_far_below_complete_edge_count(self) -> None:
        reference = complete_reference_counts(n=4, m=6, reps=3)
        estimate = occupant_register_resource_estimate(
            PlacementProblem(
                cells=("A", "B", "C", "D"),
                sites=((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
                nets=(("A", "B", 1.0),),
                penalty=40.0,
            ),
            topology="grid",
            reps=3,
        )
        self.assertEqual(reference["transition_applications"], 7560)
        self.assertLess(estimate["primitive_exchange_operations_total"], reference["transition_applications"])

    def test_cost_energy_defined_for_sparse_basis(self) -> None:
        model = build_qubo(self.problem)
        values = [
            qubo_energy(model, assignment_to_bits(self.problem, assignment))
            for assignment in self.assignments
        ]
        hpwls = [hpwl(self.problem, assignment) for assignment in self.assignments]
        self.assertEqual(len(values), 6)
        self.assertEqual(min(hpwls), 1.0)


class OccupantCircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = PlacementProblem(
            cells=("A", "B"),
            sites=((0, 0), (1, 0), (2, 0)),
            nets=(("A", "B", 1.0),),
            penalty=20.0,
        )
        self.assignments, self.index = feasible_basis(self.problem)

    def test_occupant_encoding_round_trip_and_uniqueness(self) -> None:
        assignment = {"A": 0, "B": 2}
        bits = assignment_to_occupant_bits(self.problem, assignment)
        decoded = occupant_bits_to_assignment(self.problem, bits)
        self.assertEqual(decoded, assignment)
        label_count = occupant_label_count(self.problem)
        for site_idx in range(len(self.problem.sites)):
            self.assertEqual(sum(bits[site_idx * label_count : (site_idx + 1) * label_count]), 1)

    def test_local_exchange_gate_is_unitary(self) -> None:
        circuit = QuantumCircuit(4)
        from occupant_circuit import label_pair_exchange_gate

        circuit.append(label_pair_exchange_gate(0.37), [0, 1, 2, 3])
        matrix = Operator(circuit).data
        identity = matrix.conj().T @ matrix
        self.assertTrue(np.allclose(identity, np.eye(16), atol=1e-10))

    def test_local_mcrx_exchange_matches_reference_unitary(self) -> None:
        beta = 0.37
        reference = exchange_primitive_circuit(beta, "unitary")
        optimized = exchange_primitive_circuit(beta, "local_mcrx")
        delta = np.max(np.abs(Operator(reference).data - Operator(optimized).data))
        self.assertLess(delta, 1e-10)

    def test_local_mcrx_exchange_leaves_invalid_states_unchanged(self) -> None:
        circuit = QuantumCircuit(4)
        # |1111> is outside the local one-hot occupant subspace for two sites.
        circuit.x([0, 1, 2, 3])
        from occupant_circuit import append_local_mcrx_exchange

        append_local_mcrx_exchange(circuit, [0, 1, 2, 3], beta=0.41)
        probs = Statevector.from_instruction(circuit).probabilities_dict()
        self.assertAlmostEqual(probs.get("1111", 0.0), 1.0, places=10)

    def test_empty_site_movement_preserves_legality(self) -> None:
        assignment = {"A": 0, "B": 2}
        circuit = QuantumCircuit(len(self.problem.sites) * occupant_label_count(self.problem))
        prepare_occupant_basis_state(circuit, self.problem, assignment)
        append_site_pair_exchange(circuit, self.problem, site_a=0, site_b=1, beta=np.pi / 2, primitive="local_mcrx")
        state = Statevector.from_instruction(circuit)
        probs = probabilities_by_assignment(self.problem, state)
        self.assertAlmostEqual(invalid_probability(self.problem, state), 0.0, places=10)
        self.assertAlmostEqual(probs[(1, 2)], 1.0, places=10)

    def test_unitary_circuit_matches_reduced_sparse_dynamics_on_tiny_case(self) -> None:
        self._assert_circuit_matches_reduced_sparse_dynamics("unitary")

    def test_local_mcrx_circuit_matches_reduced_sparse_dynamics_on_tiny_case(self) -> None:
        self._assert_circuit_matches_reduced_sparse_dynamics("local_mcrx")

    def _assert_circuit_matches_reduced_sparse_dynamics(self, primitive: str) -> None:
        beta = 0.23
        initial = self.assignments[0]
        circuit = build_sparse_occupant_mixer_circuit(
            self.problem,
            topology="line",
            beta=beta,
            reps=1,
            initial_assignment=initial,
            include_initial_state=True,
            primitive=primitive,
        )
        circuit_probs = probabilities_by_assignment(self.problem, Statevector.from_instruction(circuit))
        ordered_edges = circuit_ordered_sparse_edges(self.problem, self.assignments, self.index, "line")
        state = np.zeros(len(self.assignments), dtype=np.complex128)
        state[0] = 1.0
        apply_ordered_edges(state, beta, ordered_edges)
        reduced_probs = {
            tuple(assignment[cell] for cell in self.problem.cells): float(abs(state[idx]) ** 2)
            for idx, assignment in enumerate(self.assignments)
        }
        for key in set(circuit_probs) | set(reduced_probs):
            value = reduced_probs.get(key, 0.0)
            self.assertAlmostEqual(circuit_probs.get(key, 0.0), value, places=10)


class CompactBinaryOccupantCircuitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = PlacementProblem(
            cells=("A", "B"),
            sites=((0, 0), (1, 0), (2, 0)),
            nets=(("A", "B", 1.0),),
            penalty=20.0,
        )
        self.assignments, self.index = feasible_basis(self.problem)

    def test_binary_encoding_round_trip_and_qubit_count(self) -> None:
        assignment = {"A": 0, "B": 2}
        bits = assignment_to_binary_bits(self.problem, assignment)
        self.assertEqual(len(bits), len(self.problem.sites) * binary_label_bits(self.problem))
        self.assertEqual(binary_bits_to_assignment(self.problem, bits), assignment)

    def test_binary_site_edge_gate_is_unitary(self) -> None:
        circuit = QuantumCircuit(2 * binary_label_bits(self.problem))
        circuit.append(binary_site_edge_exchange_gate(len(self.problem.cells) + 1, 0.37), circuit.qubits)
        matrix = Operator(circuit).data
        self.assertTrue(np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=1e-10))

    def test_binary_empty_site_movement_preserves_legality(self) -> None:
        assignment = {"A": 0, "B": 2}
        circuit = QuantumCircuit(len(self.problem.sites) * binary_label_bits(self.problem))
        prepare_binary_basis_state(circuit, self.problem, assignment)
        append_binary_site_edge_exchange(circuit, self.problem, site_a=0, site_b=1, beta=np.pi / 2)
        state = Statevector.from_instruction(circuit)
        probs = binary_probabilities_by_assignment(self.problem, state)
        self.assertAlmostEqual(binary_invalid_probability(self.problem, state), 0.0, places=10)
        self.assertAlmostEqual(probs[(1, 2)], 1.0, places=10)

    def test_binary_circuit_matches_reduced_sparse_dynamics_on_tiny_case(self) -> None:
        beta = 0.23
        initial = self.assignments[0]
        circuit = build_sparse_binary_mixer_circuit(
            self.problem,
            topology="line",
            beta=beta,
            reps=1,
            initial_assignment=initial,
            include_initial_state=True,
        )
        circuit_probs = binary_probabilities_by_assignment(self.problem, Statevector.from_instruction(circuit))
        ordered_edges = circuit_ordered_sparse_edges(self.problem, self.assignments, self.index, "line")
        state = np.zeros(len(self.assignments), dtype=np.complex128)
        state[0] = 1.0
        apply_ordered_edges(state, beta, ordered_edges)
        reduced_probs = {
            tuple(assignment[cell] for cell in self.problem.cells): float(abs(state[idx]) ** 2)
            for idx, assignment in enumerate(self.assignments)
        }
        for key in set(circuit_probs) | set(reduced_probs):
            self.assertAlmostEqual(circuit_probs.get(key, 0.0), reduced_probs.get(key, 0.0), places=10)


class RegisterPartialSwapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = PlacementProblem(
            cells=("A", "B"),
            sites=((0, 0), (1, 0), (2, 0)),
            nets=(("A", "B", 1.0),),
            penalty=20.0,
        )
        self.assignments, self.index = feasible_basis(self.problem)

    def test_phase_estimation_partial_swap_matches_target_for_widths(self) -> None:
        for width in (1, 2, 3):
            for beta in (0.0, 0.23, -0.41, np.pi / 2):
                circuit = register_partial_swap_circuit(width, beta)
                effective, leakage = effective_data_unitary_from_clean_ancilla(circuit, width)
                target = register_partial_swap_matrix(width, beta)
                self.assertLess(np.max(np.abs(effective - target)), 1e-10)
                self.assertLess(leakage, 1e-10)

    def test_equal_register_state_gets_symmetric_phase(self) -> None:
        beta = 0.37
        width = 2
        circuit = register_partial_swap_circuit(width, beta)
        effective, _leakage = effective_data_unitary_from_clean_ancilla(circuit, width)
        equal_label_state = 2 | (2 << width)
        self.assertAlmostEqual(effective[equal_label_state, equal_label_state], np.exp(-1j * beta))

    def test_distinct_register_state_partial_swaps(self) -> None:
        beta = 0.37
        width = 2
        circuit = register_partial_swap_circuit(width, beta)
        effective, _leakage = effective_data_unitary_from_clean_ancilla(circuit, width)
        left = 1 | (2 << width)
        right = 2 | (1 << width)
        self.assertAlmostEqual(effective[left, left], np.cos(beta))
        self.assertAlmostEqual(effective[right, left], -1j * np.sin(beta))

    def test_structured_binary_sparse_mixer_preserves_legality_and_clean_ancilla(self) -> None:
        beta = 0.23
        initial = self.assignments[0]
        circuit = build_structured_binary_sparse_mixer_circuit(
            self.problem,
            topology="line",
            beta=beta,
            reps=1,
            initial_assignment=initial,
            include_initial_state=True,
        )
        statevector = Statevector.from_instruction(circuit)
        data_qubits = len(self.problem.sites) * binary_label_bits(self.problem)
        self.assertLess(structured_binary_ancilla_one_probability(statevector, data_qubits), 1e-10)
        probs = structured_binary_probabilities_by_assignment(self.problem, statevector)
        # For this 2-cell/3-site tiny case there is only one EMPTY site, so
        # exp(-i beta SWAP) agrees with the reduced sparse mixer up to a global phase.
        ordered_edges = circuit_ordered_sparse_edges(self.problem, self.assignments, self.index, "line")
        reduced = np.zeros(len(self.assignments), dtype=np.complex128)
        reduced[0] = 1.0
        apply_ordered_edges(reduced, beta, ordered_edges)
        for idx, assignment in enumerate(self.assignments):
            key = tuple(assignment[cell] for cell in self.problem.cells)
            self.assertAlmostEqual(probs.get(key, 0.0), float(abs(reduced[idx]) ** 2), places=10)


class BinaryPhaseSeparatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = PlacementProblem(
            cells=("A", "B"),
            sites=((0, 0), (1, 0), (2, 0)),
            nets=(("A", "B", 2.0),),
            penalty=20.0,
        )
        self.assignments, self.index = feasible_basis(self.problem)

    def test_phase_separator_term_count_skips_zero_distance(self) -> None:
        self.assertEqual(binary_phase_term_count(self.problem), 6)

    def test_phase_separator_matches_hpwl_on_legal_basis_states(self) -> None:
        gamma = 0.19
        circuit = QuantumCircuit(binary_data_qubits(self.problem) + 1)
        append_binary_placement_phase_separator(circuit, self.problem, gamma)
        for assignment in self.assignments:
            basis = binary_basis_index(self.problem, assignment, circuit.num_qubits)
            evolved = Statevector.from_int(basis, dims=2**circuit.num_qubits).evolve(circuit)
            expected = np.exp(-1j * gamma * hpwl(self.problem, assignment))
            self.assertAlmostEqual(evolved.data[basis], expected)

    def test_complete_binary_qaoa_preserves_legality_and_moves_empty_site(self) -> None:
        initial = {"A": 0, "B": 2}
        circuit = build_complete_structured_binary_qaoa_circuit(
            self.problem,
            topology="line",
            gammas=[0.17],
            betas=[np.pi / 2],
            initial_assignment=initial,
        )
        state = Statevector.from_instruction(circuit)
        self.assertLess(complete_binary_qaoa_invalid_probability(self.problem, state), 1e-10)
        probs = complete_binary_qaoa_probabilities_by_assignment(self.problem, state)
        self.assertGreater(sum(prob for key, prob in probs.items() if key != (0, 2)), 0.0)

    def test_shared_location_phase_separator_matches_hpwl(self) -> None:
        gamma = -0.23
        circuit = shared_location_phase_circuit(self.problem, gamma)
        self._assert_phase_circuit_matches_hpwl(circuit, gamma)

    def test_distance_lookup_phase_polynomial_matches_hpwl(self) -> None:
        gamma = 0.31
        circuit = distance_lookup_phase_polynomial_circuit(self.problem, gamma)
        self._assert_phase_circuit_matches_hpwl(circuit, gamma)

    def test_ordered_distance_lookup_phase_polynomial_matches_hpwl(self) -> None:
        gamma = 0.31
        for policy in ("greedy_support_overlap", "beam_search"):
            circuit = distance_lookup_phase_polynomial_circuit(
                self.problem,
                gamma,
                ordering_policy=policy,
                cancel_adjacent_cx=True,
            )
            self._assert_phase_circuit_matches_hpwl(circuit, gamma)

    def test_flag_reuse_distance_lookup_phase_polynomial_matches_hpwl(self) -> None:
        gamma = -0.29
        circuit = distance_lookup_phase_polynomial_flag_reuse_circuit(
            self.problem,
            gamma,
            ordering_policy="beam_search",
            cancel_adjacent_cx=True,
        )
        self._assert_phase_circuit_matches_hpwl(circuit, gamma)

    def test_custom_site_code_distance_lookup_phase_polynomial_matches_hpwl(self) -> None:
        gamma = 0.23
        for codes in (gray_site_codes(self.problem), min_weight_site_codes(self.problem)):
            circuit = distance_lookup_phase_polynomial_circuit(
                self.problem,
                gamma,
                ordering_policy="beam_search",
                cancel_adjacent_cx=True,
                site_codes=codes,
            )
            self._assert_phase_circuit_matches_hpwl(circuit, gamma)

    def test_ordered_parity_terms_preserve_multiset(self) -> None:
        records, _global_phase = parity_term_records(self.problem, 0.31)
        expected = sorted(int(term["term_id"]) for term in records)
        repository = parity_order_metrics(order_parity_terms(records, "repository_order"))
        for policy in ("greedy_support_overlap", "beam_search"):
            ordered = order_parity_terms(records, policy)
            self.assertEqual(sorted(int(term["term_id"]) for term in ordered), expected)
            metrics = parity_order_metrics(ordered)
            self.assertLessEqual(metrics["explicit_parity_cx"], repository["original_parity_cx"])

    def test_coordinate_arithmetic_phase_polynomial_matches_hpwl(self) -> None:
        gamma = -0.17
        circuit = coordinate_arithmetic_phase_polynomial_circuit(self.problem, gamma)
        self._assert_phase_circuit_matches_hpwl(circuit, gamma)

    def test_cell_site_encoding_roundtrip(self) -> None:
        assignment = {"A": 0, "B": 2}
        bits = assignment_to_cell_site_bits(self.problem, assignment)
        self.assertEqual(len(bits), cell_site_data_qubits(self.problem))
        self.assertEqual(bits_to_cell_site_assignment(self.problem, bits), assignment)

    def test_cell_site_phase_separator_matches_hpwl(self) -> None:
        gamma = 0.19
        circuit = cell_site_phase_circuit(self.problem, gamma)
        for assignment in self.assignments:
            bits = assignment_to_cell_site_bits(self.problem, assignment)
            basis = sum(int(bit) << idx for idx, bit in enumerate(bits))
            evolved = Statevector.from_int(basis, dims=2**circuit.num_qubits).evolve(circuit)
            expected = np.exp(-1j * gamma * hpwl(self.problem, assignment))
            self.assertAlmostEqual(evolved.data[basis], expected)

    def test_cell_site_swap_only_freezes_occupied_site_set(self) -> None:
        problem = PlacementProblem(
            cells=("A", "B"),
            sites=((0, 0), (1, 0), (2, 0)),
            nets=(("A", "B", 1.0),),
            penalty=20.0,
        )
        assignments = cell_site_legal_assignments(problem)
        swap_metrics = cell_site_graph_metrics(
            len(assignments),
            cell_site_swap_only_edges(problem, assignments, "complete"),
        )
        move_metrics = cell_site_graph_metrics(
            len(assignments),
            cell_site_move_edges(problem, assignments, ((0, 1), (1, 2), (0, 2))),
        )
        self.assertGreater(swap_metrics["connected_components"], 1)
        self.assertTrue(move_metrics["connected"])

    def _assert_phase_circuit_matches_hpwl(self, circuit: QuantumCircuit, gamma: float) -> None:
        for assignment in self.assignments:
            basis = binary_basis_index(self.problem, assignment, circuit.num_qubits)
            evolved = Statevector.from_int(basis, dims=2**circuit.num_qubits).evolve(circuit)
            expected = np.exp(-1j * gamma * hpwl(self.problem, assignment))
            self.assertAlmostEqual(evolved.data[basis], expected)
            location_mask = ~((1 << binary_data_qubits(self.problem)) - 1)
            leaked = sum(
                prob
                for idx, prob in enumerate(evolved.probabilities())
                if idx & location_mask
            )
            self.assertLess(leaked, 1e-10)


if __name__ == "__main__":
    unittest.main()
