"""Exact binary occupant-register phase separator for placement QAOA.

This module keeps the compact site-register encoding used by the promoted
structured binary partial-SWAP mixer.  The implementation is intentionally
transparent: it applies one diagonal phase term for each weighted cell pair and
ordered site pair by detecting the relevant occupant labels in the two site
registers.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from placement_core import Assignment, PlacementProblem, hpwl, manhattan
from compact_occupant_circuit import (
    assignment_to_binary_bits,
    binary_bits_to_assignment,
    binary_label_bits,
    binary_qubit,
    prepare_binary_basis_state,
)
from register_partial_swap import append_binary_register_partial_swap_site_edge
from structured_mixers import site_graph_edges


def binary_data_qubits(problem: PlacementProblem) -> int:
    return len(problem.sites) * binary_label_bits(problem)


def binary_total_qubits(problem: PlacementProblem) -> int:
    return binary_data_qubits(problem) + 1


def cell_label_by_name(problem: PlacementProblem) -> dict[str, int]:
    return {cell: idx for idx, cell in enumerate(problem.cells)}


def _label_pattern_flips(width: int, label: int) -> list[int]:
    return [bit for bit in range(width) if ((label >> bit) & 1) == 0]


def append_register_label_match_phase(
    circuit: QuantumCircuit,
    register_a: Sequence[int],
    label_a: int,
    register_b: Sequence[int],
    label_b: int,
    angle: float,
) -> None:
    """Append phase `exp(i angle)` if two registers equal the labels.

    Qiskit's ``mcp`` applies the phase when all controls and the target are in
    |1>.  We flip zero-valued bits of each label into that all-ones condition,
    apply the multi-controlled phase, and unflip.
    """

    if len(register_a) != len(register_b):
        raise ValueError("register widths differ")
    width = len(register_a)
    if not register_a:
        raise ValueError("empty registers are not supported")

    flips = []
    for bit in _label_pattern_flips(width, label_a):
        flips.append(register_a[bit])
    for bit in _label_pattern_flips(width, label_b):
        flips.append(register_b[bit])
    for qubit in flips:
        circuit.x(qubit)

    all_qubits = list(register_a) + list(register_b)
    target = all_qubits[-1]
    controls = all_qubits[:-1]
    if controls:
        circuit.mcp(angle, controls, target)
    else:
        circuit.p(angle, target)

    for qubit in reversed(flips):
        circuit.x(qubit)


def append_binary_placement_phase_separator(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    gamma: float,
) -> int:
    """Append exact diagonal HPWL phase terms in binary occupant encoding.

    Returns the number of nonzero diagonal phase terms appended.
    """

    width = binary_label_bits(problem)
    cell_labels = cell_label_by_name(problem)
    term_count = 0
    for left_cell, right_cell, weight in problem.nets:
        left_label = cell_labels[left_cell]
        right_label = cell_labels[right_cell]
        for left_site_idx, left_site in enumerate(problem.sites):
            left_register = [binary_qubit(problem, left_site_idx, bit) for bit in range(width)]
            for right_site_idx, right_site in enumerate(problem.sites):
                distance = manhattan(left_site, right_site)
                if distance == 0:
                    continue
                phase = -float(gamma) * float(weight) * float(distance)
                right_register = [binary_qubit(problem, right_site_idx, bit) for bit in range(width)]
                append_register_label_match_phase(
                    circuit,
                    left_register,
                    left_label,
                    right_register,
                    right_label,
                    phase,
                )
                term_count += 1
    return term_count


def binary_phase_term_count(problem: PlacementProblem) -> int:
    count = 0
    for _left_cell, _right_cell, _weight in problem.nets:
        for left_site in problem.sites:
            for right_site in problem.sites:
                if manhattan(left_site, right_site) != 0:
                    count += 1
    return count


def binary_phase_cost_from_bits(problem: PlacementProblem, bits: Sequence[int]) -> float | None:
    assignment = binary_bits_to_assignment(problem, bits)
    if assignment is None:
        return None
    return hpwl(problem, assignment)


def binary_basis_index(problem: PlacementProblem, assignment: Mapping[str, int], total_qubits: int | None = None) -> int:
    bits = assignment_to_binary_bits(problem, assignment)
    value = 0
    for idx, bit in enumerate(bits):
        value |= int(bit) << idx
    # Ancillas remain |0>; total_qubits only documents intended circuit width.
    _ = total_qubits
    return value


def phase_separator_effective_phases(
    problem: PlacementProblem,
    gamma: float,
    assignments: Sequence[Assignment],
) -> dict[tuple[int, ...], complex]:
    circuit = QuantumCircuit(binary_total_qubits(problem))
    append_binary_placement_phase_separator(circuit, problem, gamma)
    state = Statevector.from_instruction(circuit)
    # Statevector.from_instruction starts from |0>; use Operator-free direct
    # basis evolution to avoid forming huge diagonal matrices.
    phases: dict[tuple[int, ...], complex] = {}
    for assignment in assignments:
        basis = binary_basis_index(problem, assignment, circuit.num_qubits)
        ket = Statevector.from_int(basis, dims=2**circuit.num_qubits)
        evolved = ket.evolve(circuit)
        phases[tuple(assignment[cell] for cell in problem.cells)] = evolved.data[basis]
    _ = state
    return phases


def build_complete_structured_binary_qaoa_circuit(
    problem: PlacementProblem,
    topology: str,
    gammas: Sequence[float],
    betas: Sequence[float],
    initial_assignment: Mapping[str, int],
    measure: bool = False,
) -> QuantumCircuit:
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    data_qubits = binary_data_qubits(problem)
    mixer_ancilla = data_qubits
    circuit = QuantumCircuit(data_qubits + 1, data_qubits if measure else 0)
    prepare_binary_basis_state(circuit, problem, initial_assignment)
    for gamma, beta in zip(gammas, betas):
        append_binary_placement_phase_separator(circuit, problem, gamma)
        for site_a, site_b in site_graph_edges(problem, topology):
            append_binary_register_partial_swap_site_edge(circuit, problem, site_a, site_b, mixer_ancilla, beta)
    if measure:
        circuit.measure(range(data_qubits), range(data_qubits))
    return circuit


def complete_binary_qaoa_probabilities_by_assignment(problem: PlacementProblem, statevector: Statevector) -> dict[tuple[int, ...], float]:
    data_qubits = binary_data_qubits(problem)
    probs: dict[tuple[int, ...], float] = {}
    for index, probability in enumerate(statevector.probabilities()):
        if probability < 1e-12:
            continue
        if (index >> data_qubits) & 1:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(data_qubits)]
        assignment = binary_bits_to_assignment(problem, bits)
        if assignment is None:
            continue
        key = tuple(assignment[cell] for cell in problem.cells)
        probs[key] = probs.get(key, 0.0) + float(probability)
    return probs


def complete_binary_qaoa_invalid_probability(problem: PlacementProblem, statevector: Statevector) -> float:
    data_qubits = binary_data_qubits(problem)
    invalid = 0.0
    for index, probability in enumerate(statevector.probabilities()):
        if probability < 1e-12:
            continue
        if (index >> data_qubits) & 1:
            invalid += float(probability)
            continue
        bits = [(index >> qubit) & 1 for qubit in range(data_qubits)]
        if binary_bits_to_assignment(problem, bits) is None:
            invalid += float(probability)
    return invalid


def estimate_statevector_memory_gib(qubits: int) -> float:
    return (2**qubits) * 16.0 / (1024.0**3)


def projected_phase_separator_cx(problem: PlacementProblem, cx_per_term: int) -> int:
    return binary_phase_term_count(problem) * cx_per_term

