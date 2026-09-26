"""Compact binary occupant-register prototype for sparse placement mixers."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from placement_core import Assignment, PlacementProblem
from structured_mixers import site_graph_edges
from occupant_circuit import basis_transpiled_metrics, circuit_metrics, empty_label, occupants_to_assignment


def binary_label_bits(problem: PlacementProblem) -> int:
    return math.ceil(math.log2(len(problem.cells) + 1))


def binary_qubit(problem: PlacementProblem, site_idx: int, bit_idx: int) -> int:
    return site_idx * binary_label_bits(problem) + bit_idx


def assignment_to_binary_occupants(problem: PlacementProblem, assignment: Mapping[str, int]) -> Tuple[int, ...]:
    labels = [empty_label(problem)] * len(problem.sites)
    for cell_idx, cell in enumerate(problem.cells):
        labels[assignment[cell]] = cell_idx
    return tuple(labels)


def assignment_to_binary_bits(problem: PlacementProblem, assignment: Mapping[str, int]) -> list[int]:
    width = binary_label_bits(problem)
    bits = [0] * (len(problem.sites) * width)
    labels = assignment_to_binary_occupants(problem, assignment)
    for site_idx, label in enumerate(labels):
        for bit_idx in range(width):
            bits[binary_qubit(problem, site_idx, bit_idx)] = (label >> bit_idx) & 1
    return bits


def binary_bits_to_assignment(problem: PlacementProblem, bits: Sequence[int]) -> Assignment | None:
    width = binary_label_bits(problem)
    labels = []
    for site_idx in range(len(problem.sites)):
        label = 0
        for bit_idx in range(width):
            label |= int(bits[binary_qubit(problem, site_idx, bit_idx)]) << bit_idx
        labels.append(label)
    if any(label > empty_label(problem) for label in labels):
        return None
    return occupants_to_assignment(problem, labels)


def prepare_binary_basis_state(circuit: QuantumCircuit, problem: PlacementProblem, assignment: Mapping[str, int]) -> None:
    for qubit, bit in enumerate(assignment_to_binary_bits(problem, assignment)):
        if bit:
            circuit.x(qubit)


def binary_site_edge_exchange_gate(label_count: int, beta: float) -> UnitaryGate:
    """Register-level partial swap for two binary occupant registers.

    The gate acts on local qubit order [A bits..., B bits...]. For valid labels
    a,b < label_count and a != b it applies cos(beta) |a,b> - i sin(beta)
    |b,a>. Same-label states and invalid codewords are unchanged.
    """

    width = math.ceil(math.log2(label_count))
    dim = 2 ** (2 * width)
    matrix = np.eye(dim, dtype=np.complex128)
    cos = np.cos(beta)
    minus_i_sin = -1j * np.sin(beta)
    for left_label in range(label_count):
        for right_label in range(left_label + 1, label_count):
            left = left_label | (right_label << width)
            right = right_label | (left_label << width)
            matrix[left, left] = cos
            matrix[right, right] = cos
            matrix[left, right] = minus_i_sin
            matrix[right, left] = minus_i_sin
    return UnitaryGate(matrix, label=f"bin_occ_ex_{label_count}")


def append_binary_site_edge_exchange(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    site_a: int,
    site_b: int,
    beta: float,
) -> None:
    width = binary_label_bits(problem)
    qubits = [
        *(binary_qubit(problem, site_a, bit_idx) for bit_idx in range(width)),
        *(binary_qubit(problem, site_b, bit_idx) for bit_idx in range(width)),
    ]
    circuit.append(binary_site_edge_exchange_gate(len(problem.cells) + 1, beta), qubits)


def build_sparse_binary_mixer_circuit(
    problem: PlacementProblem,
    topology: str,
    beta: float,
    reps: int = 1,
    initial_assignment: Mapping[str, int] | None = None,
    include_initial_state: bool = False,
) -> QuantumCircuit:
    circuit = QuantumCircuit(len(problem.sites) * binary_label_bits(problem))
    if include_initial_state:
        if initial_assignment is None:
            raise ValueError("initial_assignment is required when include_initial_state=True")
        prepare_binary_basis_state(circuit, problem, initial_assignment)
    for _ in range(reps):
        for site_a, site_b in site_graph_edges(problem, topology):
            append_binary_site_edge_exchange(circuit, problem, site_a, site_b, beta)
    return circuit


def binary_probabilities_by_assignment(problem: PlacementProblem, statevector: Statevector) -> Dict[Tuple[int, ...], float]:
    probs: Dict[Tuple[int, ...], float] = {}
    probabilities = statevector.probabilities()
    num_qubits = len(problem.sites) * binary_label_bits(problem)
    for index, probability in enumerate(probabilities):
        if probability < 1e-12:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(num_qubits)]
        assignment = binary_bits_to_assignment(problem, bits)
        if assignment is None:
            continue
        key = tuple(assignment[cell] for cell in problem.cells)
        probs[key] = probs.get(key, 0.0) + float(probability)
    return probs


def binary_invalid_probability(problem: PlacementProblem, statevector: Statevector) -> float:
    probabilities = statevector.probabilities()
    num_qubits = len(problem.sites) * binary_label_bits(problem)
    invalid = 0.0
    for index, probability in enumerate(probabilities):
        if probability < 1e-12:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(num_qubits)]
        if binary_bits_to_assignment(problem, bits) is None:
            invalid += float(probability)
    return invalid


def binary_resource_metrics(problem: PlacementProblem, topology: str, reps: int, beta: float = 0.23) -> Dict[str, object]:
    circuit = build_sparse_binary_mixer_circuit(problem, topology, beta, reps)
    logical = circuit_metrics(circuit)
    measured = basis_transpiled_metrics(circuit, basis_gates=("u3", "cx"), optimization_level=1)
    return {
        "logical_qubits": circuit.num_qubits,
        "ancilla_qubits": 0,
        "site_edges": len(site_graph_edges(problem, topology)),
        "site_edge_blocks": len(site_graph_edges(problem, topology)) * reps,
        "logical_depth": logical["depth"],
        "cx": measured["two_qubit_gate_count"],
        "transpiled_depth": measured["depth"],
        "two_qubit_depth": measured["two_qubit_depth"],
        "one_qubit_gates": measured["operations"].get("u3", 0),
    }
