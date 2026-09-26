"""Structured register-level partial-SWAP constructions."""

from __future__ import annotations

import math
from typing import Dict, Literal, Mapping, Sequence, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from placement_core import Assignment, PlacementProblem
from structured_mixers import site_graph_edges
from compact_occupant_circuit import (
    binary_bits_to_assignment,
    binary_label_bits,
    binary_qubit,
    prepare_binary_basis_state,
)
from occupant_circuit import basis_transpiled_metrics, circuit_metrics


RegisterPartialSwapConstruction = Literal["phase_estimation_cswap"]


def register_swap_matrix(width: int) -> np.ndarray:
    dim = 2 ** (2 * width)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    mask = (1 << width) - 1
    for basis in range(dim):
        left = basis & mask
        right = (basis >> width) & mask
        swapped = right | (left << width)
        matrix[swapped, basis] = 1.0
    return matrix


def register_partial_swap_matrix(width: int, beta: float) -> np.ndarray:
    swap = register_swap_matrix(width)
    return np.cos(beta) * np.eye(2 ** (2 * width), dtype=np.complex128) - 1j * np.sin(beta) * swap


def append_controlled_register_swap(
    circuit: QuantumCircuit,
    control: int,
    register_a: Sequence[int],
    register_b: Sequence[int],
) -> None:
    if len(register_a) != len(register_b):
        raise ValueError("register widths differ")
    for left, right in zip(register_a, register_b):
        circuit.cswap(control, left, right)


def append_phase_estimation_partial_swap(
    circuit: QuantumCircuit,
    register_a: Sequence[int],
    register_b: Sequence[int],
    ancilla: int,
    beta: float,
) -> None:
    """Append exact exp(-i beta SWAP_AB) using one clean ancilla.

    The circuit performs one-bit phase estimation for the Hermitian unitary
    SWAP, phases the +1 and -1 eigenspaces with RZ(2 beta), and uncomputes.
    The ancilla returns to |0> for all input states when it starts in |0>.
    """

    circuit.h(ancilla)
    append_controlled_register_swap(circuit, ancilla, register_a, register_b)
    circuit.h(ancilla)
    circuit.rz(2.0 * beta, ancilla)
    circuit.h(ancilla)
    append_controlled_register_swap(circuit, ancilla, register_a, register_b)
    circuit.h(ancilla)


def register_partial_swap_circuit(width: int, beta: float, construction: RegisterPartialSwapConstruction = "phase_estimation_cswap") -> QuantumCircuit:
    circuit = QuantumCircuit(2 * width + 1)
    register_a = list(range(width))
    register_b = list(range(width, 2 * width))
    ancilla = 2 * width
    if construction == "phase_estimation_cswap":
        append_phase_estimation_partial_swap(circuit, register_a, register_b, ancilla, beta)
    else:
        raise ValueError(f"unknown construction: {construction}")
    return circuit


def effective_data_unitary_from_clean_ancilla(circuit: QuantumCircuit, width: int) -> Tuple[np.ndarray, float]:
    data_dim = 2 ** (2 * width)
    full_qubits = 2 * width + 1
    matrix = np.zeros((data_dim, data_dim), dtype=np.complex128)
    max_ancilla_leakage = 0.0
    for basis in range(data_dim):
        state = Statevector.from_int(basis, dims=2**full_qubits)
        evolved = state.evolve(circuit)
        for out in range(2**full_qubits):
            amplitude = evolved.data[out]
            if abs(amplitude) < 1e-12:
                continue
            data = out & (data_dim - 1)
            ancilla = (out >> (2 * width)) & 1
            if ancilla == 0:
                matrix[data, basis] += amplitude
            else:
                max_ancilla_leakage = max(max_ancilla_leakage, float(abs(amplitude) ** 2))
    return matrix, max_ancilla_leakage


def append_binary_register_partial_swap_site_edge(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    site_a: int,
    site_b: int,
    ancilla: int,
    beta: float,
    construction: RegisterPartialSwapConstruction = "phase_estimation_cswap",
) -> None:
    width = binary_label_bits(problem)
    register_a = [binary_qubit(problem, site_a, bit_idx) for bit_idx in range(width)]
    register_b = [binary_qubit(problem, site_b, bit_idx) for bit_idx in range(width)]
    if construction == "phase_estimation_cswap":
        append_phase_estimation_partial_swap(circuit, register_a, register_b, ancilla, beta)
    else:
        raise ValueError(f"unknown construction: {construction}")


def build_structured_binary_sparse_mixer_circuit(
    problem: PlacementProblem,
    topology: str,
    beta: float,
    reps: int = 1,
    initial_assignment: Mapping[str, int] | None = None,
    include_initial_state: bool = False,
    construction: RegisterPartialSwapConstruction = "phase_estimation_cswap",
) -> QuantumCircuit:
    data_qubits = len(problem.sites) * binary_label_bits(problem)
    ancilla = data_qubits
    circuit = QuantumCircuit(data_qubits + 1)
    if include_initial_state:
        if initial_assignment is None:
            raise ValueError("initial_assignment is required when include_initial_state=True")
        prepare_binary_basis_state(circuit, problem, initial_assignment)
    for _ in range(reps):
        for site_a, site_b in site_graph_edges(problem, topology):
            append_binary_register_partial_swap_site_edge(circuit, problem, site_a, site_b, ancilla, beta, construction)
    return circuit


def structured_binary_probabilities_by_assignment(problem: PlacementProblem, statevector: Statevector) -> Dict[Tuple[int, ...], float]:
    probs: Dict[Tuple[int, ...], float] = {}
    probabilities = statevector.probabilities()
    data_qubits = len(problem.sites) * binary_label_bits(problem)
    for index, probability in enumerate(probabilities):
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


def structured_binary_ancilla_one_probability(statevector: Statevector, data_qubits: int) -> float:
    total = 0.0
    for index, probability in enumerate(statevector.probabilities()):
        if probability > 1e-12 and ((index >> data_qubits) & 1):
            total += float(probability)
    return total


def structured_binary_resource_metrics(
    problem: PlacementProblem,
    topology: str,
    reps: int,
    beta: float = 0.23,
    construction: RegisterPartialSwapConstruction = "phase_estimation_cswap",
    optimization_level: int = 3,
) -> Dict[str, object]:
    circuit = build_structured_binary_sparse_mixer_circuit(problem, topology, beta, reps, construction=construction)
    logical = circuit_metrics(circuit)
    measured = basis_transpiled_metrics(circuit, basis_gates=("u3", "cx"), optimization_level=optimization_level)
    return {
        "logical_qubits": len(problem.sites) * binary_label_bits(problem),
        "clean_ancillas": 1,
        "dirty_ancillas": 0,
        "total_qubits": circuit.num_qubits,
        "site_edges": len(site_graph_edges(problem, topology)),
        "site_edge_blocks": len(site_graph_edges(problem, topology)) * reps,
        "logical_depth": logical["depth"],
        "cx": measured["two_qubit_gate_count"],
        "transpiled_depth": measured["depth"],
        "two_qubit_depth": measured["two_qubit_depth"],
        "one_qubit_gates": measured["operations"].get("u3", 0),
        "cswap_count": logical["operations"].get("cswap", 0),
        "optimization_level": optimization_level,
    }
