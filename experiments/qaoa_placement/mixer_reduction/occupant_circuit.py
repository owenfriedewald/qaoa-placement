"""Circuit prototype for sparse occupant-register placement mixers."""

from __future__ import annotations

import math
from typing import Dict, Literal, Mapping, Sequence, Tuple

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from placement_core import Assignment, PlacementProblem
from structured_mixers import SiteEdge, site_graph_edges


ExchangePrimitive = Literal["unitary", "local_mcrx"]


def occupant_label_count(problem: PlacementProblem) -> int:
    return len(problem.cells) + 1


def empty_label(problem: PlacementProblem) -> int:
    return len(problem.cells)


def occupant_qubit(problem: PlacementProblem, site_idx: int, label_idx: int) -> int:
    return site_idx * occupant_label_count(problem) + label_idx


def assignment_to_occupants(problem: PlacementProblem, assignment: Mapping[str, int]) -> Tuple[int, ...]:
    labels = [empty_label(problem)] * len(problem.sites)
    for cell_idx, cell in enumerate(problem.cells):
        labels[assignment[cell]] = cell_idx
    return tuple(labels)


def occupants_to_assignment(problem: PlacementProblem, occupants: Sequence[int]) -> Assignment | None:
    assignment: Assignment = {}
    seen_cells = set()
    empty = empty_label(problem)
    for site_idx, label in enumerate(occupants):
        if label == empty:
            continue
        if label < 0 or label >= len(problem.cells) or label in seen_cells:
            return None
        seen_cells.add(label)
        assignment[problem.cells[label]] = site_idx
    if len(assignment) != len(problem.cells):
        return None
    return assignment


def assignment_to_occupant_bits(problem: PlacementProblem, assignment: Mapping[str, int]) -> list[int]:
    labels = assignment_to_occupants(problem, assignment)
    bits = [0] * (len(problem.sites) * occupant_label_count(problem))
    for site_idx, label in enumerate(labels):
        bits[occupant_qubit(problem, site_idx, label)] = 1
    return bits


def occupant_bits_to_assignment(problem: PlacementProblem, bits: Sequence[int]) -> Assignment | None:
    labels = []
    label_count = occupant_label_count(problem)
    for site_idx in range(len(problem.sites)):
        active = [
            label
            for label in range(label_count)
            if bits[occupant_qubit(problem, site_idx, label)] == 1
        ]
        if len(active) != 1:
            return None
        labels.append(active[0])
    return occupants_to_assignment(problem, labels)


def prepare_occupant_basis_state(circuit: QuantumCircuit, problem: PlacementProblem, assignment: Mapping[str, int]) -> None:
    for qubit, bit in enumerate(assignment_to_occupant_bits(problem, assignment)):
        if bit:
            circuit.x(qubit)


def _basis_index(active_qubits: Sequence[int]) -> int:
    index = 0
    for qubit in active_qubits:
        index |= 1 << qubit
    return index


def label_pair_exchange_gate(beta: float) -> UnitaryGate:
    """Four-qubit exchange between |1001> and |0110> in local order.

    Local qubit order is:
    0: site A, label x
    1: site A, label y
    2: site B, label x
    3: site B, label y
    """

    matrix = np.eye(16, dtype=np.complex128)
    left = _basis_index([0, 3])
    right = _basis_index([1, 2])
    cos = np.cos(beta)
    minus_i_sin = -1j * np.sin(beta)
    matrix[left, left] = cos
    matrix[right, right] = cos
    matrix[left, right] = minus_i_sin
    matrix[right, left] = minus_i_sin
    return UnitaryGate(matrix, label="occ_ex")


def append_local_mcrx_exchange(circuit: QuantumCircuit, qubits: Sequence[int], beta: float) -> None:
    """Append an exact local exchange using CNOT mapping plus a 3-controlled RX.

    This implements the same four-qubit unitary as ``label_pair_exchange_gate``
    on qubit order [A_x, A_y, B_x, B_y]. It is still an exact two-level update
    on the full four-qubit space: invalid local basis states are unchanged.
    """

    if len(qubits) != 4:
        raise ValueError("local exchange requires exactly four qubits")
    target = qubits[0]
    controls = [qubits[1], qubits[2], qubits[3]]
    for other in controls:
        circuit.cx(target, other)
    # After the CNOT map, the two exchanged states share controls 1, 1, 0.
    circuit.x(qubits[3])
    circuit.mcrx(2.0 * beta, controls, target)
    circuit.x(qubits[3])
    for other in reversed(controls):
        circuit.cx(target, other)


def append_label_pair_exchange(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    beta: float,
    primitive: ExchangePrimitive = "unitary",
) -> None:
    if primitive == "unitary":
        circuit.append(label_pair_exchange_gate(beta), list(qubits))
    elif primitive == "local_mcrx":
        append_local_mcrx_exchange(circuit, qubits, beta)
    else:
        raise ValueError(f"unknown exchange primitive: {primitive}")


def append_site_pair_exchange(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    site_a: int,
    site_b: int,
    beta: float,
    primitive: ExchangePrimitive = "unitary",
) -> None:
    labels = range(occupant_label_count(problem))
    for left_label in labels:
        for right_label in range(left_label + 1, occupant_label_count(problem)):
            qubits = [
                occupant_qubit(problem, site_a, left_label),
                occupant_qubit(problem, site_a, right_label),
                occupant_qubit(problem, site_b, left_label),
                occupant_qubit(problem, site_b, right_label),
            ]
            append_label_pair_exchange(circuit, qubits, beta, primitive)


def build_sparse_occupant_mixer_circuit(
    problem: PlacementProblem,
    topology: str,
    beta: float,
    reps: int = 1,
    initial_assignment: Mapping[str, int] | None = None,
    include_initial_state: bool = False,
    primitive: ExchangePrimitive = "unitary",
) -> QuantumCircuit:
    circuit = QuantumCircuit(len(problem.sites) * occupant_label_count(problem))
    if include_initial_state:
        if initial_assignment is None:
            raise ValueError("initial_assignment is required when include_initial_state=True")
        prepare_occupant_basis_state(circuit, problem, initial_assignment)
    site_edges = site_graph_edges(problem, topology)
    for _ in range(reps):
        for site_a, site_b in site_edges:
            append_site_pair_exchange(circuit, problem, site_a, site_b, beta, primitive)
    return circuit


def exchange_primitive_circuit(beta: float, primitive: ExchangePrimitive) -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    append_label_pair_exchange(circuit, [0, 1, 2, 3], beta, primitive)
    return circuit


def circuit_ordered_sparse_edges(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    index_by_key: Mapping[Tuple[int, ...], int],
    topology: str,
) -> Tuple[Tuple[int, int], ...]:
    """Return feasible-basis edges in the same order as the circuit prototype."""

    ordered: list[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    site_edges = site_graph_edges(problem, topology)
    label_count = occupant_label_count(problem)
    occupant_rows = [assignment_to_occupants(problem, assignment) for assignment in assignments]
    for site_a, site_b in site_edges:
        for left_label in range(label_count):
            for right_label in range(left_label + 1, label_count):
                for left_idx, occupants in enumerate(occupant_rows):
                    labels = {occupants[site_a], occupants[site_b]}
                    if labels != {left_label, right_label}:
                        continue
                    moved_occupants = list(occupants)
                    moved_occupants[site_a], moved_occupants[site_b] = moved_occupants[site_b], moved_occupants[site_a]
                    moved_assignment = occupants_to_assignment(problem, moved_occupants)
                    if moved_assignment is None:
                        continue
                    right_idx = index_by_key[tuple(moved_assignment[cell] for cell in problem.cells)]
                    edge = (left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx)
                    if edge not in seen:
                        ordered.append(edge)
                        seen.add(edge)
    return tuple(ordered)


def circuit_metrics(circuit: QuantumCircuit) -> Dict[str, object]:
    ops = {name: int(count) for name, count in circuit.count_ops().items()}
    two_qubit_count = sum(count for name, count in ops.items() if name in {"cx", "cz", "ecr", "swap", "rxx", "ryy", "rzz"})
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "two_qubit_gate_count": int(two_qubit_count),
        "two_qubit_depth": two_qubit_depth(circuit),
        "operations": ops,
    }


def two_qubit_depth(circuit: QuantumCircuit) -> int:
    layers = [0] * circuit.num_qubits
    max_depth = 0
    for instruction in circuit.data:
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(qubits) != 2:
            continue
        layer = max(layers[q] for q in qubits) + 1
        for q in qubits:
            layers[q] = layer
        max_depth = max(max_depth, layer)
    return max_depth


def transpiled_metrics(circuit: QuantumCircuit, optimization_level: int = 3) -> Dict[str, object]:
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level, seed_transpiler=42)
    transpiled = pass_manager.run(circuit)
    metrics = circuit_metrics(transpiled)
    metrics["transpilation_backend"] = backend.name
    metrics["optimization_level"] = optimization_level
    return metrics


def basis_transpiled_metrics(
    circuit: QuantumCircuit,
    basis_gates: Sequence[str] = ("u3", "cx"),
    optimization_level: int = 1,
) -> Dict[str, object]:
    """Transpile while forcing opaque local unitaries into a gate basis."""

    transpiled = transpile(
        circuit,
        basis_gates=list(basis_gates),
        optimization_level=optimization_level,
        seed_transpiler=42,
    )
    metrics = circuit_metrics(transpiled)
    metrics["basis_gates"] = list(basis_gates)
    metrics["optimization_level"] = optimization_level
    return metrics


def probabilities_by_assignment(problem: PlacementProblem, statevector: Statevector) -> Dict[Tuple[int, ...], float]:
    probs: Dict[Tuple[int, ...], float] = {}
    probabilities = statevector.probabilities()
    num_qubits = len(problem.sites) * occupant_label_count(problem)
    for index, probability in enumerate(probabilities):
        if probability < 1e-12:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(num_qubits)]
        assignment = occupant_bits_to_assignment(problem, bits)
        if assignment is None:
            continue
        key = tuple(assignment[cell] for cell in problem.cells)
        probs[key] = probs.get(key, 0.0) + float(probability)
    return probs


def invalid_probability(problem: PlacementProblem, statevector: Statevector) -> float:
    probabilities = statevector.probabilities()
    num_qubits = len(problem.sites) * occupant_label_count(problem)
    invalid = 0.0
    for index, probability in enumerate(probabilities):
        if probability < 1e-12:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(num_qubits)]
        if occupant_bits_to_assignment(problem, bits) is None:
            invalid += float(probability)
    return invalid


def local_exchange_operation_count(problem: PlacementProblem, topology: str, reps: int) -> int:
    return len(site_graph_edges(problem, topology)) * math.comb(occupant_label_count(problem), 2) * reps
