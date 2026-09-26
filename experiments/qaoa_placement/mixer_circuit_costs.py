"""Circuit-cost diagnostics for legal placement transition schedules."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from mixer_diagnostics import Edge, MixerSchedule, edge_hamming_distance, schedule_logical_depth_per_qaoa_layer
from placement_core import Assignment, PlacementProblem, assignment_to_bits


def circuit_metrics(circuit: QuantumCircuit) -> Dict[str, object]:
    ops = {name: int(count) for name, count in circuit.count_ops().items()}
    two_qubit_count = sum(count for name, count in ops.items() if name in {"cx", "cz", "ecr", "swap", "rxx", "ryy", "rzz"})
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "two_qubit_gate_count": int(two_qubit_count),
        "operations": ops,
    }


def transpiled_metrics(circuit: QuantumCircuit) -> Dict[str, object]:
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=42)
    transpiled = pass_manager.run(circuit)
    metrics = circuit_metrics(transpiled)
    metrics["transpilation_backend"] = backend.name
    metrics["optimization_level"] = 3
    return metrics


def append_exact_transition(circuit: QuantumCircuit, left_bits: Sequence[int], right_bits: Sequence[int], beta: float) -> None:
    """Append an exact computational-basis two-level RX transition.

    This diagnostic decomposition maps the two bitstrings to a subspace where
    they differ only on one pivot qubit, applies an all-other-qubit controlled
    RX, and uncomputes. It is not claimed to be hardware efficient.
    """

    diff = [idx for idx, (left, right) in enumerate(zip(left_bits, right_bits)) if left != right]
    if not diff:
        return
    pivot = diff[0]
    for qubit in diff[1:]:
        circuit.cx(pivot, qubit)

    mapped_left = list(left_bits)
    mapped_right = list(right_bits)
    for qubit in diff[1:]:
        mapped_left[qubit] ^= mapped_left[pivot]
        mapped_right[qubit] ^= mapped_right[pivot]

    common = mapped_left if mapped_left[pivot] == left_bits[pivot] else mapped_right
    controls = [idx for idx in range(len(left_bits)) if idx != pivot]
    zero_controls = [idx for idx in controls if common[idx] == 0]
    for qubit in zero_controls:
        circuit.x(qubit)

    if controls:
        circuit.mcrx(2.0 * beta, controls, pivot)
    else:
        circuit.rx(2.0 * beta, pivot)

    for qubit in reversed(zero_controls):
        circuit.x(qubit)
    for qubit in reversed(diff[1:]):
        circuit.cx(pivot, qubit)


def representative_transition_circuits(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    edges: Sequence[Edge],
    beta: float = 0.123,
) -> Dict[int, QuantumCircuit]:
    circuits: Dict[int, QuantumCircuit] = {}
    for left, right in edges:
        hamming = edge_hamming_distance(problem, assignments[left], assignments[right])
        if hamming in circuits:
            continue
        circuit = QuantumCircuit(problem.num_variables)
        append_exact_transition(
            circuit,
            assignment_to_bits(problem, assignments[left]),
            assignment_to_bits(problem, assignments[right]),
            beta,
        )
        circuits[hamming] = circuit
    return circuits


def schedule_costs(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    edges: Sequence[Edge],
    schedule: MixerSchedule,
    reps: int,
    full_transpile_edge_threshold: int = 36,
) -> Dict[str, object]:
    hamming_counts = Counter(
        edge_hamming_distance(problem, assignments[left], assignments[right])
        for left, right in edges
    )
    representative = representative_transition_circuits(problem, assignments, edges)
    representative_metrics = {
        str(hamming): {
            "logical": circuit_metrics(circuit),
            "transpiled": transpiled_metrics(circuit),
        }
        for hamming, circuit in representative.items()
    }

    transitions_per_qaoa_layer = len(edges)
    total_transitions = transitions_per_qaoa_layer * reps
    logical_depth_per_qaoa_layer = schedule_logical_depth_per_qaoa_layer(schedule, reps)

    full_schedule_metrics: Optional[Dict[str, object]] = None
    full_schedule_note = None
    if problem.num_variables <= 12 and total_transitions <= full_transpile_edge_threshold:
        circuit = QuantumCircuit(problem.num_variables)
        for layer in schedule.edge_layers:
            for left, right in layer:
                append_exact_transition(
                    circuit,
                    assignment_to_bits(problem, assignments[left]),
                    assignment_to_bits(problem, assignments[right]),
                    beta=0.123,
                )
        full_schedule_metrics = {
            "logical": circuit_metrics(circuit),
            "transpiled": transpiled_metrics(circuit),
        }
    else:
        full_schedule_note = (
            "Full exact transition-circuit transpilation skipped by threshold; "
            "representative transition transpilation plus logical schedule costs reported."
        )

    estimated_two_qubit_count = 0
    estimated_transpiled_depth = 0
    for hamming, count in hamming_counts.items():
        metrics = representative_metrics[str(hamming)]["transpiled"]
        estimated_two_qubit_count += int(metrics["two_qubit_gate_count"]) * count * reps
        estimated_transpiled_depth += int(metrics["depth"]) * count * reps

    return {
        "schedule": schedule.name,
        "num_qubits": problem.num_variables,
        "reps": reps,
        "transition_edge_count": len(edges),
        "transition_hamming_counts": dict(hamming_counts),
        "logical_transition_count": total_transitions,
        "logical_depth_per_qaoa_layer": logical_depth_per_qaoa_layer,
        "logical_depth_total": logical_depth_per_qaoa_layer * reps,
        "decomposition": (
            "Each legal transition is represented diagnostically as an exact computational-basis "
            "two-level RX: CNOT ladder over differing bits, one all-other-qubit controlled RX, "
            "then uncompute. Real-empty moves have Hamming distance 2; real-real swaps have "
            "Hamming distance 4."
        ),
        "representative_transition_metrics": representative_metrics,
        "estimated_transpiled_two_qubit_count": estimated_two_qubit_count,
        "estimated_serial_transpiled_depth": estimated_transpiled_depth,
        "full_schedule_metrics": full_schedule_metrics,
        "full_schedule_note": full_schedule_note,
    }
