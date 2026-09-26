"""Cell-to-site binary encoding prototype for placement QAOA.

This module is a separate architecture branch from the promoted
site-occupant/register-swap implementation.  Each real cell owns a binary
register that stores its assigned site index.  Cost evaluation is direct in
this representation; legality-preserving movement is the difficult part.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit

from placement_core import Assignment, PlacementProblem, hpwl, manhattan
from phase_separator_optimization import (
    append_ordered_parity_network,
    order_parity_terms,
    parity_order_metrics,
    walsh_z_coefficients,
)
from register_partial_swap import append_phase_estimation_partial_swap


def site_register_width(problem: PlacementProblem) -> int:
    return math.ceil(math.log2(len(problem.sites)))


def cell_site_data_qubits(problem: PlacementProblem) -> int:
    return len(problem.cells) * site_register_width(problem)


def cell_site_qubit(problem: PlacementProblem, cell_idx: int, bit_idx: int) -> int:
    return cell_idx * site_register_width(problem) + bit_idx


def cell_register(problem: PlacementProblem, cell_idx: int) -> list[int]:
    return [cell_site_qubit(problem, cell_idx, bit) for bit in range(site_register_width(problem))]


def assignment_to_cell_site_bits(problem: PlacementProblem, assignment: Mapping[str, int]) -> list[int]:
    bits = [0] * cell_site_data_qubits(problem)
    for cell_idx, cell in enumerate(problem.cells):
        site_idx = int(assignment[cell])
        for bit in range(site_register_width(problem)):
            bits[cell_site_qubit(problem, cell_idx, bit)] = (site_idx >> bit) & 1
    return bits


def bits_to_cell_site_assignment(problem: PlacementProblem, bits: Sequence[int]) -> Assignment | None:
    assignment: Assignment = {}
    seen: set[int] = set()
    width = site_register_width(problem)
    for cell_idx, cell in enumerate(problem.cells):
        site_idx = 0
        for bit in range(width):
            site_idx |= int(bits[cell_site_qubit(problem, cell_idx, bit)]) << bit
        if site_idx >= len(problem.sites) or site_idx in seen:
            return None
        assignment[cell] = site_idx
        seen.add(site_idx)
    return assignment


def prepare_cell_site_basis_state(circuit: QuantumCircuit, problem: PlacementProblem, assignment: Mapping[str, int]) -> None:
    for qubit, bit in enumerate(assignment_to_cell_site_bits(problem, assignment)):
        if bit:
            circuit.x(qubit)


def distance_phase_values(problem: PlacementProblem, weight: float, gamma: float) -> list[float]:
    width = site_register_width(problem)
    dim = 2 ** (2 * width)
    values: list[float] = []
    mask = (1 << width) - 1
    for basis in range(dim):
        left = basis & mask
        right = (basis >> width) & mask
        if left < len(problem.sites) and right < len(problem.sites):
            distance = manhattan(problem.sites[left], problem.sites[right])
        else:
            distance = 0
        values.append(-float(gamma) * float(weight) * float(distance))
    return values


def parity_support(qubits: Sequence[int], mask: int) -> tuple[int, ...]:
    return tuple(qubits[idx] for idx in range(len(qubits)) if (mask >> idx) & 1)


def cell_site_parity_terms(problem: PlacementProblem, gamma: float) -> tuple[list[dict[str, Any]], float]:
    labels = {cell: idx for idx, cell in enumerate(problem.cells)}
    terms: list[dict[str, Any]] = []
    global_phase = 0.0
    term_id = 0
    for net_id, (left_cell, right_cell, weight) in enumerate(problem.nets):
        left_idx = labels[left_cell]
        right_idx = labels[right_cell]
        qubits = tuple(cell_register(problem, left_idx) + cell_register(problem, right_idx))
        coeffs = walsh_z_coefficients(distance_phase_values(problem, weight, gamma))
        for mask, coeff in enumerate(coeffs):
            coefficient = float(coeff)
            if abs(coefficient) < 1e-10:
                continue
            support = parity_support(qubits, mask)
            if not support:
                global_phase += coefficient
                continue
            terms.append(
                {
                    "term_id": term_id,
                    "net_id": net_id,
                    "left_cell": left_cell,
                    "right_cell": right_cell,
                    "weight": float(weight),
                    "mask": int(mask),
                    "coefficient": coefficient,
                    "qubits": qubits,
                    "support": support,
                    "support_size": len(support),
                }
            )
            term_id += 1
    return terms, global_phase


def append_cell_site_phase_separator(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    gamma: float,
    ordering_policy: str = "beam_search",
    cancel_adjacent_cx: bool = True,
) -> dict[str, int]:
    terms, global_phase = cell_site_parity_terms(problem, gamma)
    ordered = order_parity_terms(terms, ordering_policy)
    if global_phase:
        circuit.global_phase += global_phase
    append_ordered_parity_network(circuit, ordered, cancel_adjacent_cx=cancel_adjacent_cx)
    return {
        "parity_terms": len(terms),
        "global_phase_terms": int(abs(global_phase) > 1e-12),
        **parity_order_metrics(ordered),
    }


def cell_site_phase_circuit(
    problem: PlacementProblem,
    gamma: float = 0.17,
    ordering_policy: str = "beam_search",
    cancel_adjacent_cx: bool = True,
) -> QuantumCircuit:
    circuit = QuantumCircuit(cell_site_data_qubits(problem))
    append_cell_site_phase_separator(
        circuit,
        problem,
        gamma,
        ordering_policy=ordering_policy,
        cancel_adjacent_cx=cancel_adjacent_cx,
    )
    return circuit


def cell_site_swap_mixer_circuit(
    problem: PlacementProblem,
    beta: float = 0.23,
    topology: str = "complete",
    reps: int = 1,
) -> QuantumCircuit:
    data_qubits = cell_site_data_qubits(problem)
    ancilla = data_qubits
    circuit = QuantumCircuit(data_qubits + 1)
    for _ in range(reps):
        for left, right in cell_pair_edges(problem, topology):
            append_phase_estimation_partial_swap(
                circuit,
                cell_register(problem, left),
                cell_register(problem, right),
                ancilla,
                beta,
            )
    return circuit


def cell_pair_edges(problem: PlacementProblem, topology: str) -> tuple[tuple[int, int], ...]:
    n = len(problem.cells)
    if topology == "complete":
        return tuple((i, j) for i in range(n) for j in range(i + 1, n))
    if topology == "line":
        return tuple((i, i + 1) for i in range(n - 1))
    if topology == "ring":
        return tuple((i, i + 1) for i in range(n - 1)) + (((n - 1, 0),) if n > 2 else ())
    raise ValueError(topology)


def legal_assignments(problem: PlacementProblem) -> list[Assignment]:
    from itertools import permutations

    return [
        {cell: site for cell, site in zip(problem.cells, selected)}
        for selected in permutations(range(len(problem.sites)), len(problem.cells))
    ]


def assignment_key(problem: PlacementProblem, assignment: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(assignment[cell]) for cell in problem.cells)


def swap_only_edges(problem: PlacementProblem, assignments: Sequence[Assignment], topology: str) -> tuple[tuple[int, int], ...]:
    index = {assignment_key(problem, assignment): idx for idx, assignment in enumerate(assignments)}
    edges: set[tuple[int, int]] = set()
    for left_idx, assignment in enumerate(assignments):
        for cell_a, cell_b in cell_pair_edges(problem, topology):
            moved = dict(assignment)
            moved[problem.cells[cell_a]], moved[problem.cells[cell_b]] = moved[problem.cells[cell_b]], moved[problem.cells[cell_a]]
            right_idx = index[assignment_key(problem, moved)]
            if left_idx != right_idx:
                edges.add((left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx))
    return tuple(sorted(edges))


def move_edges(problem: PlacementProblem, assignments: Sequence[Assignment], site_edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    index = {assignment_key(problem, assignment): idx for idx, assignment in enumerate(assignments)}
    edges: set[tuple[int, int]] = set()
    site_neighbors = set(site_edges) | {(b, a) for a, b in site_edges}
    for left_idx, assignment in enumerate(assignments):
        occupied = set(assignment.values())
        for cell in problem.cells:
            source = assignment[cell]
            for a, b in site_neighbors:
                if a != source or b in occupied:
                    continue
                moved = dict(assignment)
                moved[cell] = b
                right_idx = index[assignment_key(problem, moved)]
                edges.add((left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx))
    return tuple(sorted(edges))


def graph_metrics(num_nodes: int, edges: Sequence[tuple[int, int]]) -> dict[str, int | bool]:
    adjacency = [[] for _ in range(num_nodes)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen = [False] * num_nodes
    components = 0
    diameter = 0
    for source in range(num_nodes):
        if seen[source]:
            continue
        components += 1
        queue = [source]
        seen[source] = True
        for node in queue:
            for nbr in adjacency[node]:
                if not seen[nbr]:
                    seen[nbr] = True
                    queue.append(nbr)
        for node in queue:
            distances = [-1] * num_nodes
            distances[node] = 0
            bfs = [node]
            for current in bfs:
                for nbr in adjacency[current]:
                    if distances[nbr] < 0:
                        distances[nbr] = distances[current] + 1
                        bfs.append(nbr)
            diameter = max(diameter, max(distances[idx] for idx in queue))
    return {
        "nodes": num_nodes,
        "edges": len(edges),
        "connected_components": components,
        "diameter": diameter,
        "connected": components == 1,
    }


def move_primitive_estimate(problem: PlacementProblem, site_topology_edges: Sequence[tuple[int, int]]) -> dict[str, int]:
    """Conservative component estimate for exact cell-to-empty-site moves.

    For each directed local move, estimate:
    - one source-register equality MCX;
    - one target-vacancy equality check for each other cell;
    - one controlled two-level update on the moving cell register.

    The two-level update cost is represented as another MCX-equivalent term.
    This is intentionally conservative and is used only for architecture
    feasibility, not as a promoted circuit.
    """

    n = len(problem.cells)
    directed_site_edges = 2 * len(site_topology_edges)
    moves_per_layer = n * directed_site_edges
    mcx_equiv_per_move = 1 + (n - 1) + 1
    return {
        "moves_per_layer": moves_per_layer,
        "mcx_equivalent_per_move": mcx_equiv_per_move,
        "mcx_equivalent_per_layer": moves_per_layer * mcx_equiv_per_move,
        "cx_per_layer_at_14cx_mcx": moves_per_layer * mcx_equiv_per_move * 14,
    }

