"""Token/permutation cell-to-site encoding with explicit EMPTY tokens.

This architecture represents a full permutation of site labels over token
registers.  The first ``n`` tokens are real cells and the remaining ``m-n``
tokens are distinguishable EMPTY tokens.  Register swaps preserve the
permutation subspace and allow real cells to move into empty sites.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import permutations
import math
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from cell_site_encoding import distance_phase_values, parity_support
from phase_separator_optimization import (
    append_ordered_parity_network,
    order_parity_terms,
    parity_order_metrics,
    walsh_z_coefficients,
)
from placement_core import Assignment, PlacementProblem, hpwl
from register_partial_swap import append_phase_estimation_partial_swap


def token_register_width(problem: PlacementProblem) -> int:
    return math.ceil(math.log2(len(problem.sites)))


def token_count(problem: PlacementProblem) -> int:
    return len(problem.sites)


def empty_token_count(problem: PlacementProblem) -> int:
    return len(problem.sites) - len(problem.cells)


def token_data_qubits(problem: PlacementProblem) -> int:
    return token_count(problem) * token_register_width(problem)


def token_qubit(problem: PlacementProblem, token_idx: int, bit_idx: int) -> int:
    return token_idx * token_register_width(problem) + bit_idx


def token_register(problem: PlacementProblem, token_idx: int) -> list[int]:
    return [token_qubit(problem, token_idx, bit) for bit in range(token_register_width(problem))]


def token_names(problem: PlacementProblem) -> tuple[str, ...]:
    empties = tuple(f"EMPTY_{idx}" for idx in range(empty_token_count(problem)))
    return tuple(problem.cells) + empties


def assignment_to_token_sites(problem: PlacementProblem, assignment: Mapping[str, int]) -> tuple[int, ...]:
    occupied = set(int(assignment[cell]) for cell in problem.cells)
    empty_sites = [site for site in range(len(problem.sites)) if site not in occupied]
    return tuple(int(assignment[cell]) for cell in problem.cells) + tuple(empty_sites)


def token_sites_to_assignment(problem: PlacementProblem, token_sites: Sequence[int]) -> Assignment | None:
    m = len(problem.sites)
    if len(token_sites) != m:
        return None
    if any(site < 0 or site >= m for site in token_sites):
        return None
    if len(set(int(site) for site in token_sites)) != m:
        return None
    return {cell: int(token_sites[idx]) for idx, cell in enumerate(problem.cells)}


def token_sites_to_bits(problem: PlacementProblem, token_sites: Sequence[int]) -> list[int]:
    bits = [0] * token_data_qubits(problem)
    width = token_register_width(problem)
    for token_idx, site_idx in enumerate(token_sites):
        for bit in range(width):
            bits[token_qubit(problem, token_idx, bit)] = (int(site_idx) >> bit) & 1
    return bits


def bits_to_token_sites(problem: PlacementProblem, bits: Sequence[int]) -> tuple[int, ...] | None:
    width = token_register_width(problem)
    sites: list[int] = []
    for token_idx in range(token_count(problem)):
        site = 0
        for bit in range(width):
            site |= int(bits[token_qubit(problem, token_idx, bit)]) << bit
        sites.append(site)
    if any(site >= len(problem.sites) for site in sites):
        return None
    return tuple(sites)


def bits_to_assignment(problem: PlacementProblem, bits: Sequence[int]) -> Assignment | None:
    sites = bits_to_token_sites(problem, bits)
    if sites is None:
        return None
    return token_sites_to_assignment(problem, sites)


def prepare_token_permutation_state(circuit: QuantumCircuit, problem: PlacementProblem, assignment: Mapping[str, int]) -> None:
    for qubit, bit in enumerate(token_sites_to_bits(problem, assignment_to_token_sites(problem, assignment))):
        if bit:
            circuit.x(qubit)


def legal_token_permutations(problem: PlacementProblem) -> list[tuple[int, ...]]:
    return [tuple(order) for order in permutations(range(len(problem.sites)), token_count(problem))]


def real_assignment_key(problem: PlacementProblem, assignment: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(assignment[cell]) for cell in problem.cells)


def token_graph_edges(problem: PlacementProblem, graph: str) -> tuple[tuple[int, int], ...]:
    n = len(problem.cells)
    m = len(problem.sites)
    if graph == "complete":
        return tuple((i, j) for i in range(m) for j in range(i + 1, m))
    if graph == "line":
        return tuple((i, i + 1) for i in range(m - 1))
    if graph == "ring":
        return tuple((i, i + 1) for i in range(m - 1)) + (((m - 1, 0),) if m > 2 else ())
    if graph == "empty_star":
        empties = range(n, m)
        return tuple((real, empty) for empty in empties for real in range(n))
    if graph == "real_empty_priority":
        edges = set(token_graph_edges(problem, "empty_star"))
        edges.update((i, i + 1) for i in range(n - 1))
        edges.update((i, i + 1) for i in range(n, m - 1))
        return tuple(sorted(edges))
    raise ValueError(graph)


def token_permutation_graph_metrics(problem: PlacementProblem, graph: str) -> dict[str, object]:
    states = legal_token_permutations(problem)
    index = {state: idx for idx, state in enumerate(states)}
    edges: set[tuple[int, int]] = set()
    for left_idx, state in enumerate(states):
        state_list = list(state)
        for token_a, token_b in token_graph_edges(problem, graph):
            moved = state_list[:]
            moved[token_a], moved[token_b] = moved[token_b], moved[token_a]
            right_idx = index[tuple(moved)]
            edges.add((left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx))
    adjacency = [[] for _ in states]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen: set[int] = set()
    components = 0
    diameter = 0
    for source in range(len(states)):
        if source in seen:
            continue
        components += 1
        queue = [source]
        seen.add(source)
        for node in queue:
            for nbr in adjacency[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        for node in queue:
            dist = bfs(adjacency, node)
            finite = [value for value in dist if value >= 0]
            diameter = max(diameter, max(finite, default=0))
    return {
        "graph": graph,
        "states": len(states),
        "edges": len(edges),
        "connected_components": components,
        "connected": components == 1,
        "diameter": diameter,
        "token_edges_per_layer": len(token_graph_edges(problem, graph)),
        "empty_degeneracy": math.factorial(empty_token_count(problem)),
    }


def bfs(adjacency: Sequence[Sequence[int]], source: int) -> list[int]:
    dist = [-1] * len(adjacency)
    dist[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for nbr in adjacency[node]:
            if dist[nbr] < 0:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def token_phase_terms(problem: PlacementProblem, gamma: float) -> tuple[list[dict[str, Any]], float]:
    labels = {cell: idx for idx, cell in enumerate(problem.cells)}
    terms: list[dict[str, Any]] = []
    global_phase = 0.0
    term_id = 0
    for net_id, (left_cell, right_cell, weight) in enumerate(problem.nets):
        left_idx = labels[left_cell]
        right_idx = labels[right_cell]
        qubits = tuple(token_register(problem, left_idx) + token_register(problem, right_idx))
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


def truncate_phase_terms(
    terms: Sequence[Mapping[str, Any]],
    min_abs_coefficient: float = 0.0,
    keep_largest: int | None = None,
) -> list[dict[str, Any]]:
    """Return a deterministic subset of phase terms after coefficient pruning."""

    filtered = [dict(term) for term in terms if abs(float(term["coefficient"])) >= float(min_abs_coefficient)]
    if keep_largest is not None:
        filtered = sorted(
            filtered,
            key=lambda term: (-abs(float(term["coefficient"])), int(term["term_id"])),
        )[: int(keep_largest)]
    return sorted(filtered, key=lambda term: int(term["term_id"]))


def append_token_phase_separator(
    circuit: QuantumCircuit,
    problem: PlacementProblem,
    gamma: float,
    ordering_policy: str = "beam_search",
    cancel_adjacent_cx: bool = True,
    min_abs_coefficient: float = 0.0,
    keep_largest: int | None = None,
) -> dict[str, int]:
    terms, global_phase = token_phase_terms(problem, gamma)
    original_terms = len(terms)
    original_l1 = float(sum(abs(float(term["coefficient"])) for term in terms))
    terms = truncate_phase_terms(terms, min_abs_coefficient=min_abs_coefficient, keep_largest=keep_largest)
    kept_l1 = float(sum(abs(float(term["coefficient"])) for term in terms))
    ordered = order_parity_terms(terms, ordering_policy)
    if global_phase:
        circuit.global_phase += global_phase
    append_ordered_parity_network(circuit, ordered, cancel_adjacent_cx=cancel_adjacent_cx)
    return {
        "parity_terms": len(terms),
        "original_parity_terms": original_terms,
        "dropped_parity_terms": original_terms - len(terms),
        "phase_l1_kept": kept_l1,
        "phase_l1_dropped": original_l1 - kept_l1,
        "min_abs_coefficient": float(min_abs_coefficient),
        "keep_largest": "" if keep_largest is None else int(keep_largest),
        **parity_order_metrics(ordered),
    }


def token_phase_circuit(
    problem: PlacementProblem,
    gamma: float,
    ordering_policy: str = "beam_search",
    min_abs_coefficient: float = 0.0,
    keep_largest: int | None = None,
) -> QuantumCircuit:
    circuit = QuantumCircuit(token_data_qubits(problem))
    append_token_phase_separator(
        circuit,
        problem,
        gamma,
        ordering_policy=ordering_policy,
        cancel_adjacent_cx=True,
        min_abs_coefficient=min_abs_coefficient,
        keep_largest=keep_largest,
    )
    return circuit


def token_mixer_circuit(problem: PlacementProblem, beta: float, graph: str, reps: int = 1) -> QuantumCircuit:
    data = token_data_qubits(problem)
    ancilla = data
    circuit = QuantumCircuit(data + 1)
    width = token_register_width(problem)
    for _ in range(reps):
        for token_a, token_b in token_graph_edges(problem, graph):
            reg_a = [token_qubit(problem, token_a, bit) for bit in range(width)]
            reg_b = [token_qubit(problem, token_b, bit) for bit in range(width)]
            append_phase_estimation_partial_swap(circuit, reg_a, reg_b, ancilla, beta)
    return circuit


def token_qaoa_circuit(
    problem: PlacementProblem,
    gamma: float,
    beta: float,
    graph: str,
    p: int = 1,
    include_initial_state: bool = False,
    initial_assignment: Mapping[str, int] | None = None,
) -> QuantumCircuit:
    data = token_data_qubits(problem)
    circuit = QuantumCircuit(data + 1)
    if include_initial_state:
        if initial_assignment is None:
            initial_assignment = {cell: idx for idx, cell in enumerate(problem.cells)}
        prepare_token_permutation_state(circuit, problem, initial_assignment)
    for _layer in range(p):
        phase = token_phase_circuit(problem, gamma)
        circuit.compose(phase, qubits=range(data), inplace=True)
        mixer = token_mixer_circuit(problem, beta, graph, reps=1)
        circuit.compose(mixer, qubits=range(data + 1), inplace=True)
    return circuit


def aggregate_probabilities_by_assignment(problem: PlacementProblem, statevector: Statevector) -> dict[tuple[int, ...], float]:
    data = token_data_qubits(problem)
    probs: dict[tuple[int, ...], float] = defaultdict(float)
    for index, probability in enumerate(statevector.probabilities()):
        if probability < 1e-12:
            continue
        if index >> data:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(data)]
        assignment = bits_to_assignment(problem, bits)
        if assignment is None:
            continue
        probs[real_assignment_key(problem, assignment)] += float(probability)
    return dict(probs)


def valid_permutation_probability(problem: PlacementProblem, statevector: Statevector) -> float:
    data = token_data_qubits(problem)
    total = 0.0
    for index, probability in enumerate(statevector.probabilities()):
        if probability < 1e-12 or index >> data:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(data)]
        sites = bits_to_token_sites(problem, bits)
        if sites is not None and token_sites_to_assignment(problem, sites) is not None:
            total += float(probability)
    return total


def expected_empty_degeneracy(problem: PlacementProblem) -> int:
    return math.factorial(empty_token_count(problem))
