"""Structured low-cost mixer candidates for placement.

The routines here do not replace the reference exact legal mixer. They provide
candidate transition graphs and resource estimates for the mixer-reduction
study.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from placement_core import Assignment, PlacementProblem, feasible_assignments
from run_qaoa_simulation import assignment_key


Edge = Tuple[int, int]
SiteEdge = Tuple[int, int]


@dataclass(frozen=True)
class SparseMixerCandidate:
    name: str
    topology: str
    site_edges: Tuple[SiteEdge, ...]
    feasible_edges: Tuple[Edge, ...]
    description: str


def site_graph_edges(problem: PlacementProblem, topology: str) -> Tuple[SiteEdge, ...]:
    """Return site-pair edges for a sparse occupant-swap topology."""

    m = len(problem.sites)
    if topology == "complete":
        return tuple((i, j) for i in range(m) for j in range(i + 1, m))
    if topology == "line":
        return tuple((i, i + 1) for i in range(m - 1))
    if topology == "ring":
        if m <= 2:
            return tuple((i, i + 1) for i in range(m - 1))
        return tuple((i, i + 1) for i in range(m - 1)) + ((m - 1, 0),)
    if topology == "grid":
        edges = []
        for i, (xi, yi) in enumerate(problem.sites):
            for j in range(i + 1, m):
                xj, yj = problem.sites[j]
                if abs(xi - xj) + abs(yi - yj) == 1:
                    edges.append((i, j))
        if not edges:
            raise ValueError("grid topology produced no edges; use line or ring for non-grid sites")
        return tuple(edges)
    raise ValueError(f"unsupported topology: {topology}")


def sparse_legal_edges(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    index_by_key: Mapping[Tuple[int, ...], int],
    site_edges: Sequence[SiteEdge],
) -> Tuple[Edge, ...]:
    """Build feasible-basis edges induced by swapping adjacent site occupants."""

    edges: set[Edge] = set()
    cells = tuple(problem.cells)

    for left_idx, assignment in enumerate(assignments):
        occupied = {assignment[cell]: cell for cell in cells}
        for site_a, site_b in site_edges:
            occ_a = occupied.get(site_a)
            occ_b = occupied.get(site_b)
            if occ_a is None and occ_b is None:
                continue
            moved = dict(assignment)
            if occ_a is None:
                moved[occ_b] = site_a
            elif occ_b is None:
                moved[occ_a] = site_b
            else:
                moved[occ_a], moved[occ_b] = moved[occ_b], moved[occ_a]
            right_idx = index_by_key[assignment_key(problem, moved)]
            if left_idx != right_idx:
                edges.add((left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx))
    return tuple(sorted(edges))


def make_sparse_candidate(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    index_by_key: Mapping[Tuple[int, ...], int],
    topology: str,
) -> SparseMixerCandidate:
    site_edges = site_graph_edges(problem, topology)
    feasible_edges = sparse_legal_edges(problem, assignments, index_by_key, site_edges)
    return SparseMixerCandidate(
        name=f"sparse_{topology}",
        topology=topology,
        site_edges=site_edges,
        feasible_edges=feasible_edges,
        description=(
            f"Legality-preserving sparse mixer induced by {topology} site-neighbor "
            "occupant swaps."
        ),
    )


def adjacency(num_nodes: int, edges: Sequence[Edge]) -> list[list[int]]:
    graph = [[] for _ in range(num_nodes)]
    for left, right in edges:
        graph[left].append(right)
        graph[right].append(left)
    return graph


def bfs(graph: Sequence[Sequence[int]], source: int) -> list[Optional[int]]:
    distances: list[Optional[int]] = [None] * len(graph)
    distances[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        base = distances[node]
        assert base is not None
        for neighbor in graph[node]:
            if distances[neighbor] is None:
                distances[neighbor] = base + 1
                queue.append(neighbor)
    return distances


def graph_metrics(num_nodes: int, edges: Sequence[Edge], exact_threshold: int = 1000) -> Dict[str, object]:
    degrees = [0] * num_nodes
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    metrics: Dict[str, object] = {
        "state_count": num_nodes,
        "edge_count": len(edges),
        "degree_min": min(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "degree_distribution": dict(Counter(degrees)),
        "exact_bfs_performed": num_nodes <= exact_threshold,
        "connected_components": None,
        "diameter": None,
    }
    if num_nodes > exact_threshold:
        return metrics

    graph = adjacency(num_nodes, edges)
    seen: set[int] = set()
    components = []
    diameter = 0
    for source in range(num_nodes):
        if source in seen:
            continue
        distances = bfs(graph, source)
        component = {idx for idx, distance in enumerate(distances) if distance is not None}
        seen.update(component)
        components.append(len(component))
        for node in component:
            node_distances = bfs(graph, node)
            finite = [distance for distance in node_distances if distance is not None]
            if finite:
                diameter = max(diameter, max(finite))
    metrics["connected_components"] = len(components)
    metrics["component_sizes"] = sorted(components, reverse=True)
    metrics["diameter"] = diameter
    return metrics


def apply_ordered_edges(state: np.ndarray, beta: float, edges: Sequence[Edge]) -> None:
    cos = np.cos(beta)
    minus_i_sin = -1j * np.sin(beta)
    for left, right in edges:
        a = state[left]
        b = state[right]
        state[left] = cos * a + minus_i_sin * b
        state[right] = minus_i_sin * a + cos * b


def sparse_qaoa_probabilities(
    params: Sequence[float],
    energies: np.ndarray,
    reps: int,
    initial_pos: int,
    edges: Sequence[Edge],
) -> np.ndarray:
    state = np.zeros(len(energies), dtype=np.complex128)
    state[initial_pos] = 1.0
    for layer in range(reps):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        state *= np.exp(-1j * gamma * energies)
        apply_ordered_edges(state, beta, edges)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def occupant_register_resource_estimate(
    problem: PlacementProblem,
    topology: str,
    reps: int,
    exchange_two_qubit_cost: int = 6,
) -> Dict[str, object]:
    """Estimate a structured local occupant-register exchange circuit.

    This is not a transpiled circuit. It is a transparent first-order estimate
    for a one-hot occupant-register architecture with one register per site and
    `n+1` occupant labels, where one label represents EMPTY. A local partial
    swap between two site registers is costed as pairwise exchanges over
    unordered occupant-label pairs.
    """

    n = len(problem.cells)
    m = len(problem.sites)
    labels = n + 1
    site_edges = site_graph_edges(problem, topology)
    label_pair_exchanges = math.comb(labels, 2)
    operations_per_layer = len(site_edges) * label_pair_exchanges
    return {
        "architecture": "one_hot_occupant_register_local_exchange",
        "topology": topology,
        "real_cells": n,
        "sites": m,
        "logical_qubits": m * labels,
        "site_edge_count": len(site_edges),
        "occupant_labels_per_site": labels,
        "label_pair_exchanges_per_site_edge": label_pair_exchanges,
        "primitive_exchange_operations_per_layer": operations_per_layer,
        "primitive_exchange_operations_total": operations_per_layer * reps,
        "assumed_two_qubit_cost_per_exchange": exchange_two_qubit_cost,
        "estimated_two_qubit_gate_count": operations_per_layer * reps * exchange_two_qubit_cost,
        "estimated_depth_lower_bound": reps * label_pair_exchanges,
        "note": (
            "First-order structured estimate, not backend transpilation. It assumes "
            "local site-register exchange channels can be decomposed with the stated "
            "two-qubit cost per occupant-label exchange."
        ),
    }


def complete_reference_counts(n: int, m: int, reps: int) -> Dict[str, int]:
    state_count = math.factorial(m) // math.factorial(m - n)
    degree = math.comb(n, 2) + n * (m - n)
    edge_count = state_count * degree // 2
    return {
        "state_count": state_count,
        "degree": degree,
        "edge_count": edge_count,
        "transition_applications": edge_count * reps,
    }


def feasible_basis(problem: PlacementProblem) -> tuple[list[Assignment], dict[Tuple[int, ...], int]]:
    assignments = list(feasible_assignments(problem))
    return assignments, {assignment_key(problem, assignment): idx for idx, assignment in enumerate(assignments)}
