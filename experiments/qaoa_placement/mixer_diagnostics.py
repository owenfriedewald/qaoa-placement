"""Diagnostics for collision-free placement mixer schedules.

The routines in this module are intentionally separated from the scalable
statevector mixer. Exact graph enumeration and BFS are useful for toy
instances, but should remain diagnostic-only.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from placement_core import Assignment, PlacementProblem, assignment_to_bits, hpwl


Edge = Tuple[int, int]


@dataclass(frozen=True)
class MixerSchedule:
    name: str
    edge_layers: Tuple[Tuple[Edge, ...], ...]
    description: str
    seed: Optional[int] = None

    @property
    def ordered_edges(self) -> Tuple[Edge, ...]:
        return tuple(edge for layer in self.edge_layers for edge in layer)


def assignment_key(problem: PlacementProblem, assignment: Mapping[str, int]) -> Tuple[int, ...]:
    return tuple(assignment[cell] for cell in problem.cells)


def edge_hamming_distance(problem: PlacementProblem, left: Assignment, right: Assignment) -> int:
    left_bits = assignment_to_bits(problem, left)
    right_bits = assignment_to_bits(problem, right)
    return sum(1 for a, b in zip(left_bits, right_bits) if a != b)


def classify_edge(problem: PlacementProblem, left: Assignment, right: Assignment) -> str:
    changed = [cell for cell in problem.cells if left[cell] != right[cell]]
    if len(changed) == 1:
        return "real-empty"
    if len(changed) == 2:
        return "real-real"
    return f"{len(changed)}-cell"


def build_adjacency(num_nodes: int, edges: Sequence[Edge]) -> List[List[int]]:
    adjacency = [[] for _ in range(num_nodes)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return adjacency


def bfs_distances(adjacency: Sequence[Sequence[int]], source: int) -> List[Optional[int]]:
    distances: List[Optional[int]] = [None] * len(adjacency)
    distances[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        base_distance = distances[node]
        assert base_distance is not None
        for neighbor in adjacency[node]:
            if distances[neighbor] is None:
                distances[neighbor] = base_distance + 1
                queue.append(neighbor)
    return distances


def graph_audit(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    edges: Sequence[Edge],
    threshold: int,
) -> Dict[str, object]:
    """Return exact graph diagnostics when below threshold.

    Above threshold, only local degree and edge-type diagnostics are returned.
    This keeps graph enumeration/BFS from becoming a hidden dependency of the
    mixer implementation.
    """

    degrees = [0] * len(assignments)
    edge_types = Counter()
    hamming = Counter()
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
        edge_types[classify_edge(problem, assignments[left], assignments[right])] += 1
        hamming[edge_hamming_distance(problem, assignments[left], assignments[right])] += 1

    audit: Dict[str, object] = {
        "state_count": len(assignments),
        "edge_count": len(edges),
        "degree_min": min(degrees) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
        "degree_distribution": dict(Counter(degrees)),
        "edge_type_counts": dict(edge_types),
        "edge_hamming_counts": dict(hamming),
        "exact_bfs_performed": len(assignments) <= threshold,
        "connected_components": None,
        "diameter": None,
        "diameter_note": None,
    }

    if len(assignments) > threshold:
        audit["diameter_note"] = f"Skipped exact BFS because state_count>{threshold}"
        return audit

    adjacency = build_adjacency(len(assignments), edges)
    seen = set()
    components = []
    diameter = 0
    for source in range(len(assignments)):
        if source in seen:
            continue
        distances = bfs_distances(adjacency, source)
        component = {idx for idx, distance in enumerate(distances) if distance is not None}
        seen.update(component)
        components.append(len(component))
        for node in component:
            node_distances = bfs_distances(adjacency, node)
            finite = [distance for distance in node_distances if distance is not None]
            if finite:
                diameter = max(diameter, max(finite))

    audit["connected_components"] = len(components)
    audit["component_sizes"] = sorted(components, reverse=True)
    audit["diameter"] = diameter
    return audit


def make_fixed_schedule(edges: Sequence[Edge]) -> MixerSchedule:
    return MixerSchedule(
        name="fixed",
        edge_layers=(tuple(edges),),
        description="Current sorted ordered product over all legal pairwise occupant swaps.",
    )


def make_reversed_schedule(edges: Sequence[Edge]) -> MixerSchedule:
    return MixerSchedule(
        name="reversed",
        edge_layers=(tuple(reversed(edges)),),
        description="Same transition edge set as fixed, with product order reversed.",
    )


def make_random_layer_schedule(edges: Sequence[Edge], reps: int, seed: int) -> MixerSchedule:
    rng = random.Random(seed)
    layers = []
    base_edges = list(edges)
    for _ in range(reps):
        shuffled = list(base_edges)
        rng.shuffle(shuffled)
        layers.append(tuple(shuffled))
    return MixerSchedule(
        name="random_per_qaoa_layer",
        edge_layers=tuple(layers),
        description="Same transition edge set, independently shuffled for each QAOA mixer layer.",
        seed=seed,
    )


def greedy_edge_coloring(edges: Sequence[Edge]) -> Tuple[Tuple[Edge, ...], ...]:
    """Greedy coloring where each color class is a matching."""

    color_layers: List[List[Edge]] = []
    used_nodes_by_color: List[set[int]] = []
    for edge in edges:
        left, right = edge
        placed = False
        for color_idx, used_nodes in enumerate(used_nodes_by_color):
            if left not in used_nodes and right not in used_nodes:
                color_layers[color_idx].append(edge)
                used_nodes.update(edge)
                placed = True
                break
        if not placed:
            color_layers.append([edge])
            used_nodes_by_color.append(set(edge))
    return tuple(tuple(layer) for layer in color_layers)


def make_edge_colored_schedule(edges: Sequence[Edge], reps: int) -> MixerSchedule:
    colors = list(greedy_edge_coloring(edges))
    layers = []
    for layer_idx in range(reps):
        offset = layer_idx % max(len(colors), 1)
        layers.extend(colors[offset:] + colors[:offset])
    return MixerSchedule(
        name="edge_colored_alternating",
        edge_layers=tuple(layers),
        description="Greedy matching/color layers; color order rotates across QAOA layers.",
    )


def schedule_for_layer(schedule: MixerSchedule, layer_idx: int, reps: int) -> Tuple[Edge, ...]:
    if schedule.name == "random_per_qaoa_layer":
        return schedule.edge_layers[layer_idx]
    if schedule.name == "edge_colored_alternating":
        colors_per_qaoa_layer = len(schedule.edge_layers) // reps
        start = layer_idx * colors_per_qaoa_layer
        stop = start + colors_per_qaoa_layer
        return tuple(edge for color in schedule.edge_layers[start:stop] for edge in color)
    return schedule.ordered_edges


def schedule_logical_depth_per_qaoa_layer(schedule: MixerSchedule, reps: int) -> int:
    if schedule.name == "edge_colored_alternating":
        return len(schedule.edge_layers) // reps
    return len(schedule.ordered_edges)


def min_optimum_distance(distances: Sequence[Optional[int]], optimal_positions: Iterable[int]) -> Optional[int]:
    values = [distances[pos] for pos in optimal_positions if distances[pos] is not None]
    if not values:
        return None
    return int(min(values))


def ideal_mixer_hamiltonian_eigen_residual(initial_pos: int, edge_count: int, edges: Sequence[Edge]) -> float:
    """Residual ||H|s>-lambda|s>|| for H=sum_edges X_edge.

    For basis state |s>, lambda is zero because this off-diagonal adjacency
    Hamiltonian has no diagonal term. The residual is therefore sqrt(degree(s)).
    """

    degree = sum(1 for left, right in edges if left == initial_pos or right == initial_pos)
    _ = edge_count
    return float(math.sqrt(degree))


def ordered_unitary_retained_probability(
    num_states: int,
    initial_pos: int,
    edges: Sequence[Edge],
    beta: float,
) -> float:
    state = np.zeros(num_states, dtype=np.complex128)
    state[initial_pos] = 1.0
    cos = np.cos(beta)
    minus_i_sin = -1j * np.sin(beta)
    for left, right in edges:
        a = state[left]
        b = state[right]
        state[left] = cos * a + minus_i_sin * b
        state[right] = minus_i_sin * a + cos * b
    return float(abs(state[initial_pos]) ** 2)


def ideal_hamiltonian_retained_probability(
    num_states: int,
    initial_pos: int,
    edges: Sequence[Edge],
    beta: float,
    threshold: int,
) -> Optional[float]:
    if num_states > threshold:
        return None
    hamiltonian = np.zeros((num_states, num_states), dtype=np.complex128)
    for left, right in edges:
        hamiltonian[left, right] = 1.0
        hamiltonian[right, left] = 1.0
    state = np.zeros(num_states, dtype=np.complex128)
    state[initial_pos] = 1.0
    evolved = expm(-1j * beta * hamiltonian) @ state
    return float(abs(evolved[initial_pos]) ** 2)


def summarize_localization(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    probs: np.ndarray,
    counts: Mapping[int, int],
    distances: Sequence[Optional[int]],
    initial_pos: int,
    optimal_positions: Sequence[int],
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    total_counts = sum(counts.values())
    per_state = []
    mass_by_distance: Dict[int, float] = Counter()
    counts_by_distance: Counter[int] = Counter()
    hpwl_by_distance: Dict[int, List[float]] = {}

    for idx, assignment in enumerate(assignments):
        distance = distances[idx]
        if distance is None:
            continue
        cost = hpwl(problem, assignment)
        count = int(counts.get(idx, 0))
        probability = float(probs[idx])
        mass_by_distance[int(distance)] += probability
        counts_by_distance[int(distance)] += count
        hpwl_by_distance.setdefault(int(distance), []).append(cost)
        displaced = sum(1 for cell in problem.cells if assignment[cell] != assignments[initial_pos][cell])
        per_state.append(
            {
                "subspace_index": idx,
                "assignment": dict(assignment),
                "swap_distance": int(distance),
                "displaced_cells": displaced,
                "assignment_hamming_distance": displaced,
                "hpwl": cost,
                "probability": probability,
                "count": count,
                "is_initial": idx == initial_pos,
                "is_optimal": idx in optimal_positions,
            }
        )

    distance_rows = []
    for distance in sorted(mass_by_distance):
        values = hpwl_by_distance[distance]
        distance_rows.append(
            {
                "swap_distance": distance,
                "probability_mass": float(mass_by_distance[distance]),
                "sample_count": int(counts_by_distance[distance]),
                "sample_fraction": counts_by_distance[distance] / max(total_counts, 1),
                "best_hpwl": min(values),
                "mean_hpwl_all_states_at_distance": float(np.mean(values)),
            }
        )

    mean_sampled_distance = sum(row["swap_distance"] * row["probability_mass"] for row in distance_rows)
    sampled_distances = [
        distance
        for distance, count in counts_by_distance.items()
        for _ in range(count)
    ]
    summary = {
        "mean_sampled_swap_distance": float(mean_sampled_distance),
        "median_sampled_swap_distance": float(np.median(sampled_distances)) if sampled_distances else None,
        "probability_distance_0": float(mass_by_distance.get(0, 0.0)),
        "sample_fraction_distance_0": counts_by_distance.get(0, 0) / max(total_counts, 1),
        "optimal_sample_probability": float(sum(probs[pos] for pos in optimal_positions)),
        "optimal_sample_count": int(sum(counts.get(pos, 0) for pos in optimal_positions)),
    }
    return summary, distance_rows, per_state
