"""Toy EDA placement utilities for the first QEDA QAOA simulation.

The instance is intentionally small enough to brute-force. That makes the
QAOA result easy to validate and keeps tomorrow's result scoped to algorithm
design rather than benchmark-scale claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Cell = str
Site = Tuple[int, int]
Net = Tuple[Cell, Cell, float]
Assignment = Dict[Cell, int]
Linear = Dict[int, float]
Quadratic = Dict[Tuple[int, int], float]


@dataclass(frozen=True)
class PlacementProblem:
    cells: Tuple[Cell, ...]
    sites: Tuple[Site, ...]
    nets: Tuple[Net, ...]
    penalty: float

    @property
    def num_variables(self) -> int:
        return len(self.cells) * len(self.sites)


@dataclass(frozen=True)
class QuboModel:
    linear: Linear
    quadratic: Quadratic
    offset: float
    num_variables: int


def default_problem() -> PlacementProblem:
    """Return a three-cell, four-site placement toy problem.

    The net weights make the instance nontrivial while remaining easy to
    inspect visually on a 2x2 grid.
    """

    return PlacementProblem(
        cells=("A", "B", "C"),
        sites=((0, 0), (1, 0), (0, 1), (1, 1)),
        nets=(
            ("A", "B", 3.0),
            ("A", "C", 2.0),
            ("B", "C", 1.5),
        ),
        penalty=25.0,
    )


def grid_sites(width: int, height: int) -> Tuple[Site, ...]:
    return tuple((x, y) for y in range(height) for x in range(width))


def random_problem(
    seed: int,
    num_cells: int = 4,
    width: int = 3,
    height: int = 2,
    edge_probability: float = 0.75,
    min_weight: float = 1.0,
    max_weight: float = 5.0,
    penalty: float = 40.0,
) -> PlacementProblem:
    """Create a deterministic random toy placement instance.

    Pairwise nets are used so brute-force HPWL remains easy to inspect. The
    generated graph is forced to be connected by adding a random spanning chain
    before optional extra edges.
    """

    if num_cells > width * height:
        raise ValueError("num_cells cannot exceed number of sites")

    rng = random.Random(seed)
    cells = tuple(chr(ord("A") + idx) for idx in range(num_cells))
    sites = grid_sites(width, height)
    nets = {}

    shuffled = list(cells)
    rng.shuffle(shuffled)
    for left, right in zip(shuffled, shuffled[1:]):
        weight = round(rng.uniform(min_weight, max_weight), 2)
        nets[tuple(sorted((left, right)))] = weight

    for left_idx, left in enumerate(cells):
        for right in cells[left_idx + 1 :]:
            key = tuple(sorted((left, right)))
            if key in nets:
                continue
            if rng.random() <= edge_probability:
                nets[key] = round(rng.uniform(min_weight, max_weight), 2)

    return PlacementProblem(
        cells=cells,
        sites=sites,
        nets=tuple((left, right, weight) for (left, right), weight in sorted(nets.items())),
        penalty=penalty,
    )


def variable_index(cell_idx: int, site_idx: int, num_sites: int) -> int:
    return cell_idx * num_sites + site_idx


def variable_name(problem: PlacementProblem, var_idx: int) -> str:
    cell_idx, site_idx = divmod(var_idx, len(problem.sites))
    return f"x_{problem.cells[cell_idx]}_{site_idx}"


def manhattan(a: Site, b: Site) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def add_linear(linear: Linear, idx: int, value: float) -> None:
    linear[idx] = linear.get(idx, 0.0) + value


def add_quadratic(quadratic: Quadratic, i: int, j: int, value: float) -> None:
    if i == j:
        raise ValueError("quadratic self-terms should be folded into linear terms")
    key = (i, j) if i < j else (j, i)
    quadratic[key] = quadratic.get(key, 0.0) + value


def build_qubo(problem: PlacementProblem) -> QuboModel:
    """Build a QUBO for weighted HPWL-like placement on a fixed grid.

    Terms:
    - weighted pairwise Manhattan distance for connected cell pairs,
    - exactly-one-site constraint for each cell,
    - at-most-one-cell constraint for each site.
    """

    linear: Linear = {}
    quadratic: Quadratic = {}
    offset = 0.0
    num_cells = len(problem.cells)
    num_sites = len(problem.sites)
    cell_to_idx = {cell: idx for idx, cell in enumerate(problem.cells)}

    for cell_idx in range(num_cells):
        indices = [
            variable_index(cell_idx, site_idx, num_sites)
            for site_idx in range(num_sites)
        ]
        offset += problem.penalty
        for idx in indices:
            add_linear(linear, idx, -problem.penalty)
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                add_quadratic(quadratic, left, right, 2.0 * problem.penalty)

    for site_idx in range(num_sites):
        indices = [
            variable_index(cell_idx, site_idx, num_sites)
            for cell_idx in range(num_cells)
        ]
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                add_quadratic(quadratic, left, right, 2.0 * problem.penalty)

    for left_cell, right_cell, weight in problem.nets:
        left_idx = cell_to_idx[left_cell]
        right_idx = cell_to_idx[right_cell]
        for left_site_idx, left_site in enumerate(problem.sites):
            for right_site_idx, right_site in enumerate(problem.sites):
                dist = manhattan(left_site, right_site)
                q_left = variable_index(left_idx, left_site_idx, num_sites)
                q_right = variable_index(right_idx, right_site_idx, num_sites)
                add_quadratic(quadratic, q_left, q_right, weight * dist)

    return QuboModel(linear, quadratic, offset, problem.num_variables)


def assignment_to_bits(problem: PlacementProblem, assignment: Mapping[Cell, int]) -> List[int]:
    bits = [0] * problem.num_variables
    num_sites = len(problem.sites)
    for cell_idx, cell in enumerate(problem.cells):
        site_idx = assignment[cell]
        bits[variable_index(cell_idx, site_idx, num_sites)] = 1
    return bits


def bits_to_assignment(problem: PlacementProblem, bits: Sequence[int]) -> Optional[Assignment]:
    num_sites = len(problem.sites)
    assignment: Assignment = {}

    for cell_idx, cell in enumerate(problem.cells):
        selected = [
            site_idx
            for site_idx in range(num_sites)
            if bits[variable_index(cell_idx, site_idx, num_sites)] == 1
        ]
        if len(selected) != 1:
            return None
        assignment[cell] = selected[0]

    if len(set(assignment.values())) != len(problem.cells):
        return None
    return assignment


def int_to_bits(value: int, num_bits: int) -> List[int]:
    return [(value >> idx) & 1 for idx in range(num_bits)]


def bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for idx, bit in enumerate(bits):
        value |= int(bit) << idx
    return value


def hpwl(problem: PlacementProblem, assignment: Mapping[Cell, int]) -> float:
    total = 0.0
    for left_cell, right_cell, weight in problem.nets:
        left_site = problem.sites[assignment[left_cell]]
        right_site = problem.sites[assignment[right_cell]]
        total += weight * manhattan(left_site, right_site)
    return total


def qubo_energy(model: QuboModel, bits: Sequence[int]) -> float:
    total = model.offset
    for idx, coeff in model.linear.items():
        total += coeff * bits[idx]
    for (left, right), coeff in model.quadratic.items():
        total += coeff * bits[left] * bits[right]
    return total


def feasible_assignments(problem: PlacementProblem) -> Iterable[Assignment]:
    for site_perm in permutations(range(len(problem.sites)), len(problem.cells)):
        yield dict(zip(problem.cells, site_perm))


def brute_force(problem: PlacementProblem) -> List[Tuple[float, Assignment]]:
    ranked = [(hpwl(problem, assignment), assignment) for assignment in feasible_assignments(problem)]
    return sorted(ranked, key=lambda item: item[0])


def random_baseline(problem: PlacementProblem, samples: int = 128, seed: int = 7) -> Dict[str, float]:
    import random

    rng = random.Random(seed)
    values = []
    site_indices = list(range(len(problem.sites)))
    for _ in range(samples):
        rng.shuffle(site_indices)
        assignment = dict(zip(problem.cells, site_indices))
        values.append(hpwl(problem, assignment))

    return {
        "samples": float(samples),
        "best": min(values),
        "mean": sum(values) / len(values),
        "worst": max(values),
    }


def ising_terms(model: QuboModel) -> Tuple[float, Dict[int, float], Dict[Tuple[int, int], float]]:
    """Convert QUBO energy over x in {0,1} to Ising terms over z in {-1,+1}.

    Uses x_i = (1 - z_i) / 2. Returned terms satisfy:
    E(z) = constant + sum_i h_i z_i + sum_ij J_ij z_i z_j.
    """

    constant = model.offset
    h: Dict[int, float] = {idx: 0.0 for idx in range(model.num_variables)}
    j: Dict[Tuple[int, int], float] = {}

    for idx, coeff in model.linear.items():
        constant += coeff / 2.0
        h[idx] -= coeff / 2.0

    for (left, right), coeff in model.quadratic.items():
        constant += coeff / 4.0
        h[left] -= coeff / 4.0
        h[right] -= coeff / 4.0
        j[(left, right)] = j.get((left, right), 0.0) + coeff / 4.0

    h = {idx: coeff for idx, coeff in h.items() if abs(coeff) > 1e-12}
    j = {pair: coeff for pair, coeff in j.items() if abs(coeff) > 1e-12}
    return constant, h, j


def summarize_problem(problem: PlacementProblem) -> Dict[str, object]:
    exact = brute_force(problem)
    model = build_qubo(problem)
    exact_assignment = exact[0][1]
    exact_bits = assignment_to_bits(problem, exact_assignment)

    return {
        "cells": problem.cells,
        "sites": problem.sites,
        "nets": problem.nets,
        "penalty": problem.penalty,
        "num_variables": problem.num_variables,
        "num_feasible_assignments": len(exact),
        "exact_hpwl": exact[0][0],
        "exact_assignment": exact_assignment,
        "exact_qubo_energy": qubo_energy(model, exact_bits),
        "random_baseline": random_baseline(problem),
    }


if __name__ == "__main__":
    problem = default_problem()
    summary = summarize_problem(problem)
    for key, value in summary.items():
        print(f"{key}: {value}")
