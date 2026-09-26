"""Frozen exact-solvable benchmark generation for QAOA placement."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from placement_core import PlacementProblem, Site, brute_force, grid_sites, hpwl, random_problem
from run_robust_schedule_experiment import initialization_classes


SCHEMA_VERSION = "qaoa-placement-benchmark-v1"


@dataclass(frozen=True)
class BenchmarkSpec:
    family: str
    seed: int
    num_cells: int
    width: int
    height: int
    edge_probability: float
    min_weight: float = 1.0
    max_weight: float = 5.0


def line_sites(count: int) -> Tuple[Site, ...]:
    return tuple((idx, 0) for idx in range(count))


def l_shape_sites(width: int, height: int) -> Tuple[Site, ...]:
    sites = {(x, 0) for x in range(width)}
    sites.update({(0, y) for y in range(height)})
    return tuple(sorted(sites, key=lambda item: (item[1], item[0])))


def problem_from_spec(spec: BenchmarkSpec) -> PlacementProblem:
    problem = random_problem(
        seed=spec.seed,
        num_cells=spec.num_cells,
        width=spec.width,
        height=spec.height,
        edge_probability=spec.edge_probability,
        min_weight=spec.min_weight,
        max_weight=spec.max_weight,
    )
    if spec.family.endswith("_line"):
        sites = line_sites(spec.width * spec.height)
        return PlacementProblem(problem.cells, sites, problem.nets, problem.penalty)
    if spec.family.endswith("_lshape"):
        sites = l_shape_sites(spec.width, spec.height)
        if len(sites) < spec.num_cells:
            sites = grid_sites(spec.width, spec.height)
        return PlacementProblem(problem.cells, sites, problem.nets, problem.penalty)
    return problem


def default_specs(seed_start: int = 500, max_instances: int = 48) -> List[BenchmarkSpec]:
    families = [
        ("sparse_grid", 0.35, 3, 3, 2),
        ("dense_grid", 0.95, 3, 3, 2),
        ("sparse_grid", 0.35, 4, 3, 2),
        ("dense_grid", 0.95, 4, 3, 2),
        ("sparse_grid_line", 0.45, 3, 6, 1),
        ("dense_grid_line", 0.95, 4, 6, 1),
        ("sparse_grid_lshape", 0.45, 3, 4, 3),
        ("dense_grid_lshape", 0.90, 4, 4, 3),
    ]
    specs: List[BenchmarkSpec] = []
    seed = seed_start
    attempts = 0
    while len(specs) < max_instances and attempts < max_instances * 20:
        for family, edge_probability, num_cells, width, height in families:
            if len(specs) >= max_instances:
                break
            attempts += 1
            spec = BenchmarkSpec(family, seed, num_cells, width, height, edge_probability)
            if is_useful_instance(spec):
                specs.append(spec)
            seed += 1
    return specs


def is_useful_instance(spec: BenchmarkSpec) -> bool:
    problem = problem_from_spec(spec)
    exact = brute_force(problem)
    exact_hpwl = exact[0][0]
    degeneracy = sum(1 for value, _assignment in exact if value == exact_hpwl)
    state_count = math.factorial(len(problem.sites)) // math.factorial(len(problem.sites) - len(problem.cells))
    if degeneracy / state_count > 0.10:
        return False
    values = [value for value, _assignment in exact]
    if max(values) - min(values) < 1e-9:
        return False
    return True


def net_stats(problem: PlacementProblem) -> Dict[str, object]:
    possible = len(problem.cells) * (len(problem.cells) - 1) / 2
    weights = [weight for _left, _right, weight in problem.nets]
    return {
        "net_count": len(problem.nets),
        "net_density": len(problem.nets) / possible if possible else 0.0,
        "weight_min": min(weights) if weights else None,
        "weight_mean": float(np.mean(weights)) if weights else None,
        "weight_max": max(weights) if weights else None,
    }


def manifest_row(spec: BenchmarkSpec, instance_id: str) -> Dict[str, object]:
    problem = problem_from_spec(spec)
    start = time.time()
    exact = brute_force(problem)
    exact_runtime = time.time() - start
    exact_hpwl = exact[0][0]
    degeneracy = sum(1 for value, _assignment in exact if value == exact_hpwl)
    init_rows = []
    for init_mode, init_seed, assignment in initialization_classes(problem, seed=9000 + spec.seed):
        init_rows.append(
            {
                "init_mode": init_mode,
                "init_seed": init_seed,
                "assignment": assignment,
                "hpwl": hpwl(problem, assignment),
                "is_optimal": hpwl(problem, assignment) == exact_hpwl,
            }
        )
    return {
        "instance_id": instance_id,
        "family": spec.family,
        "seed": spec.seed,
        "num_cells": spec.num_cells,
        "num_sites": len(problem.sites),
        "width": spec.width,
        "height": spec.height,
        "geometry": problem.sites,
        "cells": problem.cells,
        "nets": problem.nets,
        "state_count": int(math.factorial(len(problem.sites)) / math.factorial(len(problem.sites) - len(problem.cells))),
        "num_qubits": problem.num_variables,
        "exact_hpwl": exact_hpwl,
        "degenerate_optima": degeneracy,
        "exact_runtime_seconds": exact_runtime,
        "initializations": init_rows,
        **net_stats(problem),
    }


def manifest_payload(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": digest,
        "instance_count": len(rows),
        "instances": list(rows),
    }


def write_manifest(path: Path, max_instances: int = 48, seed_start: int = 500) -> List[Dict[str, object]]:
    rows = [
        manifest_row(spec, instance_id=f"{idx:04d}_{spec.family}_{spec.seed}")
        for idx, spec in enumerate(default_specs(seed_start=seed_start, max_instances=max_instances))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(manifest_payload(rows), handle, indent=2)
    return rows


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    return list(payload["instances"])


def load_manifest_payload(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return manifest_payload(payload)
    return payload


def problem_from_manifest(row: Mapping[str, object]) -> PlacementProblem:
    return PlacementProblem(
        cells=tuple(row["cells"]),
        sites=tuple(tuple(site) for site in row["geometry"]),
        nets=tuple((left, right, float(weight)) for left, right, weight in row["nets"]),
        penalty=40.0,
    )
