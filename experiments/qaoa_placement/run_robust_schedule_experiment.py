"""Robust schedule ablation for QAOA placement mixers.

This second-stage runner validates metric definitions before comparing
collision-free ordered-product schedules against row-constrained XY. It
preserves prior outputs by writing to a new directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

from mixer_circuit_costs import schedule_costs, transpiled_metrics
from mixer_diagnostics import (
    Edge,
    MixerSchedule,
    bfs_distances,
    build_adjacency,
    graph_audit,
    make_edge_colored_schedule,
    make_fixed_schedule,
    make_random_layer_schedule,
    make_reversed_schedule,
    schedule_for_layer,
)
from placement_core import (
    Assignment,
    PlacementProblem,
    assignment_to_bits,
    brute_force,
    build_qubo,
    feasible_assignments,
    hpwl,
    ising_terms,
    qubo_energy,
    random_problem,
)
from run_qaoa_simulation import (
    assignment_key,
    cell_xy_mixer_operator,
    collision_free_cost_energies,
    collision_free_swap_edges,
    constrained_cost_energies,
    expectation_from_probs,
    sampled_distribution,
    sparse_pauli_from_ising,
)


MetricRow = Dict[str, object]
COST_CACHE: Dict[Tuple[object, ...], Mapping[str, object]] = {}


def write_csv(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def bootstrap_ci(values: Sequence[float], seed: int = 123, draws: int = 1000) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(draws)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def init_params(reps: int, seed: int, jitter: float = 0.08) -> np.ndarray:
    values = []
    for layer in range(reps):
        values.extend([math.pi / (layer + 1), math.pi / (8 * (layer + 1))])
    rng = np.random.default_rng(seed)
    return np.asarray(values, dtype=float) + rng.normal(0.0, jitter, size=2 * reps)


def transferred_params(previous: Sequence[float], reps: int, seed: int) -> np.ndarray:
    if len(previous) != 2 * (reps - 1):
        return init_params(reps, seed)
    extension = [previous[-2] / 2.0, previous[-1] / 2.0]
    return np.asarray(list(previous) + extension, dtype=float)


def apply_ordered_swap_mixer(state: np.ndarray, beta: float, edges: Sequence[Edge]) -> None:
    cos = np.cos(beta)
    minus_i_sin = -1j * np.sin(beta)
    for left, right in edges:
        a = state[left]
        b = state[right]
        state[left] = cos * a + minus_i_sin * b
        state[right] = minus_i_sin * a + cos * b


def legal_qaoa_probs(
    params: Sequence[float],
    energies: np.ndarray,
    reps: int,
    initial_pos: int,
    schedule: MixerSchedule,
) -> np.ndarray:
    state = np.zeros(len(energies), dtype=np.complex128)
    state[initial_pos] = 1.0
    for layer in range(reps):
        state *= np.exp(-1j * params[2 * layer] * energies)
        apply_ordered_swap_mixer(state, params[2 * layer + 1], schedule_for_layer(schedule, layer, reps))
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def apply_cell_xy_mixer(state: np.ndarray, beta: float, problem: PlacementProblem, assignments: Sequence[Assignment], index_by_key: Mapping[Tuple[int, ...], int]) -> None:
    num_sites = len(problem.sites)
    theta = 2.0 * beta
    cos = np.cos(theta)
    minus_i_sin = -1j * np.sin(theta)
    for cell in problem.cells:
        for left_site in range(num_sites):
            for right_site in range(left_site + 1, num_sites):
                visited = set()
                for left_pos, assignment in enumerate(assignments):
                    if left_pos in visited or assignment[cell] != left_site:
                        continue
                    moved = dict(assignment)
                    moved[cell] = right_site
                    right_pos = index_by_key[assignment_key(problem, moved)]
                    visited.add(left_pos)
                    visited.add(right_pos)
                    a = state[left_pos]
                    b = state[right_pos]
                    state[left_pos] = cos * a + minus_i_sin * b
                    state[right_pos] = minus_i_sin * a + cos * b


def row_xy_qaoa_probs(
    params: Sequence[float],
    energies: np.ndarray,
    reps: int,
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    index_by_key: Mapping[Tuple[int, ...], int],
    initial_pos: int,
) -> np.ndarray:
    state = np.zeros(len(assignments), dtype=np.complex128)
    state[initial_pos] = 1.0
    for layer in range(reps):
        state *= np.exp(-1j * params[2 * layer] * energies)
        apply_cell_xy_mixer(state, params[2 * layer + 1], problem, assignments, index_by_key)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def weighted_degree_order(problem: PlacementProblem) -> List[str]:
    weights = {cell: 0.0 for cell in problem.cells}
    for left, right, weight in problem.nets:
        weights[left] += weight
        weights[right] += weight
    return sorted(problem.cells, key=lambda cell: (-weights[cell], cell))


def greedy_initial_assignment(problem: PlacementProblem) -> Assignment:
    ordered_cells = weighted_degree_order(problem)
    available = set(range(len(problem.sites)))
    assignment: Assignment = {}
    center_x = sum(site[0] for site in problem.sites) / len(problem.sites)
    center_y = sum(site[1] for site in problem.sites) / len(problem.sites)
    first_site = min(available, key=lambda idx: ((problem.sites[idx][0] - center_x) ** 2 + (problem.sites[idx][1] - center_y) ** 2, idx))
    assignment[ordered_cells[0]] = first_site
    available.remove(first_site)

    net_weight = {}
    for left, right, weight in problem.nets:
        net_weight[(left, right)] = weight
        net_weight[(right, left)] = weight

    for cell in ordered_cells[1:]:
        def score(site_idx: int) -> Tuple[float, int]:
            total = 0.0
            site = problem.sites[site_idx]
            for placed_cell, placed_site_idx in assignment.items():
                weight = net_weight.get((cell, placed_cell), 0.1)
                placed_site = problem.sites[placed_site_idx]
                total += weight * (abs(site[0] - placed_site[0]) + abs(site[1] - placed_site[1]))
            return total, site_idx

        chosen = min(available, key=score)
        assignment[cell] = chosen
        available.remove(chosen)

    return {cell: assignment[cell] for cell in problem.cells}


def initialization_classes(problem: PlacementProblem, seed: int) -> List[Tuple[str, int, Assignment]]:
    exact = brute_force(problem)
    rng = random.Random(seed)
    random_assignment = rng.choice(list(feasible_assignments(problem)))
    return [
        ("deterministic", -1, next(feasible_assignments(problem))),
        ("poor", -2, exact[-1][1]),
        ("random", seed, random_assignment),
        ("greedy", -3, greedy_initial_assignment(problem)),
    ]


def feasible_key_to_distance(problem: PlacementProblem, feasible_distances: Sequence[Optional[int]], feasible_index: Mapping[Tuple[int, ...], int], assignment: Assignment) -> Optional[int]:
    if len(set(assignment.values())) != len(problem.cells):
        return None
    idx = feasible_index.get(assignment_key(problem, assignment))
    if idx is None:
        return None
    return feasible_distances[idx]


def evaluate_distribution(
    problem: PlacementProblem,
    method: str,
    assignments: Sequence[Assignment],
    probs: np.ndarray,
    counts: Mapping[int, int],
    initial_assignment: Assignment,
    initial_pos: int,
    exact_hpwl: float,
    optimal_keys: set[Tuple[int, ...]],
    feasible_distances: Sequence[Optional[int]],
    feasible_index: Mapping[Tuple[int, ...], int],
) -> Dict[str, object]:
    total_shots = sum(counts.values())
    initial_hpwl = hpwl(problem, initial_assignment)
    initial_is_optimal = assignment_key(problem, initial_assignment) in optimal_keys
    feasible_count = 0
    feasible_hpwl_values = []
    feasible_weighted_hpwl = 0.0
    sampled_best_hpwl = None
    sampled_opt_count = 0
    optimal_probability = 0.0
    distance0_probability = 0.0
    mean_distance_probability = 0.0
    distance_mass: Dict[int, float] = {}

    for idx, assignment in enumerate(assignments):
        feasible = len(set(assignment.values())) == len(problem.cells)
        is_initial = idx == initial_pos
        if is_initial:
            distance0_probability += float(probs[idx])
        if assignment_key(problem, assignment) in optimal_keys:
            optimal_probability += float(probs[idx])
            sampled_opt_count += int(counts.get(idx, 0))
        if not feasible:
            continue
        cost = hpwl(problem, assignment)
        count = int(counts.get(idx, 0))
        feasible_count += count
        if count:
            feasible_hpwl_values.extend([cost] * count)
            if sampled_best_hpwl is None or cost < sampled_best_hpwl:
                sampled_best_hpwl = cost
        feasible_weighted_hpwl += float(probs[idx]) * cost
        distance = feasible_key_to_distance(problem, feasible_distances, feasible_index, assignment)
        if distance is not None:
            distance_mass[int(distance)] = distance_mass.get(int(distance), 0.0) + float(probs[idx])
            mean_distance_probability += int(distance) * float(probs[idx])

    feasible_probability = sum(float(probs[idx]) for idx, assignment in enumerate(assignments) if len(set(assignment.values())) == len(problem.cells))
    if method.startswith("collision-free"):
        feasible_probability = 1.0
    mean_feasible_hpwl = float(mean(feasible_hpwl_values)) if feasible_hpwl_values else None
    median_feasible_hpwl = float(median(feasible_hpwl_values)) if feasible_hpwl_values else None
    optimal_hit = sampled_opt_count > 0
    discovery_hit = optimal_hit and not initial_is_optimal
    best_hpwl = sampled_best_hpwl
    approximation_ratio = None if best_hpwl is None else best_hpwl / exact_hpwl

    return {
        "initial_hpwl": initial_hpwl,
        "initial_approximation_ratio": initial_hpwl / exact_hpwl if exact_hpwl else 1.0,
        "initial_state_is_optimal": initial_is_optimal,
        "best_sampled_hpwl": best_hpwl,
        "improvement_over_initial": None if best_hpwl is None else initial_hpwl - best_hpwl,
        "approximation_ratio": approximation_ratio,
        "mean_feasible_hpwl": mean_feasible_hpwl,
        "median_feasible_hpwl": median_feasible_hpwl,
        "feasible_fraction": feasible_count / max(total_shots, 1),
        "feasible_probability": feasible_probability,
        "optimal_probability": optimal_probability,
        "optimal_hit": optimal_hit,
        "optimal_sample_count": sampled_opt_count,
        "discovery_hit": discovery_hit,
        "distance_0_probability": distance0_probability,
        "mean_sampled_graph_distance": mean_distance_probability / max(feasible_probability, 1e-12),
        "probability_mass_by_distance": distance_mass,
    }


def make_row_xy_circuit_metrics(problem: PlacementProblem, reps: int) -> Dict[str, object]:
    cache_key = ("row_xy", problem.cells, problem.sites, problem.nets, reps)
    if cache_key in COST_CACHE:
        return dict(COST_CACHE[cache_key])
    model = build_qubo(problem)
    constant, h, j = ising_terms(model)
    cost_operator = sparse_pauli_from_ising(model.num_variables, h, j)
    mixer_operator = cell_xy_mixer_operator(problem)
    initial_state = QuantumCircuit(problem.num_variables)
    for idx, bit in enumerate(assignment_to_bits(problem, next(feasible_assignments(problem)))):
        if bit:
            initial_state.x(idx)
    ansatz = QAOAAnsatz(cost_operator=cost_operator, mixer_operator=mixer_operator, initial_state=initial_state, reps=reps)
    logical = {
        "depth": ansatz.depth(),
        "size": ansatz.size(),
        "two_qubit_gate_count": 0,
        "operations": {name: int(count) for name, count in ansatz.count_ops().items()},
    }
    transpiled = transpiled_metrics(ansatz)
    metrics = {
        "logical_depth_total": logical["depth"],
        "logical_transition_count": None,
        "estimated_transpiled_two_qubit_count": transpiled["two_qubit_gate_count"],
        "estimated_serial_transpiled_depth": transpiled["depth"],
        "logical_circuit": logical,
        "transpiled_circuit": transpiled,
    }
    COST_CACHE[cache_key] = metrics
    return metrics


def cached_legal_schedule_costs(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    edges: Sequence[Edge],
    schedule: MixerSchedule,
    reps: int,
) -> Mapping[str, object]:
    cache_key = ("legal", problem.cells, problem.sites, problem.nets, schedule.name, reps)
    if cache_key not in COST_CACHE:
        COST_CACHE[cache_key] = schedule_costs(problem, assignments, edges, schedule, reps, full_transpile_edge_threshold=0)
    return COST_CACHE[cache_key]


def optimize_run(
    objective,
    initial_params: np.ndarray,
    optimizer: str,
    maxiter: int,
) -> Tuple[object, List[float], float]:
    history: List[float] = []

    def tracked(params):
        value = float(objective(params))
        history.append(value)
        return value

    start = time.time()
    if optimizer == "COBYLA":
        result = minimize(tracked, initial_params, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2, "tol": 1e-3})
    elif optimizer == "Nelder-Mead":
        result = minimize(tracked, initial_params, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3})
    else:
        raise ValueError(f"unsupported optimizer: {optimizer}")
    return result, history, time.time() - start


def run_method(
    problem: PlacementProblem,
    instance_id: int,
    instance_seed: int,
    method: str,
    reps: int,
    maxiter: int,
    shots: int,
    optimizer: str,
    optimizer_seed: int,
    init_mode: str,
    init_seed: int,
    initial_assignment: Assignment,
    exact_hpwl: float,
    optimal_keys: set[Tuple[int, ...]],
    feasible_distances: Sequence[Optional[int]],
    feasible_index: Mapping[Tuple[int, ...], int],
    schedule_seed: int,
    transfer_params: Optional[Sequence[float]] = None,
) -> Tuple[MetricRow, MetricRow, MetricRow, Optional[Sequence[float]]]:
    model = build_qubo(problem)
    initial_key = assignment_key(problem, initial_assignment)
    initial_to_optimum_distance = None
    if initial_key in feasible_index:
        distances_from_initial = feasible_distances
        opt_distances = [distances_from_initial[feasible_index[key]] for key in optimal_keys if key in feasible_index and distances_from_initial[feasible_index[key]] is not None]
        initial_to_optimum_distance = min(opt_distances) if opt_distances else None

    if transfer_params is not None:
        params0 = transferred_params(transfer_params, reps, optimizer_seed)
        parameter_start = "transfer"
    else:
        params0 = init_params(reps, optimizer_seed)
        parameter_start = "fresh"

    cost_fields: Mapping[str, object]
    schedule_name = method
    if method == "row_xy":
        row_assignments, _state_indices, energies, _state_index = constrained_cost_energies(problem, model)
        row_index = {assignment_key(problem, assignment): idx for idx, assignment in enumerate(row_assignments)}
        initial_pos = row_index[initial_key]

        def objective(params):
            return expectation_from_probs(row_xy_qaoa_probs(params, energies, reps, problem, row_assignments, row_index, initial_pos), energies)

        initial_expectation = objective(params0)
        result, history, elapsed = optimize_run(objective, params0, optimizer, maxiter)
        probs = row_xy_qaoa_probs(result.x, energies, reps, problem, row_assignments, row_index, initial_pos)
        counts = sampled_distribution(probs, shots=shots, seed=optimizer_seed + 100 * reps + max(init_seed, 0))
        metrics = evaluate_distribution(problem, "row_xy", row_assignments, probs, counts, initial_assignment, initial_pos, exact_hpwl, optimal_keys, feasible_distances, feasible_index)
        cost_fields = make_row_xy_circuit_metrics(problem, reps)
    else:
        assignments, energies, index_by_key = collision_free_cost_energies(problem, model)
        edges = collision_free_swap_edges(problem, assignments, index_by_key)
        if method == "fixed":
            schedule = make_fixed_schedule(edges)
        elif method == "reversed":
            schedule = make_reversed_schedule(edges)
        elif method == "random_per_layer":
            schedule = make_random_layer_schedule(edges, reps, schedule_seed)
        elif method == "edge_colored":
            schedule = make_edge_colored_schedule(edges, reps)
        else:
            raise ValueError(f"unsupported method: {method}")
        schedule_name = schedule.name
        initial_pos = index_by_key[initial_key]

        def objective(params):
            return expectation_from_probs(legal_qaoa_probs(params, energies, reps, initial_pos, schedule), energies)

        initial_expectation = objective(params0)
        result, history, elapsed = optimize_run(objective, params0, optimizer, maxiter)
        probs = legal_qaoa_probs(result.x, energies, reps, initial_pos, schedule)
        counts = sampled_distribution(probs, shots=shots, seed=optimizer_seed + 200 * reps + max(init_seed, 0))
        metrics = evaluate_distribution(problem, f"collision-free {method}", assignments, probs, counts, initial_assignment, initial_pos, exact_hpwl, optimal_keys, feasible_distances, feasible_index)
        cost_fields = cached_legal_schedule_costs(problem, assignments, edges, schedule, reps)

    final_betas = [float(result.x[2 * layer + 1]) for layer in range(reps)]
    final_gammas = [float(result.x[2 * layer]) for layer in range(reps)]
    near_zero_beta = any(abs(((beta + math.pi) % (2 * math.pi)) - math.pi) < 1e-2 for beta in final_betas)
    near_identity_beta = any(min(abs(beta % math.pi), abs((beta % math.pi) - math.pi)) < 1e-2 for beta in final_betas)

    common = {
        "instance": instance_id,
        "instance_seed": instance_seed,
        "method": method,
        "schedule": schedule_name,
        "optimizer": optimizer,
        "optimizer_seed": optimizer_seed,
        "parameter_start": parameter_start,
        "init_mode": init_mode,
        "init_seed": init_seed,
        "reps": reps,
        "shots": shots,
        "maxiter": maxiter,
        "num_cells": len(problem.cells),
        "num_sites": len(problem.sites),
        "num_qubits": problem.num_variables,
        "exact_hpwl": exact_hpwl,
        "initial_assignment": dict(initial_assignment),
        "initial_to_optimum_swap_distance": initial_to_optimum_distance,
        **metrics,
        "runtime_seconds": elapsed,
        "optimizer_evaluations": len(history),
        "logical_depth": cost_fields["logical_depth_total"],
        "transpiled_depth": cost_fields["estimated_serial_transpiled_depth"],
        "two_qubit_gate_count": cost_fields["estimated_transpiled_two_qubit_count"],
    }
    optimizer_row = {
        **{key: common[key] for key in ["instance", "instance_seed", "method", "schedule", "optimizer", "optimizer_seed", "parameter_start", "init_mode", "init_seed", "reps"]},
        "initial_parameter_vector": [float(value) for value in params0],
        "final_parameter_vector": [float(value) for value in result.x],
        "final_betas": final_betas,
        "final_gammas": final_gammas,
        "near_zero_beta": near_zero_beta,
        "near_identity_beta": near_identity_beta,
        "objective_trace": [float(value) for value in history],
        "optimizer_success": bool(result.success),
        "optimizer_status": int(getattr(result, "status", -999)),
        "optimizer_message": str(result.message),
        "optimizer_evaluations": len(history),
        "initial_expectation_value": float(initial_expectation),
        "final_expectation_value": float(result.fun),
        "final_distance_0_probability": metrics["distance_0_probability"],
        "final_optimal_probability": metrics["optimal_probability"],
    }
    init_row = {
        "instance": instance_id,
        "instance_seed": instance_seed,
        "init_mode": init_mode,
        "init_seed": init_seed,
        "initial_assignment": dict(initial_assignment),
        "initial_hpwl": metrics["initial_hpwl"],
        "initial_approximation_ratio": metrics["initial_approximation_ratio"],
        "initial_state_is_optimal": metrics["initial_state_is_optimal"],
        "initial_to_optimum_swap_distance": initial_to_optimum_distance,
    }
    return common, optimizer_row, init_row, [float(value) for value in result.x]


def aggregate(rows: Sequence[MetricRow]) -> List[MetricRow]:
    group_keys = sorted({(row["method"], row["init_mode"], row["reps"], row["optimizer"]) for row in rows})
    metric_names = [
        "feasible_fraction",
        "optimal_probability",
        "approximation_ratio",
        "best_sampled_hpwl",
        "mean_feasible_hpwl",
        "improvement_over_initial",
        "distance_0_probability",
        "mean_sampled_graph_distance",
        "runtime_seconds",
        "logical_depth",
        "transpiled_depth",
        "two_qubit_gate_count",
    ]
    output = []
    for method, init_mode, reps, optimizer in group_keys:
        subset = [row for row in rows if row["method"] == method and row["init_mode"] == init_mode and row["reps"] == reps and row["optimizer"] == optimizer]
        out: MetricRow = {
            "method": method,
            "init_mode": init_mode,
            "reps": reps,
            "optimizer": optimizer,
            "runs": len(subset),
            "optimal_hit_rate": mean(1.0 if row["optimal_hit"] else 0.0 for row in subset),
            "discovery_hit_rate": mean(1.0 if row["discovery_hit"] else 0.0 for row in subset),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            if not values:
                continue
            lo, hi = bootstrap_ci(values)
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_median"] = median(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_ci95_low"] = lo
            out[f"{metric}_ci95_high"] = hi
        output.append(out)
    return output


def aggregate_by_parameter_start(rows: Sequence[MetricRow]) -> List[MetricRow]:
    group_keys = sorted({(row["method"], row["init_mode"], row["reps"], row["optimizer"], row["parameter_start"]) for row in rows})
    metric_names = [
        "feasible_fraction",
        "optimal_probability",
        "approximation_ratio",
        "best_sampled_hpwl",
        "mean_feasible_hpwl",
        "improvement_over_initial",
        "distance_0_probability",
        "mean_sampled_graph_distance",
        "runtime_seconds",
        "logical_depth",
        "transpiled_depth",
        "two_qubit_gate_count",
    ]
    output = []
    for method, init_mode, reps, optimizer, parameter_start in group_keys:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and row["init_mode"] == init_mode
            and row["reps"] == reps
            and row["optimizer"] == optimizer
            and row["parameter_start"] == parameter_start
        ]
        out: MetricRow = {
            "method": method,
            "init_mode": init_mode,
            "reps": reps,
            "optimizer": optimizer,
            "parameter_start": parameter_start,
            "runs": len(subset),
            "optimal_hit_rate": mean(1.0 if row["optimal_hit"] else 0.0 for row in subset),
            "discovery_hit_rate": mean(1.0 if row["discovery_hit"] else 0.0 for row in subset),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            if not values:
                continue
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_median"] = median(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        output.append(out)
    return output


def plot_aggregate(rows: Sequence[MetricRow], out_dir: Path) -> None:
    plt.figure(figsize=(11, 5))
    labels = []
    values = []
    for row in rows:
        if row["optimizer"] != "COBYLA":
            continue
        labels.append(f"{row['method']}\n{row['init_mode']}\np{row['reps']}")
        values.append(float(row["discovery_hit_rate"]))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right", fontsize=7)
    plt.ylabel("Discovery hit rate")
    plt.tight_layout()
    plt.savefig(out_dir / "discovery_hit_rate.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method and row["optimizer"] == "COBYLA"]
        plt.scatter(
            [float(row.get("transpiled_depth_mean", 0.0)) for row in subset],
            [float(row.get("approximation_ratio_mean", 0.0)) for row in subset],
            label=method,
        )
    plt.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    plt.xlabel("Mean transpiled/estimated depth")
    plt.ylabel("Mean approximation ratio")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "circuit_cost_vs_quality.png", dpi=180)
    plt.close()


def p1_landscape(
    problem: PlacementProblem,
    method: str,
    initial_assignment: Assignment,
    grid_size: int,
    out_dir: Path,
    seed: int,
) -> List[MetricRow]:
    model = build_qubo(problem)
    exact = brute_force(problem)
    exact_hpwl = exact[0][0]
    optimal_keys = {assignment_key(problem, assignment) for cost, assignment in exact if cost == exact_hpwl}
    feasible_assigns, feasible_energies, feasible_index = collision_free_cost_energies(problem, model)
    feasible_edges = collision_free_swap_edges(problem, feasible_assigns, feasible_index)
    feasible_distances = bfs_distances(build_adjacency(len(feasible_assigns), feasible_edges), feasible_index[assignment_key(problem, initial_assignment)])
    gammas = np.linspace(0, math.pi, grid_size)
    betas = np.linspace(0, math.pi, grid_size)
    rows = []
    expectation = np.zeros((grid_size, grid_size))
    opt_prob = np.zeros_like(expectation)
    d0_prob = np.zeros_like(expectation)
    mean_dist = np.zeros_like(expectation)
    feasible_prob = np.ones_like(expectation)

    if method == "row_xy":
        assignments, _state_indices, energies, _state_index = constrained_cost_energies(problem, model)
        index_by_key = {assignment_key(problem, assignment): idx for idx, assignment in enumerate(assignments)}
        initial_pos = index_by_key[assignment_key(problem, initial_assignment)]
        prob_fn = lambda params: row_xy_qaoa_probs(params, energies, 1, problem, assignments, index_by_key, initial_pos)
    else:
        assignments, energies, index_by_key = feasible_assigns, feasible_energies, feasible_index
        if method == "fixed":
            schedule = make_fixed_schedule(feasible_edges)
        elif method == "reversed":
            schedule = make_reversed_schedule(feasible_edges)
        elif method == "random_per_layer":
            schedule = make_random_layer_schedule(feasible_edges, 1, seed)
        else:
            raise ValueError(method)
        initial_pos = index_by_key[assignment_key(problem, initial_assignment)]
        prob_fn = lambda params: legal_qaoa_probs(params, energies, 1, initial_pos, schedule)

    for gamma_idx, gamma in enumerate(gammas):
        for beta_idx, beta in enumerate(betas):
            probs = prob_fn([gamma, beta])
            expectation[gamma_idx, beta_idx] = expectation_from_probs(probs, energies)
            counts = {idx: int(round(float(prob) * 100000)) for idx, prob in enumerate(probs) if prob > 0}
            metrics = evaluate_distribution(problem, method, assignments, probs, counts, initial_assignment, initial_pos, exact_hpwl, optimal_keys, feasible_distances, feasible_index)
            opt_prob[gamma_idx, beta_idx] = metrics["optimal_probability"]
            d0_prob[gamma_idx, beta_idx] = metrics["distance_0_probability"]
            mean_dist[gamma_idx, beta_idx] = metrics["mean_sampled_graph_distance"]
            feasible_prob[gamma_idx, beta_idx] = metrics["feasible_probability"]
            rows.append(
                {
                    "method": method,
                    "gamma": float(gamma),
                    "beta": float(beta),
                    "expected_cost": float(expectation[gamma_idx, beta_idx]),
                    "optimal_probability": float(opt_prob[gamma_idx, beta_idx]),
                    "distance_0_probability": float(d0_prob[gamma_idx, beta_idx]),
                    "mean_graph_distance": float(mean_dist[gamma_idx, beta_idx]),
                    "feasible_probability": float(feasible_prob[gamma_idx, beta_idx]),
                }
            )

    def heatmap(matrix: np.ndarray, name: str, label: str) -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, origin="lower", aspect="auto", extent=[0, math.pi, 0, math.pi])
        best_idx = np.unravel_index(np.argmin(expectation), expectation.shape)
        plt.scatter([betas[best_idx[1]]], [gammas[best_idx[0]]], marker="x", color="white", label="best expected cost")
        plt.xlabel("beta")
        plt.ylabel("gamma")
        plt.title(f"{method}: {label}")
        plt.colorbar()
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / f"landscape_{method}_{name}.png", dpi=180)
        plt.close()

    heatmap(expectation, "expected_cost", "expected cost")
    heatmap(opt_prob, "optimal_probability", "optimal probability")
    heatmap(d0_prob, "distance0_probability", "distance-0 probability")
    heatmap(mean_dist, "mean_graph_distance", "mean graph distance")
    if method == "row_xy":
        heatmap(feasible_prob, "feasible_probability", "feasible probability")
    return rows


def ideal_vs_ordered(problem: PlacementProblem, initial_assignment: Assignment, out_dir: Path, seed: int) -> List[MetricRow]:
    model = build_qubo(problem)
    assignments, energies, index_by_key = collision_free_cost_energies(problem, model)
    edges = collision_free_swap_edges(problem, assignments, index_by_key)
    initial_pos = index_by_key[assignment_key(problem, initial_assignment)]
    exact = brute_force(problem)
    exact_hpwl = exact[0][0]
    optimal_keys = {assignment_key(problem, assignment) for cost, assignment in exact if cost == exact_hpwl}
    distances = bfs_distances(build_adjacency(len(assignments), edges), initial_pos)
    adjacency_h = np.zeros((len(assignments), len(assignments)), dtype=np.complex128)
    for left, right in edges:
        adjacency_h[left, right] = 1.0
        adjacency_h[right, left] = 1.0
    state0 = np.zeros(len(assignments), dtype=np.complex128)
    state0[initial_pos] = 1.0
    schedules = [
        make_fixed_schedule(edges),
        make_reversed_schedule(edges),
        make_random_layer_schedule(edges, 1, seed),
        make_edge_colored_schedule(edges, 1),
    ]
    rows = []
    betas = np.linspace(0, math.pi / 2, 41)
    gamma = math.pi / 4
    cost_phase = np.exp(-1j * gamma * energies)
    for beta in betas:
        ideal_state = expm(-1j * beta * adjacency_h) @ (cost_phase * state0)
        ideal_probs = np.abs(ideal_state) ** 2
        for schedule in schedules:
            state = cost_phase * state0
            apply_ordered_swap_mixer(state, float(beta), schedule_for_layer(schedule, 0, 1))
            probs = np.abs(state) ** 2
            fidelity = float(abs(np.vdot(ideal_state, state)) ** 2)
            tvd = float(0.5 * np.sum(np.abs(ideal_probs - probs)))
            counts = {idx: int(round(float(prob) * 100000)) for idx, prob in enumerate(probs) if prob > 0}
            metrics = evaluate_distribution(problem, schedule.name, assignments, probs, counts, initial_assignment, initial_pos, exact_hpwl, optimal_keys, distances, index_by_key)
            rows.append(
                {
                    "schedule": schedule.name,
                    "beta": float(beta),
                    "gamma": gamma,
                    "fidelity_to_ideal": fidelity,
                    "total_variation_to_ideal": tvd,
                    "distance_0_probability": metrics["distance_0_probability"],
                    "mean_graph_distance": metrics["mean_sampled_graph_distance"],
                    "optimal_probability": metrics["optimal_probability"],
                }
            )

    plt.figure(figsize=(7, 5))
    for schedule in sorted({row["schedule"] for row in rows}):
        subset = [row for row in rows if row["schedule"] == schedule]
        plt.plot([row["beta"] for row in subset], [row["fidelity_to_ideal"] for row in subset], label=schedule)
    plt.xlabel("beta")
    plt.ylabel("fidelity to ideal summed mixer")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "ideal_vs_ordered_fidelity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for schedule in sorted({row["schedule"] for row in rows}):
        subset = [row for row in rows if row["schedule"] == schedule]
        plt.plot([row["beta"] for row in subset], [row["total_variation_to_ideal"] for row in subset], label=schedule)
    plt.xlabel("beta")
    plt.ylabel("total variation distance")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "ideal_vs_ordered_tvd.png", dpi=180)
    plt.close()
    return rows


def relabel_problem(problem: PlacementProblem, seed: int, permute_cells: bool, permute_sites: bool) -> PlacementProblem:
    rng = random.Random(seed)
    cells = list(problem.cells)
    sites = list(problem.sites)
    cell_map = {cell: cell for cell in cells}
    if permute_cells:
        shuffled_cells = list(cells)
        rng.shuffle(shuffled_cells)
        cell_map = {old: new for old, new in zip(cells, shuffled_cells)}
        cells = shuffled_cells
    if permute_sites:
        rng.shuffle(sites)
    nets = tuple((cell_map[left], cell_map[right], weight) for left, right, weight in problem.nets)
    return PlacementProblem(cells=tuple(cells), sites=tuple(sites), nets=nets, penalty=problem.penalty)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/qaoa_placement/mixer_design_results/robust_060826")
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--num-cells", type=int, default=3)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--height", type=int, default=2)
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--optimizer-seeds", type=int, nargs="+", default=[11, 17, 23, 31, 43, 59, 71, 89])
    parser.add_argument("--optimizers", nargs="+", default=["COBYLA", "Nelder-Mead"], choices=["COBYLA", "Nelder-Mead"])
    parser.add_argument("--methods", nargs="+", default=["fixed", "reversed", "random_per_layer", "edge_colored", "row_xy"])
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--graph-bfs-threshold", type=int, default=720)
    parser.add_argument("--landscape-grid", type=int, default=41)
    parser.add_argument("--skip-landscapes", action="store_true")
    parser.add_argument("--skip-label-sensitivity", action="store_true")
    parser.add_argument("--label-sensitivity-seeds", type=int, nargs="+", default=[9101, 9102, 9103])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    landscape_dir = out_dir / "landscapes"
    landscape_dir.mkdir(exist_ok=True)

    ablation_rows: List[MetricRow] = []
    optimizer_rows: List[MetricRow] = []
    init_rows_by_key: Dict[Tuple[int, str, int], MetricRow] = {}
    graph_rows: List[MetricRow] = []
    label_rows: List[MetricRow] = []
    transfer_cache: Dict[Tuple[int, str, str, int, str, int, str], Sequence[float]] = {}

    for instance_id in range(args.instances):
        instance_seed = args.seed_start + instance_id
        problem = random_problem(instance_seed, args.num_cells, args.width, args.height)
        exact = brute_force(problem)
        exact_hpwl = exact[0][0]
        optimal_keys = {assignment_key(problem, assignment) for cost, assignment in exact if cost == exact_hpwl}
        feasible_assigns, _energies, feasible_index = collision_free_cost_energies(problem, build_qubo(problem))
        edges = collision_free_swap_edges(problem, feasible_assigns, feasible_index)
        graph_rows.append({"instance": instance_id, "instance_seed": instance_seed, **graph_audit(problem, feasible_assigns, edges, args.graph_bfs_threshold)})
        adjacency = build_adjacency(len(feasible_assigns), edges)
        init_specs = initialization_classes(problem, seed=3000 + instance_id)
        for init_mode, init_seed, initial_assignment in init_specs:
            init_pos = feasible_index[assignment_key(problem, initial_assignment)]
            distances = bfs_distances(adjacency, init_pos)
            init_key = (instance_id, init_mode, init_seed)
            for optimizer in args.optimizers:
                for method in args.methods:
                    for optimizer_seed in args.optimizer_seeds:
                        for reps in args.reps:
                            transfer_key = (instance_id, method, optimizer, optimizer_seed, init_mode, init_seed, "fresh")
                            previous = transfer_cache.get(transfer_key) if reps > 1 else None
                            row, opt_row, init_row, final_params = run_method(
                                problem,
                                instance_id,
                                instance_seed,
                                method,
                                reps,
                                args.maxiter,
                                args.shots,
                                optimizer,
                                optimizer_seed,
                                init_mode,
                                init_seed,
                                initial_assignment,
                                exact_hpwl,
                                optimal_keys,
                                distances,
                                feasible_index,
                                schedule_seed=7000 + instance_id + reps + optimizer_seed,
                                transfer_params=None,
                            )
                            ablation_rows.append(row)
                            optimizer_rows.append(opt_row)
                            init_rows_by_key[init_key] = init_row
                            if optimizer == "COBYLA":
                                transfer_cache[transfer_key] = final_params
                            if optimizer == "COBYLA" and reps in (2, 3) and previous is not None:
                                transfer_row, transfer_opt, _transfer_init, transfer_final = run_method(
                                    problem,
                                    instance_id,
                                    instance_seed,
                                    method,
                                    reps,
                                    args.maxiter,
                                    args.shots,
                                    optimizer,
                                    optimizer_seed,
                                    init_mode,
                                    init_seed,
                                    initial_assignment,
                                    exact_hpwl,
                                    optimal_keys,
                                    distances,
                                    feasible_index,
                                    schedule_seed=8000 + instance_id + reps + optimizer_seed,
                                    transfer_params=previous,
                                )
                                ablation_rows.append(transfer_row)
                                optimizer_rows.append(transfer_opt)
                                transfer_cache[transfer_key] = transfer_final

    aggregate_rows = aggregate(ablation_rows)
    aggregate_by_start_rows = aggregate_by_parameter_start(ablation_rows)
    write_csv(ablation_rows, out_dir / "robust_schedule_ablation.csv")
    write_csv(aggregate_rows, out_dir / "robust_schedule_aggregate.csv")
    write_csv(aggregate_by_start_rows, out_dir / "robust_schedule_aggregate_by_start.csv")
    write_csv(optimizer_rows, out_dir / "optimizer_diagnostics.csv")
    write_csv(init_rows_by_key.values(), out_dir / "initialization_diagnostics.csv")
    write_csv(graph_rows, out_dir / "robust_graph_audit.csv")
    plot_aggregate(aggregate_rows, out_dir)

    landscape_rows: List[MetricRow] = []
    ideal_rows: List[MetricRow] = []
    if not args.skip_landscapes:
        representative = random_problem(args.seed_start, args.num_cells, args.width, args.height)
        representative_init = next(feasible_assignments(representative))
        for method in ["fixed", "reversed", "random_per_layer", "row_xy"]:
            landscape_rows.extend(p1_landscape(representative, method, representative_init, args.landscape_grid, landscape_dir, seed=777))
        ideal_rows = ideal_vs_ordered(representative, representative_init, landscape_dir, seed=777)
        write_csv(landscape_rows, out_dir / "p1_landscape_grid.csv")
        write_csv(ideal_rows, out_dir / "ideal_vs_ordered_mixer.csv")

    if not args.skip_label_sensitivity:
        base = random_problem(args.seed_start, args.num_cells, args.width, args.height)
        variants = [("original", base)]
        for seed in args.label_sensitivity_seeds:
            variants.append((f"cell_perm_{seed}", relabel_problem(base, seed, True, False)))
            variants.append((f"site_perm_{seed}", relabel_problem(base, seed, False, True)))
            variants.append((f"cell_site_perm_{seed}", relabel_problem(base, seed, True, True)))
        for variant_name, problem in variants:
            exact = brute_force(problem)
            exact_hpwl = exact[0][0]
            optimal_keys = {assignment_key(problem, assignment) for cost, assignment in exact if cost == exact_hpwl}
            feasible_assigns, _energies, feasible_index = collision_free_cost_energies(problem, build_qubo(problem))
            edges = collision_free_swap_edges(problem, feasible_assigns, feasible_index)
            distances = bfs_distances(build_adjacency(len(feasible_assigns), edges), feasible_index[assignment_key(problem, next(feasible_assignments(problem)))])
            for method in ["fixed", "reversed", "random_per_layer"]:
                row, _opt_row, _init_row, _params = run_method(
                    problem,
                    0,
                    args.seed_start,
                    method,
                    2,
                    args.maxiter,
                    args.shots,
                    "COBYLA",
                    args.optimizer_seeds[0],
                    "deterministic",
                    -1,
                    next(feasible_assignments(problem)),
                    exact_hpwl,
                    optimal_keys,
                    distances,
                    feasible_index,
                    schedule_seed=999,
                )
                row["label_variant"] = variant_name
                label_rows.append(row)
        write_csv(label_rows, out_dir / "label_permutation_sensitivity.csv")

    with (out_dir / "robust_schedule_report.json").open("w") as handle:
        json.dump(
            json_ready(
                {
                    "metric_definitions": {
                        "optimal_hit": "At least one sampled feasible placement is globally optimal in the run.",
                        "optimal_probability": "Total exact final measurement probability assigned to all globally optimal feasible placements.",
                        "approximation_ratio": "Best feasible sampled HPWL divided by exact global HPWL. It is not the expectation value.",
                        "feasible_fraction": "Fraction of finite-shot samples decoded as legal collision-free placements.",
                        "distance_0_probability": "Exact final probability retained on the initial subspace basis state.",
                        "runs": "Number of run-level rows in each aggregate group.",
                        "discovery_hit": "optimal_hit AND initial_state_is_not_optimal.",
                    },
                    "config": vars(args),
                    "aggregate": aggregate_rows,
                    "aggregate_by_parameter_start": aggregate_by_start_rows,
                    "ablation_rows": ablation_rows,
                    "optimizer_rows": optimizer_rows,
                    "initialization_rows": list(init_rows_by_key.values()),
                    "graph_rows": graph_rows,
                    "landscape_rows": landscape_rows,
                    "ideal_vs_ordered_rows": ideal_rows,
                    "label_sensitivity_rows": label_rows,
                }
            ),
            handle,
            indent=2,
        )

    print(f"Wrote robust schedule results to {out_dir}")
    for row in aggregate_rows[:20]:
        print(row)


if __name__ == "__main__":
    main()
