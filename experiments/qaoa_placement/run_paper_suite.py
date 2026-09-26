"""Paper-grade exact-simulation suite for QAOA placement.

This runner is resumable: every completed run appends to CSV outputs with a
deterministic run id, and reruns skip completed ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark_suite import load_manifest, load_manifest_payload, problem_from_manifest, write_manifest
from mixer_circuit_costs import schedule_costs
from mixer_diagnostics import (
    Edge,
    MixerSchedule,
    bfs_distances,
    build_adjacency,
    graph_audit,
    make_fixed_schedule,
    make_random_layer_schedule,
    make_reversed_schedule,
    schedule_for_layer,
)
from objective_functions import objective_label, objective_value
from ordering_control import make_palindromic_schedule, make_random_once_schedule
from placement_core import Assignment, PlacementProblem, brute_force, build_qubo, hpwl
from run_qaoa_simulation import (
    assignment_key,
    collision_free_cost_energies,
    collision_free_swap_edges,
    constrained_cost_energies,
    qubo_energy,
    sampled_distribution,
)
from run_robust_schedule_experiment import (
    apply_cell_xy_mixer,
    apply_ordered_swap_mixer,
    init_params,
    initialization_classes,
    make_row_xy_circuit_metrics,
    optimize_run,
)
from sampling_metrics import practical_sampling_metrics


MetricRow = Dict[str, object]
COST_CACHE: Dict[Tuple[object, ...], Mapping[str, object]] = {}


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


def append_csv(row: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_failure(row: Mapping[str, object], path: Path) -> None:
    append_csv(row, path)


def completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as handle:
        return {row["run_id"] for row in csv.DictReader(handle)}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(json_ready(data), handle, indent=2)


def software_versions() -> Dict[str, object]:
    versions: Dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception as exc:  # pragma: no cover - diagnostic only
        versions["scipy_error"] = str(exc)
    try:
        import qiskit

        versions["qiskit"] = getattr(qiskit, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        versions["qiskit_error"] = str(exc)
    return versions


def ordered_samples(probs: np.ndarray, shots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(len(probs)), size=shots, p=probs)


def counts_from_sequence(samples: Sequence[int]) -> Dict[int, int]:
    values, counts = np.unique(np.asarray(samples, dtype=int), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def apply_schedule_mixer(state: np.ndarray, beta: float, schedule: MixerSchedule, layer_idx: int, reps: int) -> None:
    if schedule.name == "palindromic_fr":
        apply_ordered_swap_mixer(state, beta / 2.0, schedule.edge_layers[0])
        apply_ordered_swap_mixer(state, beta / 2.0, schedule.edge_layers[1])
        return
    apply_ordered_swap_mixer(state, beta, schedule_for_layer(schedule, layer_idx, reps))


def legal_probs(params: Sequence[float], energies: np.ndarray, reps: int, initial_pos: int, schedule: MixerSchedule) -> np.ndarray:
    state = np.zeros(len(energies), dtype=np.complex128)
    state[initial_pos] = 1.0
    for layer in range(reps):
        state *= np.exp(-1j * params[2 * layer] * energies)
        apply_schedule_mixer(state, params[2 * layer + 1], schedule, layer, reps)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def row_xy_probs(
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


def schedule_for_method(method: str, edges: Sequence[Edge], reps: int, schedule_seed: int) -> MixerSchedule:
    if method == "fixed":
        return make_fixed_schedule(edges)
    if method == "reversed":
        return make_reversed_schedule(edges)
    if method == "random_once":
        return make_random_once_schedule(edges, schedule_seed)
    if method == "random_per_layer":
        return make_random_layer_schedule(edges, reps, schedule_seed)
    if method == "palindromic":
        return make_palindromic_schedule(edges)
    raise ValueError(f"unsupported legal method: {method}")


def legal_costs(problem: PlacementProblem, assignments: Sequence[Assignment], edges: Sequence[Edge], schedule: MixerSchedule, reps: int) -> Mapping[str, object]:
    key = ("legal", problem.cells, problem.sites, problem.nets, schedule.name, reps)
    if key not in COST_CACHE:
        metrics = schedule_costs(problem, assignments, edges, schedule, reps, full_transpile_edge_threshold=0)
        if schedule.name == "palindromic_fr":
            # Two half-angle passes per QAOA layer; same transition count doubled.
            metrics = dict(metrics)
            metrics["logical_depth_total"] = 2 * len(edges) * reps
            metrics["logical_transition_count"] = 2 * len(edges) * reps
            metrics["estimated_transpiled_two_qubit_count"] *= 2
            metrics["estimated_serial_transpiled_depth"] *= 2
        COST_CACHE[key] = metrics
    return COST_CACHE[key]


def arrays_for_assignments(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    exact_hpwl: float,
    optimal_keys: set[Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hpwls = []
    feasible = []
    optimal = []
    for assignment in assignments:
        is_feasible = len(set(assignment.values())) == len(problem.cells)
        feasible.append(is_feasible)
        if is_feasible:
            value = hpwl(problem, assignment)
            hpwls.append(value)
            optimal.append(assignment_key(problem, assignment) in optimal_keys)
        else:
            hpwls.append(float("inf"))
            optimal.append(False)
    return np.asarray(hpwls, dtype=float), np.asarray(feasible, dtype=bool), np.asarray(optimal, dtype=bool)


def distance_metrics(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    probs: np.ndarray,
    initial_pos: int,
    feasible_distances: Sequence[Optional[int]],
    feasible_index: Mapping[Tuple[int, ...], int],
    feasible_mask: np.ndarray,
) -> Dict[str, object]:
    d0 = float(probs[initial_pos])
    feasible_probability = float(np.sum(probs[feasible_mask]))
    total = 0.0
    for idx, assignment in enumerate(assignments):
        if not feasible_mask[idx]:
            continue
        distance_idx = feasible_index.get(assignment_key(problem, assignment))
        if distance_idx is None:
            continue
        distance = feasible_distances[distance_idx]
        if distance is not None:
            total += float(distance) * float(probs[idx])
    return {
        "distance_0_probability": d0,
        "mean_sampled_graph_distance": total / max(feasible_probability, 1e-12),
    }


def evaluate_run(
    problem: PlacementProblem,
    assignments: Sequence[Assignment],
    probs: np.ndarray,
    counts: Mapping[int, int],
    sample_sequence: Sequence[int],
    initial_assignment: Assignment,
    initial_pos: int,
    exact_hpwl: float,
    optimal_keys: set[Tuple[int, ...]],
    feasible_distances: Sequence[Optional[int]],
    feasible_index: Mapping[Tuple[int, ...], int],
    shots: int,
) -> Dict[str, object]:
    hpwls, feasible_mask, optimal_mask = arrays_for_assignments(problem, assignments, exact_hpwl, optimal_keys)
    total_shots = sum(counts.values())
    feasible_count = sum(count for idx, count in counts.items() if feasible_mask[idx])
    optimal_count = sum(count for idx, count in counts.items() if optimal_mask[idx])
    feasible_hpwls = [float(hpwls[idx]) for idx, count in counts.items() for _ in range(count) if feasible_mask[idx]]
    best_hpwl = min(feasible_hpwls) if feasible_hpwls else None
    initial_hpwl = hpwl(problem, initial_assignment)
    initial_is_optimal = assignment_key(problem, initial_assignment) in optimal_keys
    feasible_probability = float(np.sum(probs[feasible_mask]))
    optimal_probability = float(np.sum(probs[optimal_mask]))
    distance = distance_metrics(problem, assignments, probs, initial_pos, feasible_distances, feasible_index, feasible_mask)
    practical = practical_sampling_metrics(
        probs,
        counts,
        sample_sequence,
        hpwls,
        exact_hpwl,
        initial_hpwl,
        feasible_mask,
        optimal_mask,
        shots,
    )
    return {
        "initial_hpwl": initial_hpwl,
        "initial_approximation_ratio": initial_hpwl / exact_hpwl if exact_hpwl else 1.0,
        "initial_state_is_optimal": initial_is_optimal,
        "best_sampled_hpwl": best_hpwl,
        "approximation_ratio": None if best_hpwl is None else best_hpwl / exact_hpwl,
        "mean_feasible_hpwl": float(np.mean(feasible_hpwls)) if feasible_hpwls else None,
        "median_feasible_hpwl": float(np.median(feasible_hpwls)) if feasible_hpwls else None,
        "improvement_over_initial": None if best_hpwl is None else initial_hpwl - best_hpwl,
        "feasible_fraction": feasible_count / max(total_shots, 1),
        "feasible_probability": feasible_probability,
        "optimal_probability": optimal_probability,
        "optimal_hit": optimal_count > 0,
        "optimal_sample_count": optimal_count,
        "discovery_hit": optimal_count > 0 and not initial_is_optimal,
        **distance,
        **practical,
    }


def run_one(
    problem: PlacementProblem,
    manifest_row: Mapping[str, object],
    method: str,
    reps: int,
    objective: str,
    objective_params: Mapping[str, float],
    init_mode: str,
    init_seed: int,
    initial_assignment: Assignment,
    optimizer_seed: int,
    maxiter: int,
    shots: int,
    schedule_seed: int,
) -> Tuple[MetricRow, MetricRow]:
    model = build_qubo(problem)
    exact = brute_force(problem)
    exact_hpwl = exact[0][0]
    optimal_keys = {assignment_key(problem, assignment) for value, assignment in exact if value == exact_hpwl}
    feasible_assigns, _feasible_energies, feasible_index = collision_free_cost_energies(problem, model)
    edges = collision_free_swap_edges(problem, feasible_assigns, feasible_index)
    init_key = assignment_key(problem, initial_assignment)
    feasible_distances = bfs_distances(build_adjacency(len(feasible_assigns), edges), feasible_index[init_key])

    if method == "row_xy":
        assignments, _state_indices, energies, _state_index = constrained_cost_energies(problem, model)
        index_by_key = {assignment_key(problem, assignment): idx for idx, assignment in enumerate(assignments)}
        initial_pos = index_by_key[init_key]
        prob_fn = lambda params: row_xy_probs(params, energies, reps, problem, assignments, index_by_key, initial_pos)
        cost_fields = make_row_xy_circuit_metrics(problem, reps)
    else:
        assignments, energies, index_by_key = feasible_assigns, _feasible_energies, feasible_index
        schedule = schedule_for_method(method, edges, reps, schedule_seed)
        initial_pos = index_by_key[init_key]
        prob_fn = lambda params: legal_probs(params, energies, reps, initial_pos, schedule)
        cost_fields = legal_costs(problem, assignments, edges, schedule, reps)

    hpwl_costs, feasible_mask, _optimal_mask = arrays_for_assignments(problem, assignments, exact_hpwl, optimal_keys)
    objective_costs = hpwl_costs.copy()
    objective_costs[~feasible_mask] = energies[~feasible_mask] if len(energies) == len(objective_costs) else float("inf")
    params0 = init_params(reps, optimizer_seed)
    params = dict(objective_params)
    params["initial_cost"] = hpwl(problem, initial_assignment)
    if "ratio" in params:
        params["threshold_cost"] = exact_hpwl * float(params["ratio"])

    def objective_fn(theta: Sequence[float]) -> float:
        probs = prob_fn(theta)
        return objective_value(probs, objective_costs, objective, params)

    initial_value = objective_fn(params0)
    result, history, elapsed = optimize_run(objective_fn, params0, "COBYLA", maxiter)
    final_probs = prob_fn(result.x)
    sample_seed = optimizer_seed + schedule_seed + 31 * reps + max(init_seed, 0)
    sequence = ordered_samples(final_probs, shots=shots, seed=sample_seed)
    counts = counts_from_sequence(sequence)
    metrics = evaluate_run(
        problem,
        assignments,
        final_probs,
        counts,
        sequence,
        initial_assignment,
        initial_pos,
        exact_hpwl,
        optimal_keys,
        feasible_distances,
        feasible_index,
        shots,
    )
    opt_distances = [feasible_distances[feasible_index[key]] for key in optimal_keys if key in feasible_index]
    common = {
        "instance_id": manifest_row["instance_id"],
        "family": manifest_row["family"],
        "seed": manifest_row["seed"],
        "method": method,
        "objective": objective_label(objective, params),
        "objective_base": objective,
        "reps": reps,
        "init_mode": init_mode,
        "init_seed": init_seed,
        "optimizer": "COBYLA",
        "optimizer_seed": optimizer_seed,
        "schedule_seed": schedule_seed,
        "sample_seed": sample_seed,
        "shots": shots,
        "maxiter": maxiter,
        "num_cells": len(problem.cells),
        "num_sites": len(problem.sites),
        "num_qubits": problem.num_variables,
        "exact_hpwl": exact_hpwl,
        "degenerate_optima": manifest_row["degenerate_optima"],
        "initial_assignment": dict(initial_assignment),
        "initial_to_optimum_swap_distance": min(distance for distance in opt_distances if distance is not None),
        **metrics,
        "runtime_seconds": elapsed,
        "optimizer_evaluations": len(history),
        "logical_depth": cost_fields["logical_depth_total"],
        "transpiled_depth": cost_fields["estimated_serial_transpiled_depth"],
        "two_qubit_gate_count": cost_fields["estimated_transpiled_two_qubit_count"],
        "parameter_count": 2 * reps,
    }
    optimizer_row = {
        "run_id": make_run_id(common),
        "initial_parameter_vector": [float(value) for value in params0],
        "final_parameter_vector": [float(value) for value in result.x],
        "final_betas": [float(result.x[2 * layer + 1]) for layer in range(reps)],
        "final_gammas": [float(result.x[2 * layer]) for layer in range(reps)],
        "objective_trace": [float(value) for value in history],
        "initial_objective_value": float(initial_value),
        "final_objective_value": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(getattr(result, "status", -999)),
        "optimizer_message": str(result.message),
    }
    row = {"run_id": make_run_id(common), **common}
    return row, optimizer_row


def make_run_id(row: Mapping[str, object]) -> str:
    return "__".join(
        str(row[key])
        for key in (
            "instance_id",
            "method",
            "objective",
            "reps",
            "init_mode",
            "init_seed",
            "optimizer_seed",
            "schedule_seed",
        )
    )


def aggregate(rows: Sequence[MetricRow]) -> List[MetricRow]:
    groups = sorted({(str(row["method"]), str(row["objective"]), str(row["reps"]), str(row["init_mode"])) for row in rows})
    output = []
    metric_names = [
        "feasible_fraction",
        "optimal_probability",
        "approximation_ratio",
        "discovery_hit",
        "probability_beating_initial",
        "probability_within_1pct",
        "probability_within_5pct",
        "probability_within_10pct",
        "distance_0_probability",
        "mean_sampled_graph_distance",
        "runtime_seconds",
        "logical_depth",
        "transpiled_depth",
        "two_qubit_gate_count",
    ]
    def as_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value) == "True"

    def numeric(value: object) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str) and value in {"True", "False"}:
            return 1.0 if value == "True" else 0.0
        return float(value)

    for method, objective, reps, init_mode in groups:
        subset = [
            row
            for row in rows
            if str(row["method"]) == method
            and str(row["objective"]) == objective
            and str(row["reps"]) == reps
            and str(row["init_mode"]) == init_mode
        ]
        out: MetricRow = {
            "method": method,
            "objective": objective,
            "reps": reps,
            "init_mode": init_mode,
            "runs": len(subset),
            "instances": len({row["instance_id"] for row in subset}),
            "optimal_hit_rate": float(np.mean([1.0 if as_bool(row["optimal_hit"]) else 0.0 for row in subset])),
            "discovery_hit_rate": float(np.mean([1.0 if as_bool(row["discovery_hit"]) else 0.0 for row in subset])),
        }
        for metric in metric_names:
            values = [converted for row in subset if (converted := numeric(row.get(metric))) is not None]
            if values:
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_median"] = float(np.median(values))
                out[f"{metric}_std"] = float(np.std(values))
        output.append(out)
    return output


def load_rows(path: Path) -> List[MetricRow]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/qaoa_placement/paper_suite_results/verify")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--generate-manifest", action="store_true")
    parser.add_argument("--manifest-instances", type=int, default=48)
    parser.add_argument("--seed-start", type=int, default=500)
    parser.add_argument("--methods", nargs="+", default=["row_xy", "fixed", "reversed", "palindromic"])
    parser.add_argument("--objectives", nargs="+", default=["expected", "cvar_0.10", "cvar_0.25", "cvar_0.50"])
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--init-modes", nargs="+", default=["deterministic", "poor", "random", "greedy"])
    parser.add_argument("--optimizer-seeds", type=int, nargs="+", default=[11, 17, 23, 31])
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--limit-instances", type=int, default=None)
    parser.add_argument("--graph-audit-threshold", type=int, default=720)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "benchmark_manifest.json"
    if args.generate_manifest or not manifest_path.exists():
        manifest_rows = write_manifest(manifest_path, max_instances=args.manifest_instances, seed_start=args.seed_start)
    else:
        manifest_rows = load_manifest(manifest_path)
    manifest_payload = load_manifest_payload(manifest_path)
    if args.limit_instances is not None:
        manifest_rows = manifest_rows[: args.limit_instances]

    run_path = out_dir / "paper_run_level.csv"
    optimizer_path = out_dir / "paper_optimizer_diagnostics.csv"
    failure_path = out_dir / "paper_failures.csv"
    completed = completed_run_ids(run_path)
    run_rows: List[MetricRow] = load_rows(run_path)
    start = time.time()

    for manifest_row in manifest_rows:
        problem = problem_from_manifest(manifest_row)
        model = build_qubo(problem)
        feasible_assigns, _energies, feasible_index = collision_free_cost_energies(problem, model)
        edges = collision_free_swap_edges(problem, feasible_assigns, feasible_index)
        graph = graph_audit(problem, feasible_assigns, edges, args.graph_audit_threshold)
        graph_row = {"instance_id": manifest_row["instance_id"], **graph}
        graph_path = out_dir / "paper_graph_audit.csv"
        if graph_row["instance_id"] not in {row.get("instance_id") for row in load_rows(graph_path)}:
            append_csv(graph_row, graph_path)
        init_specs = initialization_classes(problem, seed=9000 + int(manifest_row["seed"]))
        init_by_mode = {mode: (seed, assignment) for mode, seed, assignment in init_specs}

        for method in args.methods:
            for objective_spec in args.objectives:
                if objective_spec.startswith("cvar_"):
                    objective = "cvar"
                    objective_params = {"alpha": float(objective_spec.split("_", 1)[1])}
                elif objective_spec == "beat_initial":
                    objective = "beat_initial"
                    objective_params = {}
                elif objective_spec.startswith("ratio_threshold_"):
                    objective = "ratio_threshold"
                    objective_params = {"ratio": float(objective_spec.rsplit("_", 1)[1])}
                else:
                    objective = "expected"
                    objective_params = {}
                for reps in args.reps:
                    for init_mode in args.init_modes:
                        init_seed, initial_assignment = init_by_mode[init_mode]
                        for optimizer_seed in args.optimizer_seeds:
                            schedule_seed = 12000 + int(manifest_row["seed"]) + optimizer_seed + 101 * reps
                            run_stub = {
                                "instance_id": manifest_row["instance_id"],
                                "method": method,
                                "objective": objective_label(objective, {**objective_params, "ratio": objective_params.get("ratio", 0.0)}),
                                "reps": reps,
                                "init_mode": init_mode,
                                "init_seed": init_seed,
                                "optimizer_seed": optimizer_seed,
                                "schedule_seed": schedule_seed,
                            }
                            run_id = make_run_id(run_stub)
                            if run_id in completed:
                                continue
                            try:
                                row, opt_row = run_one(
                                    problem,
                                    manifest_row,
                                    method,
                                    reps,
                                    objective,
                                    objective_params,
                                    init_mode,
                                    init_seed,
                                    initial_assignment,
                                    optimizer_seed,
                                    args.maxiter,
                                    args.shots,
                                    schedule_seed,
                                )
                            except Exception as exc:
                                append_failure(
                                    {
                                        "run_id": run_id,
                                        "instance_id": manifest_row["instance_id"],
                                        "method": method,
                                        "objective": objective_label(objective, {**objective_params, "ratio": objective_params.get("ratio", 0.0)}),
                                        "reps": reps,
                                        "init_mode": init_mode,
                                        "optimizer_seed": optimizer_seed,
                                        "schedule_seed": schedule_seed,
                                        "error_type": type(exc).__name__,
                                        "error_message": str(exc),
                                    },
                                    failure_path,
                                )
                                continue
                            append_csv(row, run_path)
                            append_csv(opt_row, optimizer_path)
                            completed.add(row["run_id"])
                            run_rows.append(row)

    aggregate_rows = aggregate(run_rows)
    aggregate_path = out_dir / "paper_aggregate.csv"
    if aggregate_rows:
        with aggregate_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    report = {
        "config": vars(args),
        "manifest_path": str(manifest_path),
        "manifest_schema_version": manifest_payload.get("schema_version"),
        "manifest_hash": manifest_payload.get("manifest_hash"),
        "run_rows": len(run_rows),
        "aggregate_rows": len(aggregate_rows),
        "elapsed_seconds": time.time() - start,
        "outputs": {
            "run_level": str(run_path),
            "optimizer_diagnostics": str(optimizer_path),
            "aggregate": str(aggregate_path),
            "manifest": str(manifest_path),
            "failures": str(failure_path),
        },
        "software_versions": software_versions(),
    }
    write_json(out_dir / "paper_suite_report.json", report)
    print(f"Wrote paper suite outputs to {out_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
