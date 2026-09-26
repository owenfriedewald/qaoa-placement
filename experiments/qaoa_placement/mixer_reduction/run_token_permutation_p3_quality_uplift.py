"""Narrow p=3 quality-uplift study for token/permutation QAOA."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
QAOA_DIR = ROOT / "experiments/qaoa_placement"
MIXER_DIR = QAOA_DIR / "mixer_reduction"
for path in (str(QAOA_DIR), str(MIXER_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_suite import problem_from_manifest
from objective_functions import cvar_cost
from placement_core import Assignment, PlacementProblem, hpwl
from register_partial_swap import append_phase_estimation_partial_swap
from run_token_permutation_quality_validation import (
    FIXED_BETA,
    FIXED_GAMMA,
    INIT_MODES,
    MANIFEST,
    MAXITER,
    OPTIMIZER_SEEDS,
    OUT,
    SHOTS,
    build_reduced_model,
    evaluate_distribution,
    init_assignment,
    load_manifest,
    token_swap_pairs_by_edge,
    write_csv,
)
from run_token_permutation_study import circuit_metrics, route
from token_permutation_encoding import (
    assignment_to_token_sites,
    token_data_qubits,
    token_graph_edges,
    token_permutation_graph_metrics,
    token_phase_circuit,
    token_qubit,
    token_register_width,
)


GRAPHS = ("line", "ring")
SCHEDULES = (
    "current",
    "reversed",
    "alternating_direction",
    "edge_colored",
    "empty_prioritized",
    "real_prioritized",
    "rotating",
)
OBJECTIVES = ("cvar_0.25", "optimal_probability", "useful_feasible", "balanced_opt_useful")
P = 3
OPTIONAL_P4 = False
TOKEN_P3_BASELINE = {
    "line": {"routed_cx": 4550, "routed_depth": 4624},
    "ring": {"routed_cx": 4942, "routed_depth": 5337},
}
OCCUPANT_P3 = {"optimal_probability": 0.17519501432490678, "useful_feasible_per_1000": 756.7872917092212, "routed_cx": 12114, "routed_depth": 8347}


def fixed_parameters(reps: int) -> np.ndarray:
    return np.asarray([FIXED_GAMMA, FIXED_BETA] * reps, dtype=float)


def initial_parameters(reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    init: list[float] = []
    for layer in range(reps):
        init.extend([math.pi / (layer + 1), math.pi / (8 * (layer + 1))])
    return np.asarray(init, dtype=float) + rng.normal(0.0, 0.02, size=2 * reps)


def edge_color_order(edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    remaining = list(edges)
    ordered: list[tuple[int, int]] = []
    while remaining:
        used: set[int] = set()
        group: list[tuple[int, int]] = []
        next_remaining: list[tuple[int, int]] = []
        for edge in remaining:
            if edge[0] in used or edge[1] in used:
                next_remaining.append(edge)
            else:
                group.append(edge)
                used.update(edge)
        ordered.extend(group)
        remaining = next_remaining
    return tuple(ordered)


def scheduled_token_edges(problem: PlacementProblem, graph: str, schedule: str, layer: int) -> tuple[tuple[int, int], ...]:
    base = tuple(token_graph_edges(problem, graph))
    n = len(problem.cells)
    if schedule == "current":
        return base
    if schedule == "reversed":
        return tuple(reversed(base))
    if schedule == "alternating_direction":
        return base if layer % 2 == 0 else tuple(reversed(base))
    if schedule == "edge_colored":
        return edge_color_order(base)
    if schedule == "empty_prioritized":
        return tuple(sorted(base, key=lambda edge: (not (edge[0] >= n or edge[1] >= n), edge)))
    if schedule == "real_prioritized":
        return tuple(sorted(base, key=lambda edge: (not (edge[0] < n and edge[1] < n), edge)))
    if schedule == "rotating":
        if not base:
            return base
        offset = layer % len(base)
        return base[offset:] + base[:offset]
    raise ValueError(schedule)


def scheduled_pairs_by_layer(model, graph: str, schedule: str, reps: int) -> tuple[tuple[tuple[tuple[int, int], ...], ...], ...]:
    per_base_edge = dict(zip(token_graph_edges(model.problem, graph), token_swap_pairs_by_edge(model, graph)))
    layers: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for layer in range(reps):
        layers.append(tuple(per_base_edge[edge] for edge in scheduled_token_edges(model.problem, graph, schedule, layer)))
    return tuple(layers)


def reduced_probs_scheduled(model, initial_assignment: Mapping[str, int], graph: str, schedule: str, reps: int, theta: Sequence[float]) -> np.ndarray:
    state = np.zeros(len(model.states), dtype=np.complex128)
    state[model.index[assignment_to_token_sites(model.problem, initial_assignment)]] = 1.0
    pairs_by_layer = scheduled_pairs_by_layer(model, graph, schedule, reps)
    for layer in range(reps):
        gamma = float(theta[2 * layer])
        beta = float(theta[2 * layer + 1])
        state *= np.exp(-1j * gamma * model.costs)
        c = math.cos(beta)
        s = -1j * math.sin(beta)
        for edge_pairs in pairs_by_layer[layer]:
            for left, right in edge_pairs:
                left_amp = state[left]
                right_amp = state[right]
                state[left] = c * left_amp + s * right_amp
                state[right] = s * left_amp + c * right_amp
    probs = np.abs(state) ** 2
    return probs / float(np.sum(probs))


def objective_value(model, probs: np.ndarray, initial_assignment: Mapping[str, int], objective: str) -> float:
    if objective == "cvar_0.25":
        return float(cvar_cost(probs, model.costs, alpha=0.25))
    metrics = evaluate_distribution(model, probs, initial_assignment)
    opt = float(metrics["optimal_probability"])
    useful = float(metrics["probability_beating_initial"])
    if objective == "optimal_probability":
        return -opt
    if objective == "useful_feasible":
        return -useful
    if objective == "balanced_opt_useful":
        return -(0.5 * opt + 0.5 * useful)
    raise ValueError(objective)


def optimize_scheduled(
    model,
    initial_assignment: Mapping[str, int],
    graph: str,
    schedule: str,
    reps: int,
    seed: int,
    objective: str,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    prob_fn = lambda theta: reduced_probs_scheduled(model, initial_assignment, graph, schedule, reps, theta)
    evaluations = 0

    def objective_fn(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return objective_value(model, prob_fn(theta), initial_assignment, objective)

    start = time.time()
    result = minimize(
        objective_fn,
        initial_parameters(reps, seed),
        method="COBYLA",
        options={"maxiter": MAXITER, "rhobeg": 0.2, "tol": 1e-3},
    )
    elapsed = time.time() - start
    params = np.asarray(result.x, dtype=float)
    return prob_fn(params), params, float(result.fun), evaluations, elapsed


def run_case(instance: Mapping[str, object], graph: str, schedule: str, init_mode: str, protocol: str, objective: str, seed: int | None, reps: int = P) -> dict[str, object]:
    problem = problem_from_manifest(instance)
    model = build_reduced_model(problem)
    init_seed, initial = init_assignment(instance, init_mode)
    start = time.time()
    if protocol == "fixed_smoke":
        params = fixed_parameters(reps)
        probs = reduced_probs_scheduled(model, initial, graph, schedule, reps, params)
        final_objective = objective_value(model, probs, initial, "cvar_0.25")
        evals = 1
        opt_runtime = 0.0
    elif protocol == "optimized":
        if seed is None:
            raise ValueError("optimizer seed required")
        probs, params, final_objective, evals, opt_runtime = optimize_scheduled(model, initial, graph, schedule, reps, seed, objective)
    else:
        raise ValueError(protocol)
    metrics = evaluate_distribution(model, probs, initial)
    graph_metrics = token_permutation_graph_metrics(problem, graph)
    return {
        "run_id": "__".join(
            [
                str(instance["instance_id"]),
                graph,
                schedule,
                f"p{reps}",
                init_mode,
                objective,
                str(seed if seed is not None else "fixed"),
            ]
        ),
        "instance_id": instance["instance_id"],
        "family": instance["family"],
        "graph": graph,
        "p": reps,
        "mixer_schedule": schedule,
        "init_mode": init_mode,
        "init_seed": init_seed,
        "parameter_protocol": protocol,
        "optimization_objective": objective,
        "optimizer_seed": seed if seed is not None else "",
        "parameters": json.dumps([float(value) for value in params]),
        "optimizer_evaluations": evals,
        "optimizer_runtime_seconds": opt_runtime,
        "total_wall_clock_seconds": time.time() - start,
        "final_objective_value": final_objective,
        "connected": graph_metrics["connected"],
        "connected_components": graph_metrics["connected_components"],
        "diameter": graph_metrics["diameter"],
        "invalid_probability": 0.0,
        **metrics,
    }


def append_scheduled_mixer(circuit: QuantumCircuit, problem: PlacementProblem, beta: float, graph: str, schedule: str, layer: int) -> None:
    data = token_data_qubits(problem)
    ancilla = data
    width = token_register_width(problem)
    for token_a, token_b in scheduled_token_edges(problem, graph, schedule, layer):
        reg_a = [token_qubit(problem, token_a, bit) for bit in range(width)]
        reg_b = [token_qubit(problem, token_b, bit) for bit in range(width)]
        append_phase_estimation_partial_swap(circuit, reg_a, reg_b, ancilla, beta)


def scheduled_qaoa_circuit(problem: PlacementProblem, graph: str, schedule: str, reps: int) -> QuantumCircuit:
    data = token_data_qubits(problem)
    circuit = QuantumCircuit(data + 1)
    for layer in range(reps):
        phase = token_phase_circuit(problem, FIXED_GAMMA)
        circuit.compose(phase, qubits=range(data), inplace=True)
        append_scheduled_mixer(circuit, problem, FIXED_BETA, graph, schedule, layer)
    return circuit


def aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["parameter_protocol"] == "optimized":
            groups[(row["graph"], row["p"], row["mixer_schedule"], row["optimization_objective"])].append(row)
    output: list[dict[str, object]] = []
    for (graph, reps, schedule, objective), group in sorted(groups.items()):
        output.append(
            {
                "graph": graph,
                "p": reps,
                "mixer_schedule": schedule,
                "optimization_objective": objective,
                "rows": len(group),
                "instances": len({row["instance_id"] for row in group}),
                "feasibility_probability": float(np.mean([float(row["feasibility_probability"]) for row in group])),
                "invalid_probability": float(np.mean([float(row["invalid_probability"]) for row in group])),
                "optimal_probability": float(np.mean([float(row["optimal_probability"]) for row in group])),
                "discovery": float(np.mean([1.0 if row["discovery_hit"] else 0.0 for row in group])),
                "beat_initial_probability": float(np.mean([float(row["probability_beating_initial"]) for row in group])),
                "useful_feasible_per_1000": float(np.mean([float(row["useful_feasible_samples_per_1000"]) for row in group])),
                "cvar_0.25": float(np.mean([float(row["cvar_0.25"]) for row in group])),
                "expected_hpwl": float(np.mean([float(row["expected_hpwl"]) for row in group])),
                "entropy_real_placements": float(np.mean([float(row["entropy_real_placements"]) for row in group])),
                "runtime_seconds": float(np.sum([float(row["total_wall_clock_seconds"]) for row in group])),
                "aggregation_valid": all(abs(float(row["aggregated_real_probability_mass"]) - 1.0) < 1e-9 for row in group),
                "connected": all(bool(row["connected"]) for row in group),
            }
        )
    return output


def route_guardrail(problem: PlacementProblem, summary: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    selected: dict[tuple[str, str], Mapping[str, object]] = {}
    for graph in GRAPHS:
        candidates = [row for row in summary if row["graph"] == graph]
        if not candidates:
            continue
        selected[(graph, "best_optimal_probability")] = max(
            candidates, key=lambda row: (float(row["optimal_probability"]), float(row["useful_feasible_per_1000"]))
        )
        selected[(graph, "best_useful_feasible")] = max(
            candidates, key=lambda row: (float(row["useful_feasible_per_1000"]), float(row["optimal_probability"]))
        )
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for (graph, basis), best in selected.items():
        schedule = str(best["mixer_schedule"])
        if (graph, schedule) in seen:
            continue
        seen.add((graph, schedule))
        start = time.time()
        circuit = scheduled_qaoa_circuit(problem, graph, schedule, P)
        logical = circuit_metrics(circuit)
        routed = route(circuit)
        baseline = TOKEN_P3_BASELINE[graph]
        rows.append(
            {
                "graph": graph,
                "mixer_schedule": schedule,
                "selection_basis": basis,
                "selected_optimal_probability": best["optimal_probability"],
                "selected_useful_feasible_per_1000": best["useful_feasible_per_1000"],
                "baseline_routed_cx": baseline["routed_cx"],
                "baseline_routed_depth": baseline["routed_depth"],
                **logical,
                **routed,
                "cx_delta_vs_baseline": int(routed["routed_cx"]) - baseline["routed_cx"],
                "depth_delta_vs_baseline": int(routed["routed_depth"]) - baseline["routed_depth"],
                "runtime_total_seconds": time.time() - start,
            }
        )
    return rows


def write_decision(path: Path, summary: Sequence[Mapping[str, object]], routing: Sequence[Mapping[str, object]]) -> None:
    best_line = max([row for row in summary if row["graph"] == "line"], key=lambda row: float(row["useful_feasible_per_1000"]))
    best_ring = max([row for row in summary if row["graph"] == "ring"], key=lambda row: float(row["optimal_probability"]))
    best_opt = max(summary, key=lambda row: float(row["optimal_probability"]))
    routing_ok = all(int(row["routed_cx"]) < 5000 for row in routing)
    if float(best_opt["optimal_probability"]) >= 0.165 and routing_ok:
        outcome = "Outcome A: token p=3 substantially narrows the optimal-probability gap while preserving sub-5k routed CX."
    elif routing_ok:
        outcome = "Outcome B: useful sampling remains strong, but the optimal-probability gap remains."
    else:
        outcome = "Outcome C: quality tuning risks eroding the routing advantage."
    lines = [
        "# Token/Permutation p=3 Quality-Uplift Decision",
        "",
        "## Classification",
        "",
        outcome,
        "",
        "## Best Aggregate Results",
        "",
        f"- Best line schedule for useful sampling: `{best_line['mixer_schedule']}` optimized for `{best_line['optimization_objective']}` with optimal probability {float(best_line['optimal_probability']):.4f}, useful feasible / 1000 {float(best_line['useful_feasible_per_1000']):.1f}, and CVaR {float(best_line['cvar_0.25']):.4f}.",
        f"- Best ring schedule for optimal probability: `{best_ring['mixer_schedule']}` optimized for `{best_ring['optimization_objective']}` with optimal probability {float(best_ring['optimal_probability']):.4f}, useful feasible / 1000 {float(best_ring['useful_feasible_per_1000']):.1f}, and CVaR {float(best_ring['cvar_0.25']):.4f}.",
        f"- Occupant p=3 reference: optimal probability {OCCUPANT_P3['optimal_probability']:.4f}, useful feasible / 1000 {OCCUPANT_P3['useful_feasible_per_1000']:.1f}, routed CX {OCCUPANT_P3['routed_cx']}.",
        "",
        "## Routing Guardrail",
        "",
    ]
    for row in routing:
        lines.append(
            f"- `{row['graph']}` / `{row['mixer_schedule']}` routed to {row['routed_cx']} CX and depth {row['routed_depth']} "
            f"(delta vs baseline: {row['cx_delta_vs_baseline']} CX, {row['depth_delta_vs_baseline']} depth)."
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use the best token p=3 line point as the hardware-facing useful-sampling result and the best token p=3 ring point as the optimal-probability ablation. Do not continue broad schedule tuning unless a new schedule mechanism is proposed; the next step should compare the selected token schedules against occupant p=3 in the manuscript/resource narrative after direct author review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    start = time.time()
    instances = load_manifest()
    rows: list[dict[str, object]] = []
    schedule_rows: list[dict[str, object]] = []
    for instance in instances:
        for graph in GRAPHS:
            for schedule in SCHEDULES:
                for init_mode in INIT_MODES:
                    fixed = run_case(instance, graph, schedule, init_mode, "fixed_smoke", "cvar_0.25", None, P)
                    rows.append(fixed)
                    schedule_rows.append(fixed)
                    for objective in OBJECTIVES:
                        for seed in OPTIMIZER_SEEDS:
                            row = run_case(instance, graph, schedule, init_mode, "optimized", objective, seed, P)
                            rows.append(row)
    summary = aggregate(rows)
    routing = route_guardrail(problem_from_manifest(instances[0]), summary)
    write_csv(OUT / "token_permutation_p3_schedule_results.csv", schedule_rows)
    write_csv(OUT / "token_permutation_p3_parameter_results.csv", rows)
    write_csv(OUT / "token_permutation_p3_routing_guardrail.csv", routing)
    write_csv(OUT / "token_permutation_p3_quality_summary.csv", summary)
    write_decision(OUT / "token_permutation_p3_quality_decision.md", summary, routing)
    env = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "graphs": GRAPHS,
        "schedules": SCHEDULES,
        "objectives": OBJECTIVES,
        "p": P,
        "optional_p4_run": OPTIONAL_P4,
        "optimizer_seeds": OPTIMIZER_SEEDS,
        "maxiter": MAXITER,
        "shots": SHOTS,
        "runtime_seconds": time.time() - start,
    }
    (OUT / "token_permutation_p3_quality_environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} p=3 quality rows and {len(summary)} summary rows in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
