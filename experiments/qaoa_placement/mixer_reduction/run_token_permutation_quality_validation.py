"""Bounded quality validation for explicit-EMPTY token/permutation encoding.

The simulation is performed in the legal token-permutation subspace.  Token
states are full permutations of site labels over real-cell and EMPTY-token
registers, and output metrics aggregate over distinguishable EMPTY-token
permutations before comparing real placements.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
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
from token_permutation_encoding import (
    assignment_to_token_sites,
    empty_token_count,
    legal_token_permutations,
    real_assignment_key,
    token_graph_edges,
    token_permutation_graph_metrics,
    token_sites_to_assignment,
)


OUT = MIXER_DIR
MANIFEST = QAOA_DIR / "paper_suite_results/audit_061026/preflight_manifest_one_per_family.json"
OCCUPANT_REFERENCE = OUT / "structured_binary_robustness_run_level.csv"
TOKEN_GRAPHS = ("line", "ring", "complete", "empty_star", "real_empty_priority")
INIT_MODES = ("deterministic", "poor")
OPTIMIZER_SEEDS = (11, 17)
DEPTHS = (1, 2)
SHOTS = 4096
MAXITER = 40
FIXED_GAMMA = 0.17
FIXED_BETA = 0.23


@dataclass(frozen=True)
class ReducedTokenModel:
    problem: PlacementProblem
    states: tuple[tuple[int, ...], ...]
    index: dict[tuple[int, ...], int]
    costs: np.ndarray
    assignment_keys: tuple[tuple[int, ...], ...]
    real_costs: dict[tuple[int, ...], float]
    exact_cost: float
    optimal_keys: frozenset[tuple[int, ...]]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest() -> list[Mapping[str, object]]:
    with MANIFEST.open() as handle:
        return list(json.load(handle)["instances"])


def init_assignment(instance: Mapping[str, object], init_mode: str) -> tuple[int, Assignment]:
    for row in instance["initializations"]:  # type: ignore[index]
        if row["init_mode"] == init_mode:
            return int(row["init_seed"]), dict(row["assignment"])
    raise ValueError(f"missing initialization {init_mode}")


def build_reduced_model(problem: PlacementProblem) -> ReducedTokenModel:
    states = tuple(legal_token_permutations(problem))
    index = {state: idx for idx, state in enumerate(states)}
    costs: list[float] = []
    keys: list[tuple[int, ...]] = []
    real_costs: dict[tuple[int, ...], float] = {}
    for state in states:
        assignment = token_sites_to_assignment(problem, state)
        if assignment is None:
            raise ValueError("legal_token_permutations emitted invalid state")
        key = real_assignment_key(problem, assignment)
        cost = float(hpwl(problem, assignment))
        costs.append(cost)
        keys.append(key)
        real_costs[key] = cost
    exact = min(real_costs.values())
    optimal = frozenset(key for key, cost in real_costs.items() if abs(cost - exact) < 1e-9)
    return ReducedTokenModel(problem, states, index, np.asarray(costs, dtype=float), tuple(keys), real_costs, exact, optimal)


def token_swap_pairs_by_edge(model: ReducedTokenModel, graph: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    edge_pairs: list[tuple[tuple[int, int], ...]] = []
    for left_token, right_token in token_graph_edges(model.problem, graph):
        pairs: set[tuple[int, int]] = set()
        for idx, state in enumerate(model.states):
            state_list = list(state)
            moved = state_list[:]
            moved[left_token], moved[right_token] = moved[right_token], moved[left_token]
            jdx = model.index[tuple(moved)]
            if idx != jdx:
                pairs.add((idx, jdx) if idx < jdx else (jdx, idx))
        edge_pairs.append(tuple(sorted(pairs)))
    return tuple(edge_pairs)


def reduced_token_probabilities(
    model: ReducedTokenModel,
    initial_assignment: Mapping[str, int],
    graph: str,
    reps: int,
    theta: Sequence[float],
) -> np.ndarray:
    if len(theta) != 2 * reps:
        raise ValueError(f"expected {2 * reps} parameters, got {len(theta)}")
    state = np.zeros(len(model.states), dtype=np.complex128)
    initial_state = assignment_to_token_sites(model.problem, initial_assignment)
    state[model.index[initial_state]] = 1.0
    pairs_by_edge = token_swap_pairs_by_edge(model, graph)
    for layer in range(reps):
        gamma = float(theta[2 * layer])
        beta = float(theta[2 * layer + 1])
        state *= np.exp(-1j * gamma * model.costs)
        c = math.cos(beta)
        s = -1j * math.sin(beta)
        for pairs in pairs_by_edge:
            for left, right in pairs:
                left_amp = state[left]
                right_amp = state[right]
                state[left] = c * left_amp + s * right_amp
                state[right] = s * left_amp + c * right_amp
    probs = np.abs(state) ** 2
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("zero probability state")
    return probs / total


def aggregate_token_probs(model: ReducedTokenModel, probs: np.ndarray) -> dict[tuple[int, ...], float]:
    aggregated: dict[tuple[int, ...], float] = defaultdict(float)
    for idx, probability in enumerate(probs):
        aggregated[model.assignment_keys[idx]] += float(probability)
    return dict(aggregated)


def cvar_from_aggregated(aggregated: Mapping[tuple[int, ...], float], costs: Mapping[tuple[int, ...], float], alpha: float = 0.25) -> float:
    ordered = sorted(costs, key=lambda key: costs[key])
    remaining = alpha
    weighted = 0.0
    for key in ordered:
        if remaining <= 0.0:
            break
        mass = min(float(aggregated.get(key, 0.0)), remaining)
        weighted += mass * float(costs[key])
        remaining -= mass
    if remaining > 1e-12:
        return float(costs[ordered[-1]])
    return weighted / alpha


def evaluate_distribution(
    model: ReducedTokenModel,
    probs: np.ndarray,
    initial_assignment: Mapping[str, int],
) -> dict[str, object]:
    aggregated = aggregate_token_probs(model, probs)
    initial_key = real_assignment_key(model.problem, initial_assignment)
    initial_cost = float(hpwl(model.problem, initial_assignment))
    optimum_probability = float(sum(aggregated.get(key, 0.0) for key in model.optimal_keys))
    probability_beating_initial = float(
        sum(prob for key, prob in aggregated.items() if model.real_costs[key] < initial_cost - 1e-9)
    )
    expected_cost = float(sum(prob * model.real_costs[key] for key, prob in aggregated.items()))
    nonzero_costs = [model.real_costs[key] for key, prob in aggregated.items() if prob > 1e-12]
    best_cost = min(nonzero_costs) if nonzero_costs else math.inf
    entropy = -sum(prob * math.log(prob) for prob in aggregated.values() if prob > 1e-15)
    mass = float(sum(aggregated.values()))
    return {
        "feasibility_probability": 1.0,
        "aggregated_real_probability_mass": mass,
        "invalid_token_probability": 0.0,
        "empty_aggregation_valid": abs(mass - 1.0) < 1e-9,
        "optimal_probability": optimum_probability,
        "discovery_hit": optimum_probability > 1e-12,
        "expected_hpwl": expected_cost,
        "best_shot_hpwl": best_cost,
        "best_shot_ratio": best_cost / model.exact_cost if model.exact_cost else math.inf,
        "cvar_0.25": cvar_from_aggregated(aggregated, model.real_costs, alpha=0.25),
        "probability_beating_initial": probability_beating_initial,
        "useful_feasible_samples_per_1000": 1000.0 * probability_beating_initial,
        "distance_0_probability": float(aggregated.get(initial_key, 0.0)),
        "entropy_real_placements": entropy,
        "expected_shots_to_optimum": (1.0 / optimum_probability) if optimum_probability > 0 else math.inf,
        "optimum_preserved_after_aggregation": all(key in aggregated for key in model.optimal_keys),
        "real_placement_count": len(aggregated),
        "token_state_count": len(model.states),
        "exact_hpwl": model.exact_cost,
        "initial_hpwl": initial_cost,
    }


def optimize_token_probs(
    prob_fn: Callable[[np.ndarray], np.ndarray],
    costs: np.ndarray,
    reps: int,
    seed: int,
    maxiter: int = MAXITER,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    evaluations = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return float(cvar_cost(prob_fn(theta), costs, alpha=0.25))

    rng = np.random.default_rng(seed)
    init: list[float] = []
    for layer in range(reps):
        init.extend([math.pi / (layer + 1), math.pi / (8 * (layer + 1))])
    params0 = np.asarray(init, dtype=float) + rng.normal(0.0, 0.02, size=2 * reps)
    start = time.time()
    result = minimize(objective, params0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2, "tol": 1e-3})
    elapsed = time.time() - start
    params = np.asarray(result.x, dtype=float)
    return prob_fn(params), params, float(result.fun), evaluations, elapsed


def run_token_case(
    instance: Mapping[str, object],
    graph: str,
    init_mode: str,
    reps: int,
    protocol: str,
    optimizer_seed: int | None,
) -> tuple[dict[str, object], dict[str, object]]:
    problem = problem_from_manifest(instance)
    model = build_reduced_model(problem)
    init_seed, initial_assignment = init_assignment(instance, init_mode)
    start = time.time()
    prob_fn = lambda theta: reduced_token_probabilities(model, initial_assignment, graph, reps, theta)
    if protocol == "fixed_smoke":
        params = np.asarray([FIXED_GAMMA, FIXED_BETA] * reps, dtype=float)
        probs = prob_fn(params)
        objective_value = float(cvar_cost(probs, model.costs, alpha=0.25))
        evaluations = 1
        opt_runtime = 0.0
    elif protocol == "optimized_cvar_0.25":
        if optimizer_seed is None:
            raise ValueError("optimizer_seed required for optimized protocol")
        probs, params, objective_value, evaluations, opt_runtime = optimize_token_probs(prob_fn, model.costs, reps, optimizer_seed)
    else:
        raise ValueError(protocol)
    elapsed = time.time() - start
    metrics = evaluate_distribution(model, probs, initial_assignment)
    graph_metrics = token_permutation_graph_metrics(problem, graph)
    row = {
        "run_id": "__".join(
            [
                str(instance["instance_id"]),
                f"token_permutation_{graph}",
                protocol,
                f"p{reps}",
                init_mode,
                str(init_seed),
                str(optimizer_seed if optimizer_seed is not None else "fixed"),
            ]
        ),
        "instance_id": instance["instance_id"],
        "family": instance["family"],
        "architecture": "token_permutation",
        "method": f"token_permutation_{graph}",
        "mixer_graph": graph,
        "parameter_protocol": protocol,
        "objective": "cvar_0.25" if protocol == "optimized_cvar_0.25" else "fixed_smoke",
        "p": reps,
        "init_mode": init_mode,
        "init_seed": init_seed,
        "optimizer_seed": optimizer_seed if optimizer_seed is not None else "",
        "shots": SHOTS,
        "maxiter": MAXITER if protocol == "optimized_cvar_0.25" else 0,
        "optimizer_evaluations": evaluations,
        "optimizer_runtime_seconds": opt_runtime,
        "total_wall_clock_seconds": elapsed,
        "final_objective_value": objective_value,
        "parameters": json.dumps([float(value) for value in params]),
        "num_cells": len(problem.cells),
        "num_sites": len(problem.sites),
        "empty_tokens": empty_token_count(problem),
        "empty_degeneracy": math.factorial(empty_token_count(problem)),
        "connected": graph_metrics["connected"],
        "connected_components": graph_metrics["connected_components"],
        "diameter": graph_metrics["diameter"],
        "token_edges_per_layer": graph_metrics["token_edges_per_layer"],
        **metrics,
    }
    sweep = {
        "run_id": row["run_id"],
        "instance_id": row["instance_id"],
        "family": row["family"],
        "mixer_graph": graph,
        "p": reps,
        "init_mode": init_mode,
        "parameter_protocol": protocol,
        "optimizer_seed": row["optimizer_seed"],
        "gamma_beta_parameters": row["parameters"],
        "cvar_0.25": row["cvar_0.25"],
        "optimal_probability": row["optimal_probability"],
        "useful_feasible_samples_per_1000": row["useful_feasible_samples_per_1000"],
        "probability_beating_initial": row["probability_beating_initial"],
        "runtime_seconds": row["total_wall_clock_seconds"],
    }
    return row, sweep


def summarize_token(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["p"], row["parameter_protocol"], row["init_mode"])
        groups[key].append(row)
    summary: list[dict[str, object]] = []
    for (method, reps, protocol, init_mode), group in sorted(groups.items()):
        for label, subset in (("by_start", group),):
            summary.append(
                {
                    "architecture": "token_permutation",
                    "method": method,
                    "p": reps,
                    "parameter_protocol": protocol,
                    "init_mode": init_mode,
                    "summary_scope": label,
                    "rows": len(subset),
                    "instances": len({row["instance_id"] for row in subset}),
                    "mean_feasibility": float(np.mean([float(row["feasibility_probability"]) for row in subset])),
                    "mean_discovery": float(np.mean([1.0 if row["discovery_hit"] else 0.0 for row in subset])),
                    "mean_optimal_probability": float(np.mean([float(row["optimal_probability"]) for row in subset])),
                    "median_optimal_probability": float(np.median([float(row["optimal_probability"]) for row in subset])),
                    "mean_probability_beating_initial": float(np.mean([float(row["probability_beating_initial"]) for row in subset])),
                    "mean_useful_feasible_per_1000": float(np.mean([float(row["useful_feasible_samples_per_1000"]) for row in subset])),
                    "mean_distance_0_probability": float(np.mean([float(row["distance_0_probability"]) for row in subset])),
                    "mean_expected_hpwl": float(np.mean([float(row["expected_hpwl"]) for row in subset])),
                    "mean_cvar_0.25": float(np.mean([float(row["cvar_0.25"]) for row in subset])),
                    "all_empty_aggregation_valid": all(bool(row["empty_aggregation_valid"]) for row in subset),
                }
            )
    for key_fields in (("method", "p", "parameter_protocol"), ("method", "p")):
        groups2: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            groups2[tuple(row[field] for field in key_fields)].append(row)
        for key, group in sorted(groups2.items()):
            summary.append(
                {
                    "architecture": "token_permutation",
                    **{field: value for field, value in zip(key_fields, key)},
                    "init_mode": "combined",
                    "summary_scope": "combined",
                    "rows": len(group),
                    "instances": len({row["instance_id"] for row in group}),
                    "mean_feasibility": float(np.mean([float(row["feasibility_probability"]) for row in group])),
                    "mean_discovery": float(np.mean([1.0 if row["discovery_hit"] else 0.0 for row in group])),
                    "mean_optimal_probability": float(np.mean([float(row["optimal_probability"]) for row in group])),
                    "median_optimal_probability": float(np.median([float(row["optimal_probability"]) for row in group])),
                    "mean_probability_beating_initial": float(np.mean([float(row["probability_beating_initial"]) for row in group])),
                    "mean_useful_feasible_per_1000": float(np.mean([float(row["useful_feasible_samples_per_1000"]) for row in group])),
                    "mean_distance_0_probability": float(np.mean([float(row["distance_0_probability"]) for row in group])),
                    "mean_expected_hpwl": float(np.mean([float(row["expected_hpwl"]) for row in group])),
                    "mean_cvar_0.25": float(np.mean([float(row["cvar_0.25"]) for row in group])),
                    "all_empty_aggregation_valid": all(bool(row["empty_aggregation_valid"]) for row in group),
                }
            )
    return summary


def load_occupant_reference() -> list[dict[str, object]]:
    if not OCCUPANT_REFERENCE.exists():
        return []
    with OCCUPANT_REFERENCE.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle) if row["method"] == "structured_binary_pswap_grid"]


def compare_with_occupant(token_rows: Sequence[Mapping[str, object]], occupant_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    occupant_index = {
        (row["instance_id"], row["init_mode"], str(row["optimizer_seed"])): row
        for row in occupant_rows
    }
    comparable = [
        row
        for row in token_rows
        if row["parameter_protocol"] == "optimized_cvar_0.25" and row["p"] in (1, 2)
    ]
    best_by_key: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in comparable:
        key = (row["instance_id"], row["init_mode"], str(row["optimizer_seed"]), row["p"])
        current = best_by_key.get(key)
        if current is None or float(row["useful_feasible_samples_per_1000"]) > float(current["useful_feasible_samples_per_1000"]):
            best_by_key[key] = row
    comparisons: list[dict[str, object]] = []
    for (instance_id, init_mode, optimizer_seed, reps), token in sorted(best_by_key.items()):
        occ = occupant_index.get((str(instance_id), str(init_mode), str(optimizer_seed)))
        if occ is None:
            continue
        comparisons.append(
            {
                "instance_id": instance_id,
                "family": token["family"],
                "init_mode": init_mode,
                "optimizer_seed": optimizer_seed,
                "token_p": reps,
                "token_best_method": token["method"],
                "token_optimal_probability": token["optimal_probability"],
                "occupant_reference_p": 3,
                "occupant_optimal_probability": occ["optimal_probability"],
                "delta_optimal_probability": float(token["optimal_probability"]) - float(occ["optimal_probability"]),
                "token_useful_feasible_per_1000": token["useful_feasible_samples_per_1000"],
                "occupant_useful_feasible_per_1000": occ["useful_feasible_samples_per_1000"],
                "delta_useful_feasible_per_1000": float(token["useful_feasible_samples_per_1000"])
                - float(occ["useful_feasible_samples_per_1000"]),
                "token_probability_beating_initial": token["probability_beating_initial"],
                "occupant_probability_beating_initial": occ["probability_beating_initial"],
                "delta_probability_beating_initial": float(token["probability_beating_initial"])
                - float(occ["probability_beating_initial"]),
                "comparison_note": "occupant reference is existing optimized structured-binary p=3 bounded robustness row",
            }
        )
    return comparisons


def write_decision(
    path: Path,
    token_summary: Sequence[Mapping[str, object]],
    comparisons: Sequence[Mapping[str, object]],
    elapsed: float,
) -> None:
    optimized = [row for row in token_summary if row.get("parameter_protocol") == "optimized_cvar_0.25" and row.get("summary_scope") == "combined"]
    best = max(optimized, key=lambda row: float(row["mean_useful_feasible_per_1000"])) if optimized else None
    p1_best = max(
        [row for row in optimized if int(row["p"]) == 1],
        key=lambda row: float(row["mean_useful_feasible_per_1000"]),
        default=None,
    )
    p2_best = max(
        [row for row in optimized if int(row["p"]) == 2],
        key=lambda row: float(row["mean_useful_feasible_per_1000"]),
        default=None,
    )
    comp_p2 = [row for row in comparisons if int(row["token_p"]) == 2]
    useful_wins = sum(1 for row in comp_p2 if float(row["delta_useful_feasible_per_1000"]) > 0)
    opt_wins = sum(1 for row in comp_p2 if float(row["delta_optimal_probability"]) > 0)
    if best and float(best["mean_useful_feasible_per_1000"]) >= 0.75 * np.mean(
        [float(row["occupant_useful_feasible_per_1000"]) for row in comparisons] or [1.0]
    ):
        outcome = "Outcome A/B: token/permutation is competitive enough to promote as the next architecture branch, with quality still graph- and depth-dependent."
    else:
        outcome = "Outcome B/C: token/permutation keeps the routing advantage but needs mixer-graph or parameter follow-up before replacing the occupant-register reference."
    lines = [
        "# Token/Permutation Quality Decision",
        "",
        f"Generated in {elapsed:.1f} s.",
        "",
        "## Decision",
        "",
        outcome,
        "",
        "The pass used reduced legal-token-subspace simulation and aggregated distinguishable EMPTY-token permutations before computing real-placement metrics.",
        "",
        "## Best Token Results",
        "",
    ]
    for label, row in (("Overall", best), ("p=1", p1_best), ("p=2", p2_best)):
        if row is None:
            continue
        lines.append(
            f"- {label}: `{row['method']}` at p={row['p']} with mean optimal probability "
            f"{float(row['mean_optimal_probability']):.4f}, mean beat-initial probability "
            f"{float(row['mean_probability_beating_initial']):.4f}, and "
            f"{float(row['mean_useful_feasible_per_1000']):.1f} useful feasible samples per 1000."
        )
    lines.extend(
        [
            "",
            "## Occupant Reference Comparison",
            "",
            "The occupant-register reference is the existing optimized structured-binary p=3 bounded robustness result, not a rerun in this pass.",
            f"For p=2 token best-by-run comparisons, token beat the occupant p=3 reference on useful feasible samples in {useful_wins}/{len(comp_p2)} paired rows and on optimal probability in {opt_wins}/{len(comp_p2)} paired rows.",
            "",
            "Known routed references retained for architecture context:",
            "",
            "- Token/permutation line p=1: 1451 CX / depth 1696.",
            "- Token/permutation line p=2: 3005 CX / depth 3245.",
            "- Occupant-register p=1 best: 3768 CX / depth 2707.",
            "- Occupant-register p=2 best: 7619 CX / depth 4905.",
            "",
            "## Recommendation",
            "",
            "Promote token/permutation as the leading low-routing-cost architecture branch only if the paired quality losses are acceptable for the next paper story; otherwise run one bounded mixer-graph/parameter follow-up centered on the best p=2 graph from this study. The next implementation should keep the reduced token simulator as the common quality harness and add direct comparison to the occupant reference under matched p/depth if needed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    start = time.time()
    instances = load_manifest()
    quality_rows: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    for instance in instances:
        for init_mode in INIT_MODES:
            for reps in DEPTHS:
                for graph in TOKEN_GRAPHS:
                    row, sweep = run_token_case(instance, graph, init_mode, reps, "fixed_smoke", None)
                    quality_rows.append(row)
                    sweep_rows.append(sweep)
                    for optimizer_seed in OPTIMIZER_SEEDS:
                        row, sweep = run_token_case(instance, graph, init_mode, reps, "optimized_cvar_0.25", optimizer_seed)
                        quality_rows.append(row)
                        sweep_rows.append(sweep)
    summary_rows = summarize_token(quality_rows)
    occupant_rows = load_occupant_reference()
    comparisons = compare_with_occupant(quality_rows, occupant_rows)

    write_csv(OUT / "token_permutation_quality_validation.csv", quality_rows)
    write_csv(OUT / "token_permutation_quality_summary.csv", summary_rows)
    write_csv(OUT / "token_permutation_vs_occupant_reference.csv", comparisons)
    write_csv(OUT / "token_permutation_parameter_sweep.csv", sweep_rows)
    elapsed = time.time() - start
    write_decision(OUT / "token_permutation_quality_decision.md", summary_rows, comparisons, elapsed)
    env = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "occupant_reference": str(OCCUPANT_REFERENCE.relative_to(ROOT)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "instances": len(instances),
        "token_graphs": TOKEN_GRAPHS,
        "depths": DEPTHS,
        "init_modes": INIT_MODES,
        "optimizer_seeds": OPTIMIZER_SEEDS,
        "fixed_parameters": {"gamma": FIXED_GAMMA, "beta": FIXED_BETA},
        "maxiter": MAXITER,
        "shots": SHOTS,
        "runtime_seconds": elapsed,
    }
    (OUT / "token_permutation_quality_environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(quality_rows)} token quality rows in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
