"""Evaluate frozen real-placement windows with exact, classical, and QAOA methods.

The QAOA row is ideal reduced-permutation simulation.  It uses the previously
selected explicit-EMPTY token architecture (line graph, rotating schedule,
p=3) and optimizes only deployable CVaR(0.25), never exact-optimum probability.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import permutations
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import minimize

from .placement_windows import canonical_manifest_hash, sha256


P = 3
CVaR_ALPHA = 0.25
EVALUATION_BUDGET = 40
READOUT_SHOTS = 4096
SEEDS = (11, 17)
METHODS = ("unchanged", "exact", "greedy", "random_search", "simulated_annealing", "token_qaoa")


@dataclass(frozen=True)
class Landscape:
    assignments: tuple[tuple[int, ...], ...]
    costs: np.ndarray
    cost_by_assignment: Mapping[tuple[int, ...], int]
    initial: tuple[int, ...]
    exact_cost: int
    maximum_cost: int


def load_landscape(window: Mapping[str, object]) -> Landscape:
    rows = [(tuple(int(site) for site in row[0]), int(row[1])) for row in window["cost_landscape"]]
    assignments = tuple(row[0] for row in rows)
    cost_by_assignment = {assignment: cost for assignment, cost in rows}
    initial_map = {str(key): int(value) for key, value in window["initial_assignment"].items()}
    cells = [str(cell["cell_id"]) for cell in window["movable_cells"]]
    initial = tuple(initial_map[cell] for cell in cells)
    if len(cost_by_assignment) != math.perm(int(window["site_count"]), int(window["cell_count"])):
        raise ValueError(f"incomplete landscape for {window['window_id']}")
    return Landscape(
        assignments=assignments,
        costs=np.asarray([cost_by_assignment[assignment] for assignment in assignments], dtype=float),
        cost_by_assignment=cost_by_assignment,
        initial=initial,
        exact_cost=min(cost_by_assignment.values()),
        maximum_cost=max(cost_by_assignment.values()),
    )


def neighbors(assignment: tuple[int, ...], site_count: int) -> tuple[tuple[int, ...], ...]:
    output: set[tuple[int, ...]] = set()
    for cell_index, old_site in enumerate(assignment):
        for new_site in range(site_count):
            if new_site == old_site:
                continue
            moved = list(assignment)
            if new_site in assignment:
                other = assignment.index(new_site)
                moved[cell_index], moved[other] = moved[other], moved[cell_index]
            else:
                moved[cell_index] = new_site
            output.add(tuple(moved))
    return tuple(sorted(output))


def greedy_search(landscape: Landscape, budget: int) -> tuple[tuple[int, ...], int, int]:
    current = landscape.initial
    current_cost = int(landscape.cost_by_assignment[current])
    evaluations = 0
    while evaluations < budget:
        best = current
        best_cost = current_cost
        for candidate in neighbors(current, 6):
            if evaluations == budget:
                break
            cost = int(landscape.cost_by_assignment[candidate])
            evaluations += 1
            if (cost, candidate) < (best_cost, best):
                best, best_cost = candidate, cost
        if best_cost >= current_cost:
            break
        current, current_cost = best, best_cost
    return current, current_cost, evaluations


def random_search(landscape: Landscape, budget: int, seed: int) -> tuple[tuple[int, ...], int, int]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(landscape.assignments))[:budget]
    best = landscape.initial
    best_cost = int(landscape.cost_by_assignment[best])
    for index in order:
        candidate = landscape.assignments[int(index)]
        cost = int(landscape.costs[int(index)])
        if (cost, candidate) < (best_cost, best):
            best, best_cost = candidate, cost
    return best, best_cost, len(order)


def simulated_annealing(
    landscape: Landscape, budget: int, seed: int
) -> tuple[tuple[int, ...], int, int]:
    rng = np.random.default_rng(seed)
    current = landscape.initial
    current_cost = int(landscape.cost_by_assignment[current])
    best, best_cost = current, current_cost
    span = max(landscape.maximum_cost - landscape.exact_cost, 1)
    for step in range(budget):
        candidate_rows = neighbors(current, 6)
        candidate = candidate_rows[int(rng.integers(0, len(candidate_rows)))]
        candidate_cost = int(landscape.cost_by_assignment[candidate])
        fraction = step / max(budget - 1, 1)
        temperature = span * (0.20 * (0.005 / 0.20) ** fraction)
        delta = candidate_cost - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            current, current_cost = candidate, candidate_cost
        if (current_cost, current) < (best_cost, best):
            best, best_cost = current, current_cost
    return best, best_cost, budget


TOKEN_STATES = tuple(permutations(range(6), 6))
TOKEN_INDEX = {state: index for index, state in enumerate(TOKEN_STATES)}
LINE_EDGES = tuple((index, index + 1) for index in range(5))


def _swap_pairs(edge: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    left_rows: list[int] = []
    right_rows: list[int] = []
    for index, state in enumerate(TOKEN_STATES):
        moved = list(state)
        moved[edge[0]], moved[edge[1]] = moved[edge[1]], moved[edge[0]]
        other = TOKEN_INDEX[tuple(moved)]
        if index < other:
            left_rows.append(index)
            right_rows.append(other)
    return np.asarray(left_rows), np.asarray(right_rows)


PAIR_ROWS = {edge: _swap_pairs(edge) for edge in LINE_EDGES}


def qaoa_probabilities(
    landscape: Landscape, normalized_state_costs: np.ndarray, theta: Sequence[float]
) -> np.ndarray:
    state = np.zeros(len(TOKEN_STATES), dtype=np.complex128)
    empty_sites = tuple(site for site in range(6) if site not in landscape.initial)
    initial_state = landscape.initial + empty_sites
    state[TOKEN_INDEX[initial_state]] = 1.0
    for layer in range(P):
        gamma = float(theta[2 * layer])
        beta = float(theta[2 * layer + 1])
        state *= np.exp(-1j * gamma * normalized_state_costs)
        offset = layer % len(LINE_EDGES)
        edge_order = LINE_EDGES[offset:] + LINE_EDGES[:offset]
        c = math.cos(beta)
        s = -1j * math.sin(beta)
        for edge in edge_order:
            left, right = PAIR_ROWS[edge]
            left_amp = state[left].copy()
            right_amp = state[right].copy()
            state[left] = c * left_amp + s * right_amp
            state[right] = s * left_amp + c * right_amp
    probabilities = np.abs(state) ** 2
    return probabilities / float(np.sum(probabilities))


def aggregate_qaoa(landscape: Landscape, state_probabilities: np.ndarray) -> np.ndarray:
    assignment_index = {assignment: index for index, assignment in enumerate(landscape.assignments)}
    aggregated = np.zeros(len(landscape.assignments), dtype=float)
    for state, probability in zip(TOKEN_STATES, state_probabilities):
        aggregated[assignment_index[state[:4]]] += float(probability)
    return aggregated


def cvar(costs: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    order = np.argsort(costs, kind="stable")
    remaining = alpha
    total = 0.0
    for index in order:
        mass = min(float(probabilities[int(index)]), remaining)
        total += mass * float(costs[int(index)])
        remaining -= mass
        if remaining <= 1e-15:
            break
    if remaining > 1e-10:
        raise ValueError("probability mass is incomplete")
    return total / alpha


def expected_best(costs: np.ndarray, probabilities: np.ndarray, shots: int) -> float:
    grouped: dict[float, float] = {}
    for cost, probability in zip(costs, probabilities):
        grouped[float(cost)] = grouped.get(float(cost), 0.0) + float(probability)
    cumulative = 0.0
    expectation = 0.0
    for cost in sorted(grouped):
        previous_survival = (1.0 - cumulative) ** shots
        cumulative += grouped[cost]
        next_survival = max(0.0, 1.0 - cumulative) ** shots
        expectation += cost * (previous_survival - next_survival)
    return expectation


def run_qaoa(landscape: Landscape, seed: int) -> dict[str, object]:
    state_costs = np.asarray(
        [landscape.cost_by_assignment[state[:4]] for state in TOKEN_STATES], dtype=float
    )
    span = max(float(landscape.maximum_cost - landscape.exact_cost), 1.0)
    normalized_state_costs = (state_costs - landscape.exact_cost) / span
    assignment_costs = landscape.costs
    normalized_assignment_costs = (assignment_costs - landscape.exact_cost) / span
    evaluations = 0

    def distribution(theta: np.ndarray) -> np.ndarray:
        return aggregate_qaoa(
            landscape, qaoa_probabilities(landscape, normalized_state_costs, theta)
        )

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return cvar(normalized_assignment_costs, distribution(theta), CVaR_ALPHA)

    rng = np.random.default_rng(seed)
    initial_parameters: list[float] = []
    for layer in range(P):
        initial_parameters.extend([math.pi / (layer + 1), math.pi / (8 * (layer + 1))])
    theta0 = np.asarray(initial_parameters) + rng.normal(0.0, 0.02, 2 * P)
    started = time.time()
    result = minimize(
        objective,
        theta0,
        method="COBYLA",
        options={"maxiter": EVALUATION_BUDGET, "rhobeg": 0.2, "tol": 1e-3},
    )
    probabilities = distribution(np.asarray(result.x))
    optimum_mask = assignment_costs == landscape.exact_cost
    initial_cost = int(landscape.cost_by_assignment[landscape.initial])
    readout_rng = np.random.default_rng(100_000 + seed)
    samples = readout_rng.choice(len(landscape.assignments), READOUT_SHOTS, p=probabilities)
    sampled_index = min(
        (int(index) for index in samples),
        key=lambda index: (landscape.costs[index], landscape.assignments[index]),
    )
    return {
        "assignment": landscape.assignments[sampled_index],
        "cost": int(landscape.costs[sampled_index]),
        "objective_evaluations": evaluations,
        "parameters": [float(value) for value in result.x],
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_runtime_seconds": time.time() - started,
        "optimal_probability": float(np.sum(probabilities[optimum_mask])),
        "probability_beating_initial": float(np.sum(probabilities[assignment_costs < initial_cost])),
        "expected_hpwl_dbu": float(np.dot(probabilities, assignment_costs)),
        "cvar_0.25_hpwl_dbu": float(cvar(assignment_costs, probabilities, CVaR_ALPHA)),
        "expected_best_64_hpwl_dbu": expected_best(assignment_costs, probabilities, 64),
        "expected_best_256_hpwl_dbu": expected_best(assignment_costs, probabilities, 256),
        "expected_best_4096_hpwl_dbu": expected_best(assignment_costs, probabilities, READOUT_SHOTS),
        "readout_shots": READOUT_SHOTS,
    }


def result_row(
    window: Mapping[str, object],
    landscape: Landscape,
    method: str,
    seed: int | None,
    assignment: tuple[int, ...],
    cost: int,
    evaluations: int,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    initial_cost = int(landscape.cost_by_assignment[landscape.initial])
    available = initial_cost - landscape.exact_cost
    improvement = initial_cost - cost
    return {
        "window_id": window["window_id"],
        "design": str(window["window_id"]).split("_w", 1)[0],
        "selection_stratum": window["selection_stratum"],
        "method": method,
        "seed": "" if seed is None else seed,
        "objective_evaluation_budget": 0 if method == "unchanged" else (360 if method == "exact" else EVALUATION_BUDGET),
        "objective_evaluations": evaluations,
        "solution_assignment": json.dumps(list(assignment), separators=(",", ":")),
        "initial_hpwl_dbu": initial_cost,
        "exact_hpwl_dbu": landscape.exact_cost,
        "best_hpwl_dbu": cost,
        "improvement_dbu": improvement,
        "improvement_percent": 100.0 * improvement / initial_cost if initial_cost else 0.0,
        "available_improvement_dbu": available,
        "available_gap_closed": improvement / available if available > 0 else (1.0 if cost == landscape.exact_cost else 0.0),
        "approximation_ratio": cost / landscape.exact_cost if landscape.exact_cost else math.inf,
        "optimal_hit": cost == landscape.exact_cost,
        **(dict(extra) if extra else {}),
    }


def evaluate_window(window: Mapping[str, object]) -> list[dict[str, object]]:
    landscape = load_landscape(window)
    rows = [
        result_row(
            window, landscape, "unchanged", None, landscape.initial,
            int(landscape.cost_by_assignment[landscape.initial]), 0,
        )
    ]
    exact_assignment = min(
        (assignment for assignment in landscape.assignments if landscape.cost_by_assignment[assignment] == landscape.exact_cost)
    )
    rows.append(result_row(window, landscape, "exact", None, exact_assignment, landscape.exact_cost, 360))
    assignment, cost, evaluations = greedy_search(landscape, EVALUATION_BUDGET)
    rows.append(result_row(window, landscape, "greedy", None, assignment, cost, evaluations))
    for seed in SEEDS:
        assignment, cost, evaluations = random_search(landscape, EVALUATION_BUDGET, seed)
        rows.append(result_row(window, landscape, "random_search", seed, assignment, cost, evaluations))
        assignment, cost, evaluations = simulated_annealing(landscape, EVALUATION_BUDGET, seed)
        rows.append(result_row(window, landscape, "simulated_annealing", seed, assignment, cost, evaluations))
        qaoa = run_qaoa(landscape, seed)
        rows.append(
            result_row(
                window,
                landscape,
                "token_qaoa",
                seed,
                qaoa.pop("assignment"),
                int(qaoa.pop("cost")),
                int(qaoa["objective_evaluations"]),
                {
                    **qaoa,
                    "qaoa_p": P,
                    "qaoa_graph": "line",
                    "qaoa_schedule": "rotating",
                    "optimization_objective": "cvar_0.25",
                    "simulation_mode": "ideal reduced token-permutation subspace",
                    "phase_oracle_cost_table_entries": 360,
                },
            )
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(str(field))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_metadata(root: Path) -> dict[str, object]:
    def command(*args: str) -> str:
        return subprocess.run(args, cwd=root, check=True, text=True, capture_output=True).stdout.strip()

    return {
        "commit": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "scoped_status": command("git", "status", "--short", "--", ".", ":(exclude)qrl/**").splitlines(),
        "qrl_excluded_from_status_capture": True,
    }


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["method"]), []).append(row)
    method_summary: dict[str, object] = {}
    for method, group in sorted(groups.items()):
        improving = [row for row in group if float(row["available_improvement_dbu"]) > 0]
        method_row: dict[str, object] = {
            "rows": len(group),
            "optimal_hit_rate_all_windows": sum(bool(row["optimal_hit"]) for row in group) / len(group),
            "mean_approximation_ratio": float(np.mean([float(row["approximation_ratio"]) for row in group])),
            "mean_improvement_dbu_all_windows": float(np.mean([float(row["improvement_dbu"]) for row in group])),
            "improving_window_rows": len(improving),
            "optimal_hit_rate_improving_windows": (
                sum(bool(row["optimal_hit"]) for row in improving) / len(improving) if improving else None
            ),
            "mean_available_gap_closed_improving_windows": (
                float(np.mean([float(row["available_gap_closed"]) for row in improving])) if improving else None
            ),
        }
        if method == "token_qaoa":
            method_row.update(
                {
                    "mean_optimal_probability": float(np.mean([float(row["optimal_probability"]) for row in group])),
                    "mean_probability_beating_initial": float(
                        np.mean([float(row["probability_beating_initial"]) for row in group])
                    ),
                    "mean_expected_best_256_hpwl_dbu": float(
                        np.mean([float(row["expected_best_256_hpwl_dbu"]) for row in group])
                    ),
                }
            )
        method_summary[method] = method_row
    first_by_window: dict[str, Mapping[str, object]] = {}
    for row in rows:
        first_by_window.setdefault(str(row["window_id"]), row)
    return {
        "row_count": len(rows),
        "window_count": len({str(row["window_id"]) for row in rows}),
        "initially_exact_window_count": sum(
            int(row["initial_hpwl_dbu"]) == int(row["exact_hpwl_dbu"]) for row in first_by_window.values()
        ),
        "methods": method_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard index must satisfy 0 <= index < count")
    payload = json.loads(args.manifest.read_text())
    if payload.get("manifest_hash") != canonical_manifest_hash(payload):
        raise SystemExit("benchmark manifest hash mismatch")
    windows = list(payload["windows"])[args.shard_index :: args.shard_count]
    if args.limit is not None:
        windows = windows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    run_manifest = {
        "benchmark_manifest": str(args.manifest),
        "benchmark_manifest_sha256": sha256(args.manifest),
        "benchmark_manifest_hash": payload["manifest_hash"],
        "window_ids": [window["window_id"] for window in windows],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "limit": args.limit,
        "protocol": {
            "methods": METHODS,
            "qaoa": {"architecture": "explicit-EMPTY token permutation", "graph": "line", "schedule": "rotating", "p": P, "objective": "CVaR(0.25)"},
            "evaluation_budget": EVALUATION_BUDGET,
            "readout_shots": READOUT_SHOTS,
            "seeds": SEEDS,
            "budget_caveat": "QAOA distribution evaluations and classical candidate evaluations are matched numerically but are not computationally equivalent oracle calls.",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        },
        "git": git_metadata(root),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    rows: list[dict[str, object]] = []
    started = time.time()
    try:
        for window in windows:
            rows.extend(evaluate_window(window))
            write_csv(args.output_dir / "run_level.csv", rows)
    except Exception as error:
        (args.output_dir / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n"
        )
        raise
    summary = summarize(rows)
    summary["runtime_seconds"] = time.time() - started
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "windows": len(windows), "runtime_seconds": summary["runtime_seconds"]}))


if __name__ == "__main__":
    main()
