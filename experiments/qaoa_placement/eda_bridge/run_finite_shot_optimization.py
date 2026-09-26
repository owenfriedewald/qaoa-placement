"""Evaluate finite-shot outer-objective optimization for the selected QAOA."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .develop_qaoa_protocol import (
    ProtocolConfig,
    initial_parameters,
    qaoa_probabilities,
)
from .placement_windows import canonical_manifest_hash, sha256
from .run_real_benchmark import (
    TOKEN_STATES,
    aggregate_qaoa,
    cvar,
    expected_best,
    load_landscape,
)


CONFIG = ProtocolConfig("selected_ring_p3_beat", "ring", "reversed", 3, "beat_initial", 40)
SEEDS = (11, 17, 23, 29)
READOUT_SHOTS = (64, 256, 4096)


def measurement_seed(seed: int, objective_shots: int) -> int:
    return 10_000_019 + 10_007 * seed + 101 * objective_shots


def run_case(
    window: Mapping[str, object], objective_shots: int, seed: int
) -> dict[str, object]:
    landscape = load_landscape(window)
    state_costs = np.asarray(
        [landscape.cost_by_assignment[state[:4]] for state in TOKEN_STATES], dtype=float
    )
    initial_cost = float(landscape.cost_by_assignment[landscape.initial])
    span = max(float(np.max(np.abs(state_costs - initial_cost))), 1.0)
    normalized_state_costs = (state_costs - initial_cost) / span
    rng = np.random.default_rng(measurement_seed(seed, objective_shots))
    evaluations = 0
    estimator_squared_errors: list[float] = []

    def distribution(theta: np.ndarray) -> np.ndarray:
        return aggregate_qaoa(
            landscape,
            qaoa_probabilities(landscape, normalized_state_costs, CONFIG, theta),
        )

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        probabilities = distribution(theta)
        true_probability = float(np.sum(probabilities[landscape.costs < initial_cost]))
        estimate = float(rng.binomial(objective_shots, true_probability)) / objective_shots
        estimator_squared_errors.append((estimate - true_probability) ** 2)
        return -estimate

    started = time.time()
    result = minimize(
        objective,
        initial_parameters(CONFIG, seed),
        method="COBYLA",
        options={"maxiter": CONFIG.budget, "rhobeg": 0.2, "tol": 1e-3},
    )
    probabilities = distribution(np.asarray(result.x))
    exact = float(landscape.exact_cost)
    opportunity = initial_cost - exact
    metrics: dict[str, object] = {}
    for shots in READOUT_SHOTS:
        best = expected_best(landscape.costs, probabilities, shots)
        metrics[f"expected_best_{shots}_hpwl_dbu"] = best
        metrics[f"expected_best_{shots}_gap_closure"] = (
            (initial_cost - best) / opportunity if opportunity > 0 else ""
        )
    readout_rng = np.random.default_rng(500_000 + seed)
    samples = readout_rng.choice(len(landscape.assignments), 4096, p=probabilities)
    sampled_index = min(
        (int(index) for index in samples),
        key=lambda index: (landscape.costs[index], landscape.assignments[index]),
    )
    sampled_best = float(landscape.costs[sampled_index])
    return {
        "window_id": window["window_id"],
        "design": str(window["window_id"]).split("_w")[0],
        "selection_stratum": window["selection_stratum"],
        "split": window.get("development_split", "finite_shot_holdout"),
        "objective_shots": objective_shots,
        "seed": seed,
        "graph": CONFIG.graph,
        "schedule": CONFIG.schedule,
        "p": CONFIG.p,
        "edge_layers": CONFIG.edge_layers,
        "optimizer": "COBYLA",
        "objective": CONFIG.objective,
        "objective_evaluation_budget": CONFIG.budget,
        "objective_evaluations": evaluations,
        "total_objective_shots": evaluations * objective_shots,
        "initial_hpwl_dbu": initial_cost,
        "exact_hpwl_dbu": exact,
        "exact_opportunity_dbu": opportunity,
        "expected_hpwl_dbu": float(np.dot(probabilities, landscape.costs)),
        "cvar_0.25_hpwl_dbu": cvar(landscape.costs, probabilities, 0.25),
        "probability_beating_initial": float(
            np.sum(probabilities[landscape.costs < initial_cost])
        ),
        "optimal_probability": float(np.sum(probabilities[landscape.costs == exact])),
        **metrics,
        "sampled_best_4096_hpwl_dbu": sampled_best,
        "sampled_best_4096_gap_closure": (
            (initial_cost - sampled_best) / opportunity if opportunity > 0 else ""
        ),
        "sampled_best_4096_assignment": json.dumps(landscape.assignments[sampled_index]),
        "sampled_exact_hit_4096": sampled_best == exact,
        "objective_estimator_rmse": math.sqrt(float(np.mean(estimator_squared_errors))),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_runtime_seconds": time.time() - started,
        "parameters": json.dumps([float(value) for value in result.x]),
    }


def _mean_by_window(
    rows: Sequence[Mapping[str, object]], field: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row[field] == "":
            continue
        grouped.setdefault(str(row["window_id"]), []).append(float(row[field]))
    return {window_id: float(np.mean(values)) for window_id, values in grouped.items()}


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for shots in sorted({int(row["objective_shots"]) for row in rows}):
        group = [row for row in rows if int(row["objective_shots"]) == shots]
        improving = [row for row in group if float(row["exact_opportunity_dbu"]) > 0]
        gap_64 = _mean_by_window(improving, "expected_best_64_gap_closure")
        gap_256 = _mean_by_window(improving, "expected_best_256_gap_closure")
        gap_4096 = _mean_by_window(improving, "expected_best_4096_gap_closure")
        sampled = _mean_by_window(improving, "sampled_best_4096_gap_closure")
        output.append(
            {
                "objective_shots": shots,
                "window_count": len({str(row["window_id"]) for row in group}),
                "improving_window_count": len(gap_256),
                "seed_count": len({int(row["seed"]) for row in group}),
                "mean_expected_best_64_gap_closure": float(np.mean(list(gap_64.values()))),
                "mean_expected_best_256_gap_closure": float(np.mean(list(gap_256.values()))),
                "mean_expected_best_4096_gap_closure": float(
                    np.mean(list(gap_4096.values()))
                ),
                "mean_sampled_best_4096_gap_closure": float(
                    np.mean(list(sampled.values()))
                ),
                "mean_probability_beating_initial_all_windows": float(
                    np.mean([float(row["probability_beating_initial"]) for row in group])
                ),
                "mean_objective_estimator_rmse": float(
                    np.mean([float(row["objective_estimator_rmse"]) for row in group])
                ),
                "mean_total_objective_shots": float(
                    np.mean([float(row["total_objective_shots"]) for row in group])
                ),
                "optimizer_budget_status_count": sum(
                    not bool(row["optimizer_success"]) for row in group
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -float(row["mean_expected_best_256_gap_closure"]),
            int(row["objective_shots"]),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "finite_shot_holdout"), required=True
    )
    parser.add_argument("--objective-shots", action="append", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(shots < 1 for shots in args.objective_shots):
        raise ValueError("objective shots must be positive")
    if len(set(args.objective_shots)) != len(args.objective_shots):
        raise ValueError("objective shot counts must be unique")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("manifest_hash") != canonical_manifest_hash(manifest):
        raise ValueError("input manifest hash is invalid")
    if args.split == "finite_shot_holdout":
        if any("development_split" in window for window in manifest["windows"]):
            raise ValueError("holdout mode requires an unlabeled benchmark")
        windows = list(manifest["windows"])
    else:
        windows = [
            window for window in manifest["windows"] if window.get("development_split") == args.split
        ]
    if not windows:
        raise ValueError(f"no windows for split {args.split}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        run_case(window, shots, seed)
        for shots in args.objective_shots
        for window in windows
        for seed in SEEDS
    ]
    aggregates = aggregate_rows(rows)
    write_csv(args.output_dir / "run_level.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregates)
    run_manifest = {
        "input_manifest": str(args.manifest),
        "input_manifest_sha256": sha256(args.manifest),
        "input_manifest_hash": manifest["manifest_hash"],
        "split": args.split,
        "objective_shots": args.objective_shots,
        "seeds": list(SEEDS),
        "protocol": {
            "graph": CONFIG.graph,
            "schedule": CONFIG.schedule,
            "p": CONFIG.p,
            "edge_layers": CONFIG.edge_layers,
            "objective": CONFIG.objective,
            "optimizer": "COBYLA",
            "optimizer_budget": CONFIG.budget,
            "finite_shot_objective": True,
            "gate_noise": False,
            "readout_noise": False,
            "exact_information_used_by_optimizer": False,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "ranking": aggregates}, indent=2))


if __name__ == "__main__":
    main()
