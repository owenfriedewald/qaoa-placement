"""Run declared token-QAOA configurations on a frozen development split.

Configurations use the compact form
``name,graph,schedule,p,objective,budget``.  Exact optima are used only in
retrospective output metrics, never by an optimizer objective.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .placement_windows import canonical_manifest_hash, sha256
from .run_real_benchmark import (
    Landscape,
    TOKEN_INDEX,
    TOKEN_STATES,
    aggregate_qaoa,
    cvar,
    expected_best,
    load_landscape,
)


SEEDS = (11, 17)
READOUT_SHOTS = 256
GRAPHS = ("line", "ring", "real_empty_priority")
SCHEDULES = (
    "current",
    "reversed",
    "alternating_direction",
    "edge_colored",
    "empty_prioritized",
    "rotating",
)
OBJECTIVES = ("cvar_0.10", "cvar_0.25", "cvar_0.50", "expected", "beat_initial", "cvar_beat")


@dataclass(frozen=True)
class ProtocolConfig:
    name: str
    graph: str
    schedule: str
    p: int
    objective: str
    budget: int

    @property
    def token_edges(self) -> tuple[tuple[int, int], ...]:
        if self.graph == "line":
            return tuple((index, index + 1) for index in range(5))
        if self.graph == "ring":
            return tuple((index, index + 1) for index in range(5)) + ((5, 0),)
        if self.graph == "real_empty_priority":
            edges = {(real, empty) for empty in (4, 5) for real in range(4)}
            edges.update((index, index + 1) for index in range(3))
            edges.add((4, 5))
            return tuple(sorted(edges))
        raise ValueError(f"unsupported graph: {self.graph}")

    @property
    def edge_layers(self) -> int:
        return self.p * len(self.token_edges)


def parse_config(text: str) -> ProtocolConfig:
    fields = text.split(",")
    if len(fields) != 6:
        raise ValueError("config must be name,graph,schedule,p,objective,budget")
    config = ProtocolConfig(fields[0], fields[1], fields[2], int(fields[3]), fields[4], int(fields[5]))
    if config.graph not in GRAPHS:
        raise ValueError(f"unknown graph {config.graph}")
    if config.schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {config.schedule}")
    if config.objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {config.objective}")
    if config.p < 1 or config.budget < 1:
        raise ValueError("p and budget must be positive")
    return config


def edge_color_order(edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    remaining = list(edges)
    ordered: list[tuple[int, int]] = []
    while remaining:
        used: set[int] = set()
        deferred: list[tuple[int, int]] = []
        for edge in remaining:
            if edge[0] in used or edge[1] in used:
                deferred.append(edge)
            else:
                ordered.append(edge)
                used.update(edge)
        remaining = deferred
    return tuple(ordered)


def scheduled_edges(config: ProtocolConfig, layer: int) -> tuple[tuple[int, int], ...]:
    base = config.token_edges
    if config.schedule == "current":
        return base
    if config.schedule == "reversed":
        return tuple(reversed(base))
    if config.schedule == "alternating_direction":
        return base if layer % 2 == 0 else tuple(reversed(base))
    if config.schedule == "edge_colored":
        return edge_color_order(base)
    if config.schedule == "empty_prioritized":
        return tuple(sorted(base, key=lambda edge: (not (edge[0] >= 4 or edge[1] >= 4), edge)))
    if config.schedule == "rotating":
        offset = layer % len(base)
        return base[offset:] + base[:offset]
    raise ValueError(config.schedule)


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


PAIR_ROWS = {
    edge: _swap_pairs(edge)
    for graph in GRAPHS
    for edge in ProtocolConfig("cache", graph, "current", 1, "expected", 1).token_edges
}


def qaoa_probabilities(
    landscape: Landscape,
    normalized_state_costs: np.ndarray,
    config: ProtocolConfig,
    theta: Sequence[float],
) -> np.ndarray:
    state = np.zeros(len(TOKEN_STATES), dtype=np.complex128)
    empty_sites = tuple(site for site in range(6) if site not in landscape.initial)
    state[TOKEN_INDEX[landscape.initial + empty_sites]] = 1.0
    for layer in range(config.p):
        gamma = float(theta[2 * layer])
        beta = float(theta[2 * layer + 1])
        state *= np.exp(-1j * gamma * normalized_state_costs)
        cosine = math.cos(beta)
        sine = -1j * math.sin(beta)
        for edge in scheduled_edges(config, layer):
            left, right = PAIR_ROWS[edge]
            left_amp = state[left].copy()
            right_amp = state[right].copy()
            state[left] = cosine * left_amp + sine * right_amp
            state[right] = sine * left_amp + cosine * right_amp
    probabilities = np.abs(state) ** 2
    return probabilities / float(np.sum(probabilities))


def objective_value(
    objective: str,
    costs: np.ndarray,
    probabilities: np.ndarray,
    initial_cost: float,
) -> float:
    span = max(float(np.max(np.abs(costs - initial_cost))), 1.0)
    normalized = (costs - initial_cost) / span
    if objective.startswith("cvar_0"):
        return cvar(normalized, probabilities, float(objective.split("_")[1]))
    if objective == "expected":
        return float(np.dot(probabilities, normalized))
    beating = float(np.sum(probabilities[costs < initial_cost]))
    if objective == "beat_initial":
        return -beating
    if objective == "cvar_beat":
        return 0.5 * cvar(normalized, probabilities, 0.25) + 0.5 * (1.0 - beating)
    raise ValueError(objective)


def initial_parameters(config: ProtocolConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for layer in range(config.p):
        values.extend([math.pi / (layer + 1), math.pi / (8 * (layer + 1))])
    return np.asarray(values) + rng.normal(0.0, 0.02, 2 * config.p)


def run_case(
    window: Mapping[str, object], config: ProtocolConfig, seed: int
) -> dict[str, object]:
    landscape = load_landscape(window)
    state_costs = np.asarray(
        [landscape.cost_by_assignment[state[:4]] for state in TOKEN_STATES], dtype=float
    )
    initial_cost = float(landscape.cost_by_assignment[landscape.initial])
    span = max(float(np.max(np.abs(state_costs - initial_cost))), 1.0)
    normalized_state_costs = (state_costs - initial_cost) / span
    evaluations = 0

    def distribution(theta: np.ndarray) -> np.ndarray:
        return aggregate_qaoa(
            landscape,
            qaoa_probabilities(landscape, normalized_state_costs, config, theta),
        )

    def objective(theta: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return objective_value(
            config.objective,
            landscape.costs,
            distribution(theta),
            initial_cost,
        )

    started = time.time()
    result = minimize(
        objective,
        initial_parameters(config, seed),
        method="COBYLA",
        options={"maxiter": config.budget, "rhobeg": 0.2, "tol": 1e-3},
    )
    probabilities = distribution(np.asarray(result.x))
    best_256 = expected_best(landscape.costs, probabilities, READOUT_SHOTS)
    exact = float(landscape.exact_cost)
    opportunity = initial_cost - exact
    return {
        "window_id": window["window_id"],
        "design": str(window["window_id"]).split("_w")[0],
        "selection_stratum": window["selection_stratum"],
        "development_split": window.get("development_split", "sealed_holdout"),
        "config": config.name,
        "graph": config.graph,
        "schedule": config.schedule,
        "p": config.p,
        "objective": config.objective,
        "budget": config.budget,
        "token_edges_per_layer": len(config.token_edges),
        "edge_layers": config.edge_layers,
        "seed": seed,
        "initial_hpwl_dbu": initial_cost,
        "exact_hpwl_dbu": exact,
        "exact_opportunity_dbu": opportunity,
        "expected_hpwl_dbu": float(np.dot(probabilities, landscape.costs)),
        "cvar_0.25_hpwl_dbu": cvar(landscape.costs, probabilities, 0.25),
        "probability_beating_initial": float(
            np.sum(probabilities[landscape.costs < initial_cost])
        ),
        "optimal_probability": float(np.sum(probabilities[landscape.costs == exact])),
        "expected_best_256_hpwl_dbu": best_256,
        "expected_best_256_gap_closure": (
            (initial_cost - best_256) / opportunity if opportunity > 0 else ""
        ),
        "objective_evaluations": evaluations,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_runtime_seconds": time.time() - started,
        "parameters": json.dumps([float(value) for value in result.x]),
    }


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for config_name in sorted({str(row["config"]) for row in rows}):
        group = [row for row in rows if row["config"] == config_name]
        by_window: dict[str, list[Mapping[str, object]]] = {}
        for row in group:
            by_window.setdefault(str(row["window_id"]), []).append(row)
        window_rows = []
        for window_id, seed_rows in by_window.items():
            window_rows.append(
                {
                    "window_id": window_id,
                    "opportunity": float(seed_rows[0]["exact_opportunity_dbu"]),
                    "gap": np.mean(
                        [
                            float(row["expected_best_256_gap_closure"])
                            for row in seed_rows
                            if row["expected_best_256_gap_closure"] != ""
                        ]
                    )
                    if float(seed_rows[0]["exact_opportunity_dbu"]) > 0
                    else math.nan,
                    "beat": np.mean([float(row["probability_beating_initial"]) for row in seed_rows]),
                    "opt": np.mean([float(row["optimal_probability"]) for row in seed_rows]),
                }
            )
        improving = [row for row in window_rows if row["opportunity"] > 0]
        exemplar = group[0]
        output.append(
            {
                "config": config_name,
                "graph": exemplar["graph"],
                "schedule": exemplar["schedule"],
                "p": exemplar["p"],
                "objective": exemplar["objective"],
                "budget": exemplar["budget"],
                "edge_layers": exemplar["edge_layers"],
                "window_count": len(window_rows),
                "improving_window_count": len(improving),
                "zero_opportunity_fraction": 1.0 - len(improving) / len(window_rows),
                "mean_expected_best_256_gap_closure": float(
                    np.mean([row["gap"] for row in improving])
                )
                if improving
                else "",
                "mean_probability_beating_initial": float(
                    np.mean([row["beat"] for row in window_rows])
                ),
                "mean_optimal_probability": float(np.mean([row["opt"] for row in window_rows])),
                "mean_optimizer_runtime_seconds": float(
                    np.mean([float(row["optimizer_runtime_seconds"]) for row in group])
                ),
                "optimizer_failure_count": sum(not bool(row["optimizer_success"]) for row in group),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -float(row["mean_expected_best_256_gap_closure"] or -math.inf),
            int(row["edge_layers"]),
            int(row["budget"]),
            str(row["config"]),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "sealed_holdout"), required=True)
    parser.add_argument("--config", action="append", type=parse_config, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("manifest_hash") != canonical_manifest_hash(manifest):
        raise ValueError("development manifest hash is invalid")
    if args.split == "sealed_holdout":
        if any("development_split" in window for window in manifest["windows"]):
            raise ValueError("sealed_holdout mode requires a benchmark without development labels")
        windows = list(manifest["windows"])
    else:
        windows = [
            window for window in manifest["windows"] if window.get("development_split") == args.split
        ]
    if not windows:
        raise ValueError(f"no {args.split} windows")
    if len({config.name for config in args.config}) != len(args.config):
        raise ValueError("configuration names must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        run_case(window, config, seed)
        for config in args.config
        for window in windows
        for seed in SEEDS
    ]
    aggregates = aggregate_rows(rows)
    write_csv(args.output_dir / "run_level.csv", rows)
    write_csv(args.output_dir / "aggregate.csv", aggregates)
    run_manifest = {
        "development_manifest": str(args.manifest),
        "development_manifest_sha256": sha256(args.manifest),
        "development_manifest_hash": manifest["manifest_hash"],
        "split": args.split,
        "seeds": list(SEEDS),
        "readout_metric": "analytical expected best of 256",
        "configs": [asdict(config) | {"edge_layers": config.edge_layers} for config in args.config],
        "selection_rule": "descending mean expected-best-256 gap closure on improving windows",
        "exact_information_used_by_optimizer": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir), "ranking": aggregates}, indent=2))


if __name__ == "__main__":
    main()
