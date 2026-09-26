"""Summarize the once-selected QAOA protocol against the frozen baseline row."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .develop_qaoa_protocol import ProtocolConfig, qaoa_probabilities
from .placement_windows import canonical_manifest_hash, sha256
from .run_real_benchmark import TOKEN_STATES, aggregate_qaoa, expected_best, load_landscape


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_by_window(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["window_id"])].append(float(row[field]))
    return {window_id: float(np.mean(values)) for window_id, values in grouped.items()}


def bootstrap_mean_ci(values: Sequence[float], seed: int = 20260827) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def reconstruct_selected_rows(
    manifest: Mapping[str, object], rows: Sequence[Mapping[str, str]]
) -> list[dict[str, object]]:
    windows = {str(window["window_id"]): window for window in manifest["windows"]}
    output: list[dict[str, object]] = []
    for row in rows:
        window = windows[row["window_id"]]
        landscape = load_landscape(window)
        config = ProtocolConfig(
            str(row["config"]),
            str(row["graph"]),
            str(row["schedule"]),
            int(row["p"]),
            str(row["objective"]),
            int(row["budget"]),
        )
        state_costs = np.asarray(
            [landscape.cost_by_assignment[state[:4]] for state in TOKEN_STATES], dtype=float
        )
        initial_cost = float(landscape.cost_by_assignment[landscape.initial])
        span = max(float(np.max(np.abs(state_costs - initial_cost))), 1.0)
        theta = np.asarray(json.loads(row["parameters"]), dtype=float)
        probabilities = aggregate_qaoa(
            landscape,
            qaoa_probabilities(
                landscape,
                (state_costs - initial_cost) / span,
                config,
                theta,
            ),
        )
        reproduced_256 = expected_best(landscape.costs, probabilities, 256)
        if not np.isclose(reproduced_256, float(row["expected_best_256_hpwl_dbu"]), atol=1e-7):
            raise ValueError(f"failed to reproduce {row['window_id']}/{row['seed']}")
        seed = int(row["seed"])
        rng = np.random.default_rng(100_000 + seed)
        samples = rng.choice(len(landscape.assignments), 4096, p=probabilities)
        sampled_index = min(
            (int(index) for index in samples),
            key=lambda index: (landscape.costs[index], landscape.assignments[index]),
        )
        sampled_best = float(landscape.costs[sampled_index])
        opportunity = initial_cost - landscape.exact_cost
        output.append(
            {
                **row,
                "method": "selected_token_qaoa",
                "solution_assignment": json.dumps(landscape.assignments[sampled_index]),
                "best_hpwl_dbu": sampled_best,
                "expected_best_64_hpwl_dbu": expected_best(landscape.costs, probabilities, 64),
                "expected_best_4096_hpwl_dbu": expected_best(
                    landscape.costs, probabilities, 4096
                ),
                "sampled_best_4096_hpwl_dbu": sampled_best,
                "expected_best_64_gap_closure": (
                    (initial_cost - expected_best(landscape.costs, probabilities, 64))
                    / opportunity
                    if opportunity > 0
                    else ""
                ),
                "expected_best_4096_gap_closure": (
                    (initial_cost - expected_best(landscape.costs, probabilities, 4096))
                    / opportunity
                    if opportunity > 0
                    else ""
                ),
                "sampled_best_4096_gap_closure": (
                    (initial_cost - sampled_best) / opportunity if opportunity > 0 else ""
                ),
                "sampled_exact_hit_4096": sampled_best == landscape.exact_cost,
            }
        )
    return output


def summarize(
    manifest: Mapping[str, object],
    selected_rows: Sequence[Mapping[str, object]],
    frozen_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    improving_ids = {
        str(window["window_id"])
        for window in manifest["windows"]
        if int(window["initial_hpwl_dbu"]) > int(window["exact_hpwl_dbu"])
    }
    selected_improving = [row for row in selected_rows if row["window_id"] in improving_ids]
    frozen_qaoa = [
        row for row in frozen_rows if row["method"] == "token_qaoa" and row["window_id"] in improving_ids
    ]
    frozen_expected_256_rows = []
    for row in frozen_qaoa:
        opportunity = float(row["available_improvement_dbu"])
        frozen_expected_256_rows.append(
            {
                "window_id": row["window_id"],
                "gap": (
                    float(row["initial_hpwl_dbu"])
                    - float(row["expected_best_256_hpwl_dbu"])
                )
                / opportunity,
            }
        )
    old_by_window = mean_by_window(frozen_expected_256_rows, "gap")
    new_by_window = mean_by_window(selected_improving, "expected_best_256_gap_closure")
    paired = [new_by_window[key] - old_by_window[key] for key in sorted(improving_ids)]
    paired_ci = bootstrap_mean_ci(paired)

    sampled_by_window = mean_by_window(selected_improving, "sampled_best_4096_gap_closure")
    expected_64_by_window = mean_by_window(selected_improving, "expected_best_64_gap_closure")
    expected_4096_by_window = mean_by_window(selected_improving, "expected_best_4096_gap_closure")
    exact_hit_by_window = mean_by_window(selected_rows, "sampled_exact_hit_4096")
    improving_exact_hit_by_window = mean_by_window(
        selected_improving, "sampled_exact_hit_4096"
    )
    frozen_sampled = mean_by_window(frozen_qaoa, "available_gap_closed")

    classical: dict[str, float] = {}
    classical_by_window: dict[str, dict[str, float]] = {}
    for method in ("greedy", "simulated_annealing", "random_search"):
        method_rows = [
            row for row in frozen_rows if row["method"] == method and row["window_id"] in improving_ids
        ]
        method_by_window = mean_by_window(method_rows, "available_gap_closed")
        classical_by_window[method] = method_by_window
        classical[method] = float(np.mean(list(method_by_window.values())))

    selected_minus_greedy = [
        sampled_by_window[key] - classical_by_window["greedy"][key]
        for key in sorted(improving_ids)
    ]
    selected_minus_greedy_ci = bootstrap_mean_ci(selected_minus_greedy)

    return {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_manifest_hash": manifest["manifest_hash"],
        "window_count": len(manifest["windows"]),
        "improving_window_count": len(improving_ids),
        "selected_protocol": {
            "graph": selected_rows[0]["graph"],
            "schedule": selected_rows[0]["schedule"],
            "p": int(selected_rows[0]["p"]),
            "objective": selected_rows[0]["objective"],
            "optimizer_budget": int(selected_rows[0]["budget"]),
            "edge_layers": int(selected_rows[0]["edge_layers"]),
            "seeds": sorted({int(row["seed"]) for row in selected_rows}),
        },
        "analytical_expected_best_gap_closure": {
            "selected_64": float(np.mean(list(expected_64_by_window.values()))),
            "selected_256": float(np.mean(list(new_by_window.values()))),
            "selected_4096": float(np.mean(list(expected_4096_by_window.values()))),
            "frozen_line_256": float(np.mean(list(old_by_window.values()))),
            "selected_minus_frozen_line_256": float(np.mean(paired)),
            "selected_minus_frozen_line_256_bootstrap_95_ci": list(paired_ci),
            "paired_window_wins_ties_losses": [
                sum(value > 1e-12 for value in paired),
                sum(abs(value) <= 1e-12 for value in paired),
                sum(value < -1e-12 for value in paired),
            ],
        },
        "seeded_best_of_4096_gap_closure": {
            "selected_ring": float(np.mean(list(sampled_by_window.values()))),
            "frozen_line": float(np.mean(list(frozen_sampled.values()))),
            **classical,
            "selected_minus_greedy": float(np.mean(selected_minus_greedy)),
            "selected_minus_greedy_bootstrap_95_ci": list(selected_minus_greedy_ci),
        },
        "selected_seeded_best_of_4096_exact_hit_rate_all_windows": float(
            np.mean(list(exact_hit_by_window.values()))
        ),
        "selected_seeded_best_of_4096_exact_hit_rate_improving_windows": float(
            np.mean(list(improving_exact_hit_by_window.values()))
        ),
        "selected_mean_probability_beating_initial_all_windows": float(
            np.mean([float(row["probability_beating_initial"]) for row in selected_rows])
        ),
        "selected_mean_optimal_probability_all_windows": float(
            np.mean([float(row["optimal_probability"]) for row in selected_rows])
        ),
        "optimizer_max_budget_status_count": sum(
            "MAXFUN" in str(row["optimizer_message"]).upper()
            or "MAXIMUM NUMBER" in str(row["optimizer_message"]).upper()
            for row in selected_rows
        ),
    }


def report(summary: Mapping[str, object]) -> str:
    expected = summary["analytical_expected_best_gap_closure"]
    sampled = summary["seeded_best_of_4096_gap_closure"]
    protocol = summary["selected_protocol"]
    wins, ties, losses = expected["paired_window_wins_ties_losses"]
    ci_low, ci_high = expected["selected_minus_frozen_line_256_bootstrap_95_ci"]
    greedy_ci_low, greedy_ci_high = sampled["selected_minus_greedy_bootstrap_95_ci"]
    validation = summary.get("openroad_validation")
    validation_paragraph = ""
    if validation:
        validation_paragraph = (
            f"\nAll {validation['total_validation_count']} seeded best-of-4,096 assignments "
            "passed OpenROAD `check_placement` when reinserted into their complete legal "
            "source placements (12/12 each for GCD, AES, and Ibex).\n"
        )
    return f"""# Selected QAOA Protocol: Sealed-Holdout Result

The development-selected protocol materially improves the frozen token-QAOA
row on the untouched 36-window GCD/AES/Ibex benchmark. The selected method is
ring/reversed token mixing, `p={protocol['p']}`, probability of beating the
initial placement as the outer objective, and {protocol['optimizer_budget']}
COBYLA distribution evaluations. It uses {protocol['edge_layers']} mixer
edge-layers versus 15 for the frozen line/rotating method.

## Primary result

Over the {summary['improving_window_count']} windows with exact improvement
opportunity, analytical expected-best-of-256 gap closure is
{100 * expected['selected_256']:.1f}% versus
{100 * expected['frozen_line_256']:.1f}% for the frozen QAOA row. The paired
gain is {100 * expected['selected_minus_frozen_line_256']:.1f} percentage
points (window-bootstrap 95% CI
[{100 * ci_low:.1f}, {100 * ci_high:.1f}]); the selected method wins/ties/loses
{wins}/{ties}/{losses} windows.

Expected gap closure is {100 * expected['selected_64']:.1f}% at 64 readouts and
{100 * expected['selected_4096']:.1f}% at 4,096 readouts. With the same seeded
4,096-readout rule used by the frozen comparison, selected QAOA closes
{100 * sampled['selected_ring']:.1f}% of the gap, compared with
{100 * sampled['frozen_line']:.1f}% for frozen QAOA,
{100 * sampled['simulated_annealing']:.1f}% for simulated annealing, and
{100 * sampled['greedy']:.1f}% for greedy search. These are quality numbers,
not query- or runtime-equivalent comparisons. The selected-minus-greedy
quality difference is {100 * sampled['selected_minus_greedy']:.1f} percentage
points with a paired window-bootstrap 95% CI
[{100 * greedy_ci_low:.1f}, {100 * greedy_ci_high:.1f}].

The negative 64-readout expectation is an important resource warning: this
protocol relies on enough readouts to expose its useful tail and is not a
quality improvement in the low-readout regime.

The seeded best-of-4,096 exact-hit rate is
{100 * summary['selected_seeded_best_of_4096_exact_hit_rate_all_windows']:.1f}%
over all windows. Mean single-readout probability of beating the initial
placement is
{100 * summary['selected_mean_probability_beating_initial_all_windows']:.2f}%.
The exact-hit rate restricted to improving windows is
{100 * summary['selected_seeded_best_of_4096_exact_hit_rate_improving_windows']:.1f}%.
{validation_paragraph}

## Interpretation guardrails

The holdout was evaluated once after protocol selection on a separate
JPEG/dynamic_node train-validation corpus. The selected objective needs only
sampled HPWL and the known starting HPWL; it does not use the exact optimum.
Exact values enter this report only retrospectively. This remains ideal
reduced-permutation simulation with an exact cost lookup phase. It does not
establish hardware practicality, quantum advantage, or equality between a
QAOA distribution evaluation and a classical candidate-cost evaluation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-run-level", type=Path, required=True)
    parser.add_argument("--frozen-run-level", type=Path, required=True)
    parser.add_argument("--openroad-validation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("manifest_hash") != canonical_manifest_hash(manifest):
        raise ValueError("benchmark manifest hash is invalid")
    selected_rows = reconstruct_selected_rows(manifest, read_csv(args.selected_run_level))
    frozen_rows = read_csv(args.frozen_run_level)
    summary = summarize(manifest, selected_rows, frozen_rows)
    if args.openroad_validation:
        validation = json.loads(args.openroad_validation.read_text())
        if validation.get("status") != "pass":
            raise ValueError("OpenROAD validation did not pass")
        summary["openroad_validation"] = validation
    summary["provenance"] = {
        "manifest_sha256": sha256(args.manifest),
        "selected_run_level_sha256": sha256(args.selected_run_level),
        "frozen_run_level_sha256": sha256(args.frozen_run_level),
    }
    if args.openroad_validation:
        summary["provenance"]["openroad_validation_sha256"] = sha256(
            args.openroad_validation
        )
    (args.output_dir / "selected_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS_REPORT.md").write_text(report(summary))
    fieldnames = list(selected_rows[0])
    with (args.output_dir / "reconstructed_readout.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
