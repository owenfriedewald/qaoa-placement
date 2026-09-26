"""Compare the frozen finite-shot QAOA protocol with uniform feasible sampling.

The analysis is deterministic.  It uses the exact 360-assignment landscapes
and the final ideal QAOA optimal probabilities already recorded for the four
frozen optimizer seeds.  No optimizer is rerun and no sampled outcome is used
to choose an analysis convention.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .placement_windows import canonical_manifest_hash, sha256
from .run_real_benchmark import expected_best, load_landscape


CURVE_SHOTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)
BEST_COST_SHOTS = (64, 256, 4096)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hit_probability(probability: float, shots: int) -> float:
    return 1.0 - (1.0 - probability) ** shots


def grouped_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, list[Mapping[str, str]]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["window_id"])].append(row)
    for window_id, group in grouped.items():
        seeds = [int(row["seed"]) for row in group]
        if len(group) != 4 or len(set(seeds)) != 4:
            raise ValueError(f"{window_id}: expected four distinct optimizer seeds")
    return dict(grouped)


def first_crossover(
    windows: Sequence[Mapping[str, object]],
    grouped: Mapping[str, Sequence[Mapping[str, str]]],
    matched_four_seed_selection: bool,
    maximum_shots: int = 4096,
) -> int | None:
    previous = None
    for shots in range(1, maximum_shots + 1):
        differences = []
        for window in windows:
            window_id = str(window["window_id"])
            landscape = load_landscape(window)
            uniform_probability = float(np.mean(landscape.costs == landscape.exact_cost))
            qaoa_probabilities = [
                float(row["optimal_probability"]) for row in grouped[window_id]
            ]
            if matched_four_seed_selection:
                qaoa_hit = 1.0 - float(
                    np.prod([(1.0 - probability) ** shots for probability in qaoa_probabilities])
                )
                uniform_hit = hit_probability(uniform_probability, 4 * shots)
            else:
                qaoa_hit = float(
                    np.mean([hit_probability(probability, shots) for probability in qaoa_probabilities])
                )
                uniform_hit = hit_probability(uniform_probability, shots)
            differences.append(qaoa_hit - uniform_hit)
        difference = float(np.mean(differences))
        if previous is not None and previous > 0.0 and difference <= 0.0:
            return shots
        previous = difference
    return None


def analyze(
    manifest: Mapping[str, object], rows: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    windows = list(manifest["windows"])
    grouped = grouped_rows(rows)
    window_ids = {str(window["window_id"]) for window in windows}
    if set(grouped) != window_ids:
        raise ValueError("run-level and manifest window sets differ")

    curve_rows: list[dict[str, object]] = []
    for shots in CURVE_SHOTS:
        qaoa_single = []
        uniform_single = []
        qaoa_four_seed = []
        uniform_four_seed = []
        for window in windows:
            window_id = str(window["window_id"])
            landscape = load_landscape(window)
            uniform_probability = float(np.mean(landscape.costs == landscape.exact_cost))
            qaoa_probabilities = [
                float(row["optimal_probability"]) for row in grouped[window_id]
            ]
            qaoa_single.append(
                float(np.mean([hit_probability(probability, shots) for probability in qaoa_probabilities]))
            )
            uniform_single.append(hit_probability(uniform_probability, shots))
            qaoa_four_seed.append(
                1.0
                - float(
                    np.prod([(1.0 - probability) ** shots for probability in qaoa_probabilities])
                )
            )
            uniform_four_seed.append(hit_probability(uniform_probability, 4 * shots))
        curve_rows.append(
            {
                "readouts_per_seed": shots,
                "total_readouts_four_seed_selector": 4 * shots,
                "qaoa_single_seed_mean_exact_hit_probability": float(np.mean(qaoa_single)),
                "uniform_single_seed_mean_exact_hit_probability": float(np.mean(uniform_single)),
                "qaoa_four_seed_selector_mean_exact_hit_probability": float(np.mean(qaoa_four_seed)),
                "uniform_four_seed_selector_mean_exact_hit_probability": float(np.mean(uniform_four_seed)),
                "qaoa_minus_uniform_single_seed": float(np.mean(qaoa_single) - np.mean(uniform_single)),
                "qaoa_minus_uniform_four_seed_selector": float(
                    np.mean(qaoa_four_seed) - np.mean(uniform_four_seed)
                ),
            }
        )

    budget_rows: list[dict[str, object]] = []
    for shots in BEST_COST_SHOTS:
        qaoa_improvements = []
        uniform_improvements = []
        qaoa_gap_closures = []
        uniform_gap_closures = []
        for window in windows:
            window_id = str(window["window_id"])
            landscape = load_landscape(window)
            group = grouped[window_id]
            initial = float(group[0]["initial_hpwl_dbu"])
            exact = float(group[0]["exact_hpwl_dbu"])
            qaoa_best = float(
                np.mean([float(row[f"expected_best_{shots}_hpwl_dbu"]) for row in group])
            )
            uniform_best = expected_best(
                landscape.costs,
                np.ones(len(landscape.costs), dtype=float) / len(landscape.costs),
                shots,
            )
            qaoa_improvements.append(initial - qaoa_best)
            uniform_improvements.append(initial - uniform_best)
            if initial > exact:
                qaoa_gap_closures.append((initial - qaoa_best) / (initial - exact))
                uniform_gap_closures.append((initial - uniform_best) / (initial - exact))
        budget_rows.append(
            {
                "readouts": shots,
                "qaoa_mean_absolute_improvement_dbu": float(np.mean(qaoa_improvements)),
                "uniform_mean_absolute_improvement_dbu": float(np.mean(uniform_improvements)),
                "qaoa_median_absolute_improvement_dbu": float(np.median(qaoa_improvements)),
                "uniform_median_absolute_improvement_dbu": float(np.median(uniform_improvements)),
                "qaoa_better_equal_worse_than_initial": [
                    int(sum(value > 1e-9 for value in qaoa_improvements)),
                    int(sum(abs(value) <= 1e-9 for value in qaoa_improvements)),
                    int(sum(value < -1e-9 for value in qaoa_improvements)),
                ],
                "uniform_better_equal_worse_than_initial": [
                    int(sum(value > 1e-9 for value in uniform_improvements)),
                    int(sum(abs(value) <= 1e-9 for value in uniform_improvements)),
                    int(sum(value < -1e-9 for value in uniform_improvements)),
                ],
                "qaoa_mean_gap_closure_improving_windows": float(np.mean(qaoa_gap_closures)),
                "uniform_mean_gap_closure_improving_windows": float(np.mean(uniform_gap_closures)),
            }
        )

    summary = {
        "benchmark_id": manifest["benchmark_id"],
        "window_count": len(windows),
        "optimizer_seeds_per_window": 4,
        "real_assignments_per_window": sorted(
            {len(load_landscape(window).costs) for window in windows}
        ),
        "token_states_per_window": 720,
        "single_seed_qaoa_to_uniform_crossover_readouts": first_crossover(
            windows, grouped, False
        ),
        "four_seed_selector_qaoa_to_uniform_crossover_readouts_per_seed": first_crossover(
            windows, grouped, True
        ),
        "curve": curve_rows,
        "best_cost_budgets": budget_rows,
        "interpretation": (
            "QAOA concentrates optimal probability relative to uniform feasible sampling only "
            "in the low-readout regime. Both methods saturate, and the matched four-seed uniform "
            "selector overtakes before the manuscript's 4096-readout selection point."
        ),
    }
    return curve_rows, budget_rows, summary


def report(summary: Mapping[str, object]) -> str:
    curve = {int(row["readouts_per_seed"]): row for row in summary["curve"]}
    budgets = {int(row["readouts"]): row for row in summary["best_cost_budgets"]}
    return f"""# Uniform-Feasible Null Analysis

The frozen selector uses four optimizer seeds and 4,096 final readouts per seed,
so its maximum final-selection exposure is 16,384 readouts per window.  A
uniform sampler over the 360 legal real assignments is therefore the required
null comparator.  Distinguishable EMPTY tokens duplicate each real assignment
in the 720-state token representation but do not change the uniform real-
assignment distribution.

## Exact-hit curve

| Readouts / seed | QAOA single | Uniform single | QAOA four-seed selector | Uniform four-seed selector |
|---:|---:|---:|---:|---:|
| 32 | {curve[32]['qaoa_single_seed_mean_exact_hit_probability']:.3f} | {curve[32]['uniform_single_seed_mean_exact_hit_probability']:.3f} | {curve[32]['qaoa_four_seed_selector_mean_exact_hit_probability']:.3f} | {curve[32]['uniform_four_seed_selector_mean_exact_hit_probability']:.3f} |
| 64 | {curve[64]['qaoa_single_seed_mean_exact_hit_probability']:.3f} | {curve[64]['uniform_single_seed_mean_exact_hit_probability']:.3f} | {curve[64]['qaoa_four_seed_selector_mean_exact_hit_probability']:.3f} | {curve[64]['uniform_four_seed_selector_mean_exact_hit_probability']:.3f} |
| 128 | {curve[128]['qaoa_single_seed_mean_exact_hit_probability']:.3f} | {curve[128]['uniform_single_seed_mean_exact_hit_probability']:.3f} | {curve[128]['qaoa_four_seed_selector_mean_exact_hit_probability']:.3f} | {curve[128]['uniform_four_seed_selector_mean_exact_hit_probability']:.3f} |
| 256 | {curve[256]['qaoa_single_seed_mean_exact_hit_probability']:.3f} | {curve[256]['uniform_single_seed_mean_exact_hit_probability']:.3f} | {curve[256]['qaoa_four_seed_selector_mean_exact_hit_probability']:.3f} | {curve[256]['uniform_four_seed_selector_mean_exact_hit_probability']:.3f} |
| 4096 | {curve[4096]['qaoa_single_seed_mean_exact_hit_probability']:.3f} | {curve[4096]['uniform_single_seed_mean_exact_hit_probability']:.6f} | {curve[4096]['qaoa_four_seed_selector_mean_exact_hit_probability']:.3f} | {curve[4096]['uniform_four_seed_selector_mean_exact_hit_probability']:.6f} |

The mean single-seed QAOA exact-hit probability first falls below the uniform
null at {summary['single_seed_qaoa_to_uniform_crossover_readouts']} readouts.
For the actual four-seed selection structure, the crossover occurs at
{summary['four_seed_selector_qaoa_to_uniform_crossover_readouts_per_seed']}
readouts per seed.  Thus 4,096 readouts is a saturated, non-discriminating
regime; it cannot support an algorithmic-performance headline.

## Expected best local HPWL

| Readouts | QAOA mean improvement (DBU) | Uniform mean improvement (DBU) | QAOA better/equal/worse | Uniform better/equal/worse | QAOA gap closure | Uniform gap closure |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | {budgets[64]['qaoa_mean_absolute_improvement_dbu']:.1f} | {budgets[64]['uniform_mean_absolute_improvement_dbu']:.1f} | {budgets[64]['qaoa_better_equal_worse_than_initial']} | {budgets[64]['uniform_better_equal_worse_than_initial']} | {100 * budgets[64]['qaoa_mean_gap_closure_improving_windows']:.1f}% | {100 * budgets[64]['uniform_mean_gap_closure_improving_windows']:.1f}% |
| 256 | {budgets[256]['qaoa_mean_absolute_improvement_dbu']:.1f} | {budgets[256]['uniform_mean_absolute_improvement_dbu']:.1f} | {budgets[256]['qaoa_better_equal_worse_than_initial']} | {budgets[256]['uniform_better_equal_worse_than_initial']} | {100 * budgets[256]['qaoa_mean_gap_closure_improving_windows']:.1f}% | {100 * budgets[256]['uniform_mean_gap_closure_improving_windows']:.1f}% |
| 4096 | {budgets[4096]['qaoa_mean_absolute_improvement_dbu']:.1f} | {budgets[4096]['uniform_mean_absolute_improvement_dbu']:.1f} | {budgets[4096]['qaoa_better_equal_worse_than_initial']} | {budgets[4096]['uniform_better_equal_worse_than_initial']} | {100 * budgets[4096]['qaoa_mean_gap_closure_improving_windows']:.1f}% | {100 * budgets[4096]['uniform_mean_gap_closure_improving_windows']:.1f}% |

Gap closure is secondary because small opportunity denominators make its mean
unstable.  Absolute improvement and the exact-hit curve are the primary
descriptive metrics.  At 64 readouts QAOA has useful concentration relative to
uniform sampling, but neither result is a scaling or advantage claim.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-level", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("manifest_hash") != canonical_manifest_hash(manifest):
        raise ValueError("benchmark manifest hash is invalid")
    rows = [row for path in args.run_level for row in read_csv(path)]
    curve, budgets, summary = analyze(manifest, rows)
    summary["provenance"] = {
        "manifest_sha256": sha256(args.manifest),
        "run_level_sha256": {str(path): sha256(path) for path in args.run_level},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "exact_hit_curve.csv", curve)
    write_csv(args.output_dir / "expected_best_budget_summary.csv", budgets)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS_REPORT.md").write_text(report(summary))


if __name__ == "__main__":
    main()
