"""Summarize finite-shot optimizer development and confirmatory holdout evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .placement_windows import canonical_manifest_hash, sha256
from .summarize_selected_protocol import bootstrap_mean_ci, mean_by_window, read_csv


def improving_ids(manifest: Mapping[str, object]) -> set[str]:
    return {
        str(window["window_id"])
        for window in manifest["windows"]
        if int(window["initial_hpwl_dbu"]) > int(window["exact_hpwl_dbu"])
    }


def finite_metric_by_window(
    rows: Sequence[Mapping[str, str]], ids: set[str], field: str
) -> dict[str, float]:
    selected = [dict(row) for row in rows if row["window_id"] in ids]
    if field == "sampled_exact_hit_4096":
        for row in selected:
            row[field] = 1.0 if str(row[field]).lower() == "true" else 0.0
    return mean_by_window(selected, field)


def frozen_expected_256_by_window(
    rows: Sequence[Mapping[str, str]], ids: set[str]
) -> dict[str, float]:
    converted = []
    for row in rows:
        if row["method"] != "token_qaoa" or row["window_id"] not in ids:
            continue
        converted.append(
            {
                "window_id": row["window_id"],
                "gap": (
                    float(row["initial_hpwl_dbu"])
                    - float(row["expected_best_256_hpwl_dbu"])
                )
                / float(row["available_improvement_dbu"]),
            }
        )
    return mean_by_window(converted, "gap")


def paired_summary(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, object]:
    keys = sorted(set(left) & set(right))
    differences = [left[key] - right[key] for key in keys]
    return {
        "mean_difference": float(np.mean(differences)),
        "bootstrap_95_ci": list(bootstrap_mean_ci(differences)),
        "wins_ties_losses": [
            sum(value > 1e-12 for value in differences),
            sum(abs(value) <= 1e-12 for value in differences),
            sum(value < -1e-12 for value in differences),
        ],
    }


def selected_validation_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["window_id"], []).append(row)
    output = []
    for window_id, group in sorted(grouped.items()):
        selected = min(
            group,
            key=lambda row: (
                float(row["sampled_best_4096_hpwl_dbu"]),
                int(row["seed"]),
            ),
        )
        output.append(
            {
                **selected,
                "method": "finite_shot_token_qaoa",
                "solution_assignment": selected["sampled_best_4096_assignment"],
                "best_hpwl_dbu": selected["sampled_best_4096_hpwl_dbu"],
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    manifest: Mapping[str, object],
    finite_rows: Sequence[Mapping[str, str]],
    ideal_rows: Sequence[Mapping[str, str]],
    frozen_rows: Sequence[Mapping[str, str]],
    train_aggregate: Sequence[Mapping[str, str]],
    validation_aggregate: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    ids = improving_ids(manifest)
    finite_64 = finite_metric_by_window(finite_rows, ids, "expected_best_64_gap_closure")
    finite_256 = finite_metric_by_window(finite_rows, ids, "expected_best_256_gap_closure")
    finite_4096 = finite_metric_by_window(finite_rows, ids, "expected_best_4096_gap_closure")
    finite_sampled = finite_metric_by_window(finite_rows, ids, "sampled_best_4096_gap_closure")
    ideal_256 = finite_metric_by_window(ideal_rows, ids, "expected_best_256_gap_closure")
    frozen_256 = frozen_expected_256_by_window(frozen_rows, ids)
    greedy_rows = [
        row for row in frozen_rows if row["method"] == "greedy" and row["window_id"] in ids
    ]
    greedy = mean_by_window(greedy_rows, "available_gap_closed")
    finite_exact = finite_metric_by_window(finite_rows, ids, "sampled_exact_hit_4096")
    all_exact = finite_metric_by_window(
        finite_rows,
        {str(row["window_id"]) for row in finite_rows},
        "sampled_exact_hit_4096",
    )
    return {
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_manifest_hash": manifest["manifest_hash"],
        "window_count": len(manifest["windows"]),
        "improving_window_count": len(ids),
        "selected_objective_shots": 8192,
        "development_train": list(train_aggregate),
        "development_validation": list(validation_aggregate),
        "holdout": {
            "mean_expected_best_64_gap_closure": float(np.mean(list(finite_64.values()))),
            "mean_expected_best_256_gap_closure": float(np.mean(list(finite_256.values()))),
            "mean_expected_best_4096_gap_closure": float(
                np.mean(list(finite_4096.values()))
            ),
            "mean_sampled_best_4096_gap_closure": float(
                np.mean(list(finite_sampled.values()))
            ),
            "mean_ideal_objective_expected_best_256_gap_closure": float(
                np.mean(list(ideal_256.values()))
            ),
            "mean_frozen_line_expected_best_256_gap_closure": float(
                np.mean(list(frozen_256.values()))
            ),
            "mean_greedy_gap_closure": float(np.mean(list(greedy.values()))),
            "finite_minus_ideal_objective_expected_best_256": paired_summary(
                finite_256, ideal_256
            ),
            "finite_minus_frozen_line_expected_best_256": paired_summary(
                finite_256, frozen_256
            ),
            "finite_sampled_4096_minus_greedy": paired_summary(finite_sampled, greedy),
            "sampled_best_4096_exact_hit_rate_improving_windows": float(
                np.mean(list(finite_exact.values()))
            ),
            "sampled_best_4096_exact_hit_rate_all_windows": float(
                np.mean(list(all_exact.values()))
            ),
            "mean_probability_beating_initial_all_windows": float(
                np.mean([float(row["probability_beating_initial"]) for row in finite_rows])
            ),
            "mean_objective_estimator_rmse": float(
                np.mean([float(row["objective_estimator_rmse"]) for row in finite_rows])
            ),
            "mean_total_objective_shots_per_run": float(
                np.mean([float(row["total_objective_shots"]) for row in finite_rows])
            ),
            "mean_objective_evaluations_per_run": float(
                np.mean([float(row["objective_evaluations"]) for row in finite_rows])
            ),
        },
    }


def report(summary: Mapping[str, object]) -> str:
    holdout = summary["holdout"]
    ideal = holdout["finite_minus_ideal_objective_expected_best_256"]
    frozen = holdout["finite_minus_frozen_line_expected_best_256"]
    greedy = holdout["finite_sampled_4096_minus_greedy"]
    validation = summary.get("openroad_validation")
    legality = ""
    if validation:
        legality = (
            f"\nAll {validation['total_validation_count']} reported assignments passed "
            "OpenROAD `check_placement` in their complete source designs.\n"
        )
    return f"""# Finite-Shot Outer-Loop Confirmatory Result

The selected ring/reversed/p=3 placement-QAOA protocol remains useful when its
probability-of-beating-initial objective is estimated with 8,192 shots per
outer evaluation. The shot budget was selected on the independent
JPEG/dynamic_node development corpus before this GCD/AES/Ibex confirmation.

## Development selection

On validation, 8,192 objective shots achieved 81.4% analytical
expected-best-of-256 gap closure, versus 77.4% for the 2,048-shot finalist and
80.7% for the exact-probability reference. It therefore passed both
predeclared robustness guardrails.

## Holdout result

Across the {summary['improving_window_count']} improving windows, finite-shot
optimization reaches {100 * holdout['mean_expected_best_256_gap_closure']:.1f}%
analytical expected-best-of-256 gap closure. This is
{100 * ideal['mean_difference']:.1f} percentage points relative to exact-
probability optimization (95% CI
[{100 * ideal['bootstrap_95_ci'][0]:.1f}, {100 * ideal['bootstrap_95_ci'][1]:.1f}])
and {100 * frozen['mean_difference']:.1f} points relative to the original
line/rotating QAOA (95% CI
[{100 * frozen['bootstrap_95_ci'][0]:.1f}, {100 * frozen['bootstrap_95_ci'][1]:.1f}]).

Analytical gap closure is
{100 * holdout['mean_expected_best_64_gap_closure']:.1f}% at 64 readouts and
{100 * holdout['mean_expected_best_4096_gap_closure']:.1f}% at 4,096. The
seeded best-of-4,096 rule closes
{100 * holdout['mean_sampled_best_4096_gap_closure']:.1f}% versus
{100 * holdout['mean_greedy_gap_closure']:.1f}% for greedy search; the paired
difference is {100 * greedy['mean_difference']:.1f} points (95% CI
[{100 * greedy['bootstrap_95_ci'][0]:.1f}, {100 * greedy['bootstrap_95_ci'][1]:.1f}]).

The method remains strongly readout-dependent: the negative 64-readout
expectation means its useful tail is not exposed at a small readout budget.
Each optimizer run consumed a mean of
{holdout['mean_total_objective_shots_per_run']:.0f} objective-estimation shots
across {holdout['mean_objective_evaluations_per_run']:.1f} evaluations. The
mean beating-probability estimator RMSE was
{holdout['mean_objective_estimator_rmse']:.4f}.
{legality}

## Scope

This confirms robustness to finite-shot estimation of the outer objective,
not to gate, readout, calibration, compilation, or phase-oracle noise. Final
quality metrics are computed from the ideal final distribution to isolate
optimizer shot noise. The exact lookup cost phase remains uncompiled and
unresource-costed. No runtime, query-equivalence, hardware-practicality, or
quantum-advantage claim follows.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--finite-run-level", type=Path, required=True)
    parser.add_argument("--ideal-run-level", type=Path, required=True)
    parser.add_argument("--frozen-run-level", type=Path, required=True)
    parser.add_argument("--train-aggregate", type=Path, required=True)
    parser.add_argument("--validation-aggregate", type=Path, required=True)
    parser.add_argument("--openroad-validation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("manifest_hash") != canonical_manifest_hash(manifest):
        raise ValueError("benchmark manifest hash is invalid")
    finite_rows = read_csv(args.finite_run_level)
    summary = summarize(
        manifest,
        finite_rows,
        read_csv(args.ideal_run_level),
        read_csv(args.frozen_run_level),
        read_csv(args.train_aggregate),
        read_csv(args.validation_aggregate),
    )
    if args.openroad_validation:
        validation = json.loads(args.openroad_validation.read_text())
        if validation.get("status") != "pass":
            raise ValueError("OpenROAD validation did not pass")
        summary["openroad_validation"] = validation
    summary["provenance"] = {
        "manifest_sha256": sha256(args.manifest),
        "finite_run_level_sha256": sha256(args.finite_run_level),
        "ideal_run_level_sha256": sha256(args.ideal_run_level),
        "frozen_run_level_sha256": sha256(args.frozen_run_level),
        "train_aggregate_sha256": sha256(args.train_aggregate),
        "validation_aggregate_sha256": sha256(args.validation_aggregate),
    }
    if args.openroad_validation:
        summary["provenance"]["openroad_validation_sha256"] = sha256(
            args.openroad_validation
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = selected_validation_rows(finite_rows)
    write_csv(args.output_dir / "selected_readout.csv", validation_rows)
    (args.output_dir / "finite_shot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS_REPORT.md").write_text(report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
