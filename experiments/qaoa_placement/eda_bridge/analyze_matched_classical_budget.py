"""Evaluate classical local-search arms beyond the original 40-evaluation budget."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .placement_windows import canonical_manifest_hash, sha256
from .run_real_benchmark import load_landscape, random_search, simulated_annealing


BUDGETS = (40, 360, 4096)
SEEDS = (11, 17, 23, 29)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(manifest: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows = []
    for window in manifest["windows"]:
        landscape = load_landscape(window)
        initial = int(landscape.cost_by_assignment[landscape.initial])
        opportunity = initial - landscape.exact_cost
        for budget in BUDGETS:
            for method in ("random_feasible_without_replacement", "simulated_annealing"):
                for seed in SEEDS:
                    if method.startswith("random"):
                        assignment, cost, evaluations = random_search(landscape, budget, seed)
                    else:
                        assignment, cost, evaluations = simulated_annealing(landscape, budget, seed)
                    run_rows.append(
                        {
                            "window_id": window["window_id"],
                            "design": str(window["window_id"]).split("_w")[0],
                            "method": method,
                            "budget": budget,
                            "seed": seed,
                            "evaluations": evaluations,
                            "best_hpwl_dbu": cost,
                            "exact_hit": cost == landscape.exact_cost,
                            "improvement_dbu": initial - cost,
                            "gap_closure": (
                                (initial - cost) / opportunity
                                if opportunity > 0
                                else (1.0 if cost == landscape.exact_cost else 0.0)
                            ),
                            "assignment": json.dumps(assignment),
                        }
                    )
    summary = []
    for budget in BUDGETS:
        for method in ("random_feasible_without_replacement", "simulated_annealing"):
            subset = [row for row in run_rows if row["budget"] == budget and row["method"] == method]
            selected = []
            for window_id in sorted({str(row["window_id"]) for row in subset}):
                group = [row for row in subset if row["window_id"] == window_id]
                selected.append(min(group, key=lambda row: (int(row["best_hpwl_dbu"]), int(row["seed"]))))
            summary.append(
                {
                    "method": method,
                    "budget_per_seed": budget,
                    "seeds": len(SEEDS),
                    "mean_actual_evaluations_per_seed": float(
                        np.mean([int(row["evaluations"]) for row in subset])
                    ),
                    "mean_actual_selector_evaluations_per_window": float(
                        len(SEEDS)
                        * np.mean([int(row["evaluations"]) for row in subset])
                    ),
                    "windows": len(selected),
                    "selected_exact_hits": int(sum(bool(row["exact_hit"]) for row in selected)),
                    "mean_selected_improvement_dbu": float(np.mean([float(row["improvement_dbu"]) for row in selected])),
                    "mean_selected_gap_closure": float(np.mean([float(row["gap_closure"]) for row in selected])),
                }
            )
    return run_rows, summary


def report(summary: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Matched Classical Evaluation-Budget Analysis",
        "",
        "Each method uses the same four frozen seeds. The selected result is the",
        "lowest local HPWL across seeds with seed as tie break.",
        "",
        "| Method | Budget / seed | Actual evaluations / seed | Actual selector evaluations | Exact selected | Mean improvement (DBU) | Mean gap closure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['budget_per_seed']} | "
            f"{float(row['mean_actual_evaluations_per_seed']):.0f} | "
            f"{float(row['mean_actual_selector_evaluations_per_window']):.0f} | "
            f"{row['selected_exact_hits']}/{row['windows']} | {float(row['mean_selected_improvement_dbu']):.1f} | "
            f"{100 * float(row['mean_selected_gap_closure']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "The 360-state space can be exhausted classically.  Consequently,",
            "the 40-evaluation greedy/annealing arms used for the routed sensitivity",
            "table are not competitive baselines for a 4,096-readout quantum",
            "selector.  Their downstream rows are retained only as assignment",
            "sensitivity cases; no performance ordering is inferred.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("manifest_hash") != canonical_manifest_hash(manifest):
        raise ValueError("benchmark manifest hash is invalid")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    run_rows, summary = analyze(manifest)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "run_level.csv", run_rows)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "RESULTS_REPORT.md").write_text(report(summary))
    (args.output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "manifest_sha256": sha256(args.manifest),
                "budgets": list(BUDGETS),
                "seeds": list(SEEDS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
