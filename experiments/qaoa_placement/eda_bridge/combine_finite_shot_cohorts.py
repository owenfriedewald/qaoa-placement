"""Combine frozen finite-shot confirmation cohorts with hierarchical intervals."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from .prepare_downstream_routing import sha256


ANALYTICAL_METRICS = (
    "expected_best_64_gap_closure",
    "expected_best_256_gap_closure",
    "expected_best_4096_gap_closure",
)
FROZEN_SEEDS = {11, 17, 23, 29}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def hierarchical_bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    field: str,
    seed: int,
    replicates: int,
) -> list[float]:
    by_design: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        by_design.setdefault(str(row["design"]), []).append(row)
    designs = sorted(by_design)
    if not designs or any(not by_design[design] for design in designs):
        raise ValueError("hierarchical bootstrap requires nonempty design groups")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled: list[float] = []
        for _design_index in designs:
            design = rng.choice(designs)
            group = by_design[design]
            sampled.extend(float(rng.choice(group)[field]) for _row in group)
        estimates.append(statistics.fmean(sampled))
    estimates.sort()
    return [
        estimates[int(0.025 * (replicates - 1))],
        estimates[int(0.975 * (replicates - 1))],
    ]


def aggregate_windows(run_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    by_window: dict[str, list[Mapping[str, str]]] = {}
    for row in run_rows:
        by_window.setdefault(row["window_id"], []).append(row)
    aggregated: list[dict[str, object]] = []
    for window_id, rows in sorted(by_window.items()):
        seeds = {int(row["seed"]) for row in rows}
        if seeds != FROZEN_SEEDS or len(rows) != len(FROZEN_SEEDS):
            raise ValueError(f"{window_id} does not contain the four frozen seeds")
        for key, expected in (
            ("objective_shots", "8192"),
            ("graph", "ring"),
            ("schedule", "reversed"),
            ("p", "3"),
            ("objective", "beat_initial"),
        ):
            if {row[key] for row in rows} != {expected}:
                raise ValueError(f"{window_id} violates frozen {key}={expected}")
        opportunity = float(rows[0]["exact_opportunity_dbu"])
        if any(float(row["exact_opportunity_dbu"]) != opportunity for row in rows):
            raise ValueError(f"{window_id} opportunity changes across seeds")
        aggregated.append(
            {
                "window_id": window_id,
                "design": rows[0]["design"],
                "exact_opportunity_dbu": opportunity,
                "probability_beating_initial": statistics.fmean(
                    float(row["probability_beating_initial"]) for row in rows
                ),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in rows)
                    for metric in ANALYTICAL_METRICS
                    if opportunity > 0
                },
            }
        )
    return aggregated


def analyze(
    run_rows: Sequence[Mapping[str, str]],
    selected_rows: Sequence[Mapping[str, str]],
    seed: int,
    replicates: int,
) -> dict[str, object]:
    windows = aggregate_windows(run_rows)
    window_ids = {str(row["window_id"]) for row in windows}
    if len(selected_rows) != len(window_ids) or {
        row["window_id"] for row in selected_rows
    } != window_ids:
        raise ValueError("selected readout does not exactly cover run-level windows")
    improving = [row for row in windows if float(row["exact_opportunity_dbu"]) > 0]
    selected_improving = [
        row for row in selected_rows if float(row["exact_opportunity_dbu"]) > 0
    ]
    analytical = {}
    for index, metric in enumerate(ANALYTICAL_METRICS):
        analytical[metric] = {
            "mean": statistics.fmean(float(row[metric]) for row in improving),
            "hierarchical_bootstrap_95_ci": hierarchical_bootstrap_ci(
                improving, metric, seed + index, replicates
            ),
        }
    return {
        "design_count": len({str(row["design"]) for row in windows}),
        "window_count": len(windows),
        "improving_window_count": len(improving),
        "seed_count_per_window": len(FROZEN_SEEDS),
        "analytical_improving_windows": analytical,
        "mean_probability_beating_initial_all_windows": statistics.fmean(
            float(row["probability_beating_initial"]) for row in windows
        ),
        "selected_best_of_4096": {
            "exact_hit_count": sum(
                str(row["sampled_exact_hit_4096"]).lower() == "true"
                for row in selected_rows
            ),
            "exact_hit_fraction": statistics.fmean(
                str(row["sampled_exact_hit_4096"]).lower() == "true"
                for row in selected_rows
            ),
            "mean_gap_closure_improving_windows": statistics.fmean(
                float(row["sampled_best_4096_gap_closure"])
                for row in selected_improving
            ),
        },
    }


def render_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Original, Replication, and Pooled Finite-Shot Analysis",
        "",
        "Run-level analytical metrics are averaged over the four frozen seeds",
        "within each window. Intervals then resample designs and windows using",
        "the frozen hierarchical bootstrap. Gap closure is defined only for",
        "windows with positive exact local-HPWL opportunity.",
        "",
        "| Cohort | Designs | Windows | Improving | Expected best @64 | @256 | @4096 | Selected exact hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in [*summary["cohort_order"], "pooled"]:  # type: ignore[index]
        item = summary["analyses"][label]  # type: ignore[index]
        analytical = item["analytical_improving_windows"]
        selected = item["selected_best_of_4096"]
        lines.append(
            f"| {label} | {item['design_count']} | {item['window_count']} | "
            f"{item['improving_window_count']} | "
            f"{analytical['expected_best_64_gap_closure']['mean']:.4f} | "
            f"{analytical['expected_best_256_gap_closure']['mean']:.4f} | "
            f"{analytical['expected_best_4096_gap_closure']['mean']:.4f} | "
            f"{selected['exact_hit_count']}/{item['window_count']} |"
        )
    lines.extend(
        [
            "",
            "The selected rule is the frozen lowest-HPWL best-of-4,096 sample",
            "across seeds 11, 17, 23, and 29 with seed as the deterministic tie",
            "break. This is ideal reduced-space simulation with finite-shot outer",
            "objective estimates, not gate-noise or hardware evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        nargs=3,
        action="append",
        metavar=("LABEL", "RUN_LEVEL_CSV", "SELECTED_READOUT_CSV"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    cohort_data: dict[str, tuple[list[dict[str, str]], list[dict[str, str]]]] = {}
    provenance: dict[str, object] = {}
    all_windows: set[str] = set()
    for label, run_text, selected_text in args.cohort:
        if label in cohort_data or label == "pooled":
            raise ValueError(f"invalid or duplicate cohort label: {label}")
        run_path, selected_path = Path(run_text), Path(selected_text)
        run_rows, selected_rows = load_csv(run_path), load_csv(selected_path)
        windows = {row["window_id"] for row in selected_rows}
        if windows & all_windows:
            raise ValueError("cohort window identifiers overlap")
        all_windows.update(windows)
        cohort_data[label] = (run_rows, selected_rows)
        provenance[label] = {
            "run_level_sha256": sha256(run_path),
            "selected_readout_sha256": sha256(selected_path),
        }
    analyses = {
        label: analyze(run_rows, selected_rows, args.bootstrap_seed, args.bootstrap_replicates)
        for label, (run_rows, selected_rows) in cohort_data.items()
    }
    analyses["pooled"] = analyze(
        [row for run_rows, _selected_rows in cohort_data.values() for row in run_rows],
        [row for _run_rows, selected_rows in cohort_data.values() for row in selected_rows],
        args.bootstrap_seed,
        args.bootstrap_replicates,
    )
    summary = {
        "schema_version": "qeda-finite-shot-cohort-combination-v1",
        "cohort_order": list(cohort_data),
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "replicates": args.bootstrap_replicates,
            "resampling": "design then window after seed aggregation",
        },
        "provenance": provenance,
        "analyses": analyses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS_REPORT.md").write_text(render_report(summary))
    print(json.dumps({label: item["window_count"] for label, item in analyses.items()}))


if __name__ == "__main__":
    main()
