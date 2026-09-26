"""Combine frozen downstream-routing cohorts without discarding cohort results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from .prepare_downstream_routing import sha256
from .summarize_downstream_classical_comparison import (
    DIRECTIONS,
    LABELS,
    METHODS,
    paired_summary,
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def analyze_rows(
    rows: Sequence[Mapping[str, object]], seed: int, replicates: int
) -> dict[str, object]:
    window_ids = {str(row["window_id"]) for row in rows}
    design_count = len({str(row["design"]) for row in rows})
    method_rows = {
        method: [row for row in rows if row["method"] == method] for method in METHODS
    }
    for method, group in method_rows.items():
        if len(group) != len(window_ids) or {str(row["window_id"]) for row in group} != window_ids:
            raise ValueError(f"{method} does not exactly cover the cohort windows")
    pairwise: dict[str, object] = {}
    retained: dict[str, object] = {}
    for comparator_index, comparator in enumerate(METHODS[1:]):
        pairwise[comparator] = {
            metric: paired_summary(
                method_rows["qaoa"],
                method_rows[comparator],
                metric,
                direction,
                seed + 100 * comparator_index + metric_index,
                replicates,
            )
            for metric_index, (metric, direction) in enumerate(DIRECTIONS.items())
        }
        retained_ids = {
            str(row["window_id"])
            for row in method_rows["qaoa"]
            if as_bool(row["all_cells_retained"])
        } & {
            str(row["window_id"])
            for row in method_rows[comparator]
            if as_bool(row["all_cells_retained"])
        }
        retained[comparator] = {
            "pair_count": len(retained_ids),
            "routed_wirelength": paired_summary(
                [row for row in method_rows["qaoa"] if row["window_id"] in retained_ids],
                [row for row in method_rows[comparator] if row["window_id"] in retained_ids],
                "routed_wirelength",
                DIRECTIONS["routed_wirelength"],
                seed + 1000 + comparator_index,
                replicates,
            ),
        }
    return {
        "window_count": len(window_ids),
        "design_count": design_count,
        "method_routed_wirelength": {
            method: {
                "mean_delta": statistics.fmean(
                    float(row["routed_wirelength_delta"]) for row in group
                ),
                "sum_delta": sum(float(row["routed_wirelength_delta"]) for row in group),
            }
            for method, group in method_rows.items()
        },
        "qaoa_pairwise": pairwise,
        "qaoa_pairwise_all_cells_retained": retained,
    }


def render_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Original, Replication, and Pooled Downstream Analysis",
        "",
        "Every cohort is reported separately before the pooled six-design result.",
        "Differences are QAOA minus comparator; negative routed wirelength favors QAOA.",
        "Intervals use the frozen design-then-window hierarchical bootstrap.",
        "",
        "| Cohort | Designs | Windows | Comparator | Wins/ties/losses | Mean difference (µm) | 95% interval |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for cohort_name in [*summary["cohort_order"], "pooled"]:  # type: ignore[index]
        cohort = summary["analyses"][cohort_name]  # type: ignore[index]
        for comparator in METHODS[1:]:
            item = cohort["qaoa_pairwise"][comparator]["routed_wirelength"]
            ci = item["hierarchical_bootstrap_95_ci"]
            lines.append(
                f"| {cohort_name} | {cohort['design_count']} | {cohort['window_count']} | "
                f"{LABELS[comparator]} | {item['qaoa_wins']}/{item['ties']}/{item['qaoa_losses']} | "
                f"{item['mean_difference']:+.6g} | [{ci[0]:+.6g}, {ci[1]:+.6g}] |"
            )
    lines.extend(
        [
            "",
            "## Full-Retention Sensitivity",
            "",
            "| Cohort | Comparator | Retained pairs | Mean difference (µm) | 95% interval |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for cohort_name in [*summary["cohort_order"], "pooled"]:  # type: ignore[index]
        cohort = summary["analyses"][cohort_name]  # type: ignore[index]
        for comparator in METHODS[1:]:
            retained = cohort["qaoa_pairwise_all_cells_retained"][comparator]
            item = retained["routed_wirelength"]
            ci = item["hierarchical_bootstrap_95_ci"]
            lines.append(
                f"| {cohort_name} | {LABELS[comparator]} | {retained['pair_count']} | "
                f"{item['mean_difference']:+.6g} | [{ci[0]:+.6g}, {ci[1]:+.6g}] |"
            )
    lines.extend(
        [
            "",
            "The pooled analysis broadens design diversity but remains limited to six",
            "open-source Nangate45 designs and a local four-cell placement operator.",
            "It does not establish runtime advantage, detailed-route closure, hardware",
            "practicality, or quantum advantage.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        nargs=3,
        action="append",
        metavar=("LABEL", "SUMMARY_JSON", "RESULTS_CSV"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    cohort_rows: dict[str, list[dict[str, str]]] = {}
    provenance: dict[str, object] = {}
    all_windows: set[str] = set()
    for label, summary_text, results_text in args.cohort:
        if label in cohort_rows or label == "pooled":
            raise ValueError(f"invalid or duplicate cohort label: {label}")
        summary_path, results_path = Path(summary_text), Path(results_text)
        source_summary = json.loads(summary_path.read_text())
        rows = load_csv(results_path)
        windows = {row["window_id"] for row in rows}
        if windows & all_windows:
            raise ValueError("cohort window identifiers overlap")
        if int(source_summary["window_count"]) != len(windows):
            raise ValueError(f"{label} summary and results window counts differ")
        all_windows.update(windows)
        cohort_rows[label] = rows
        provenance[label] = {
            "summary_sha256": sha256(summary_path),
            "results_sha256": sha256(results_path),
        }
    pooled_rows = [row for label in cohort_rows for row in cohort_rows[label]]
    analyses = {
        label: analyze_rows(rows, args.bootstrap_seed, args.bootstrap_replicates)
        for label, rows in cohort_rows.items()
    }
    analyses["pooled"] = analyze_rows(
        pooled_rows, args.bootstrap_seed, args.bootstrap_replicates
    )
    summary = {
        "schema_version": "qeda-downstream-cohort-combination-v1",
        "cohort_order": list(cohort_rows),
        "bootstrap": {
            "seed": args.bootstrap_seed,
            "replicates": args.bootstrap_replicates,
            "resampling": "design then window",
        },
        "provenance": provenance,
        "analyses": analyses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS_REPORT.md").write_text(render_report(summary))
    print(json.dumps({name: value["window_count"] for name, value in analyses.items()}))


if __name__ == "__main__":
    main()
