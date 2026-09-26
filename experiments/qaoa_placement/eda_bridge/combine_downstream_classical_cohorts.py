"""Pool frozen downstream QAOA/classical routing cohorts.

Each input cohort must already have passed
``summarize_downstream_classical_comparison``.  This module revalidates the
run-level coverage, reproduces the cohort summaries, and then applies the same
design/window hierarchical bootstrap to the pooled design corpus.
"""

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
    distribution,
    paired_summary,
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    numeric = tuple(DIRECTIONS) + tuple(f"{metric}_delta" for metric in DIRECTIONS)
    for raw in rows:
        row: dict[str, object] = dict(raw)
        for field in numeric:
            row[field] = float(raw[field])
        row["total_overflow"] = int(raw["total_overflow"])
        row["retained_cell_count"] = int(raw["retained_cell_count"])
        row["all_cells_retained"] = str(raw["all_cells_retained"]).lower() == "true"
        normalized.append(row)
    return normalized


def identity_sum(
    summaries: Sequence[Mapping[str, object]], window_count: int
) -> dict[str, object]:
    combined: dict[str, object] = {}
    for comparator in METHODS[1:]:
        items = [
            summary["assignment_identity_with_qaoa"][comparator]  # type: ignore[index]
            for summary in summaries
        ]
        identical = sum(int(item["identical_assignment_count"]) for item in items)
        same_routed_windows = [
            str(window)
            for item in items
            for window in item["different_assignment_same_routed_wirelength_windows"]
        ]
        combined[comparator] = {
            "identical_assignment_count": identical,
            "different_assignment_count": window_count - identical,
            "different_assignment_same_routed_wirelength_count": len(same_routed_windows),
            "different_assignment_same_routed_wirelength_windows": sorted(same_routed_windows),
        }
    return combined


def analyze_rows(
    rows: Sequence[Mapping[str, object]],
    source_summaries: Sequence[Mapping[str, object]],
    seed: int,
    replicates: int,
) -> dict[str, object]:
    window_ids = {str(row["window_id"]) for row in rows}
    designs = {str(row["design"]) for row in rows}
    if not window_ids or not designs:
        raise ValueError("downstream cohort must be nonempty")
    for window_id in window_ids:
        methods = [str(row["method"]) for row in rows if row["window_id"] == window_id]
        if sorted(methods) != sorted(METHODS):
            raise ValueError(f"{window_id} does not contain exactly one row per method")

    method_rows = {
        method: [row for row in rows if row["method"] == method] for method in METHODS
    }
    method_summaries: dict[str, object] = {}
    for method, group in method_rows.items():
        method_summaries[method] = {
            "zero_overflow_count": sum(int(row["total_overflow"]) == 0 for row in group),
            "retained_cell_count": sum(int(row["retained_cell_count"]) for row in group),
            "all_cells_retained_count": sum(bool(row["all_cells_retained"]) for row in group),
            "routed_wirelength_sum_delta": sum(
                float(row["routed_wirelength_delta"]) for row in group
            ),
            "distributions": {
                metric: distribution(group, f"{metric}_delta", direction)
                for metric, direction in DIRECTIONS.items()
            },
        }

    paired: dict[str, object] = {}
    retained_paired: dict[str, object] = {}
    for comparator_index, comparator in enumerate(METHODS[1:]):
        paired[comparator] = {
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
            if bool(row["all_cells_retained"])
        } & {
            str(row["window_id"])
            for row in method_rows[comparator]
            if bool(row["all_cells_retained"])
        }
        retained_paired[comparator] = {
            "pair_count": len(retained_ids),
            "window_ids": sorted(retained_ids),
            "routed_wirelength": paired_summary(
                [row for row in method_rows["qaoa"] if row["window_id"] in retained_ids],
                [row for row in method_rows[comparator] if row["window_id"] in retained_ids],
                "routed_wirelength",
                DIRECTIONS["routed_wirelength"],
                seed + 1000 + comparator_index,
                replicates,
            ),
        }

    classical = [row for row in rows if row["method"] != "qaoa"]
    return {
        "design_count": len(designs),
        "window_count": len(window_ids),
        "routing_case_count": len(rows),
        "routing_zero_overflow_count": sum(int(row["total_overflow"]) == 0 for row in rows),
        "technical": {
            "attempted_count": len(classical),
            "completed_count": len(classical),
            "zero_overflow_count": sum(
                int(row["total_overflow"]) == 0 for row in classical
            ),
            "retained_cell_count": sum(int(row["retained_cell_count"]) for row in classical),
            "all_cells_retained_count": sum(
                bool(row["all_cells_retained"]) for row in classical
            ),
        },
        "method_summaries": method_summaries,
        "qaoa_pairwise": paired,
        "qaoa_pairwise_all_cells_retained": retained_paired,
        "assignment_identity_with_qaoa": identity_sum(source_summaries, len(window_ids)),
    }


def validate_reproduction(
    label: str, calculated: Mapping[str, object], source: Mapping[str, object]
) -> None:
    for field in ("window_count", "design_count", "technical"):
        if calculated[field] != source[field]:
            raise ValueError(f"{label} pooled-input reproduction failed for {field}")
    for method in METHODS:
        got = calculated["method_summaries"][method]["routed_wirelength_sum_delta"]  # type: ignore[index]
        expected = source["method_summaries"][method]["routed_wirelength_sum_delta"]  # type: ignore[index]
        if float(got) != float(expected):
            raise ValueError(f"{label} routed-wirelength reproduction failed for {method}")


def render_report(summary: Mapping[str, object]) -> str:
    order = [*summary["cohort_order"], "pooled"]  # type: ignore[index]
    lines = [
        "# Original, Replication, and Pooled Downstream Comparison",
        "",
        "Each cohort was frozen before downstream comparator routing. Cohort and",
        "pooled intervals use the same design-then-window hierarchical bootstrap.",
        "Negative routed-wirelength deltas improve on unchanged placement; negative",
        "QAOA-minus-comparator differences favor QAOA.",
        "",
        "| Cohort | Designs | Windows | Routed cases | Zero overflow | QAOA sum delta (µm) | QAOA=exact assignment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in order:
        item = summary["analyses"][label]  # type: ignore[index]
        identity = item["assignment_identity_with_qaoa"]["exact"]
        lines.append(
            f"| {label} | {item['design_count']} | {item['window_count']} | "
            f"{item['routing_case_count']} | {item['routing_zero_overflow_count']} | "
            f"{item['method_summaries']['qaoa']['routed_wirelength_sum_delta']:+.0f} | "
            f"{identity['identical_assignment_count']}/{item['window_count']} |"
        )
    pooled = summary["analyses"]["pooled"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "## Pooled Routed-Wirelength Outcome",
            "",
            "| Method | Improved | Unchanged | Worsened | Mean delta (µm) | Sum delta (µm) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        method_summary = pooled["method_summaries"][method]
        dist = method_summary["distributions"]["routed_wirelength"]
        lines.append(
            f"| {LABELS[method]} | {dist['improved_count']} | {dist['unchanged_count']} | "
            f"{dist['worsened_count']} | {dist['mean']:+.6g} | "
            f"{method_summary['routed_wirelength_sum_delta']:+.0f} |"
        )
    lines.extend(
        [
            "",
            "| Comparator | QAOA wins | Ties | QAOA losses | Mean difference (µm) | 95% interval |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparator in METHODS[1:]:
        item = pooled["qaoa_pairwise"][comparator]["routed_wirelength"]
        ci = item["hierarchical_bootstrap_95_ci"]
        lines.append(
            f"| {LABELS[comparator]} | {item['qaoa_wins']} | {item['ties']} | "
            f"{item['qaoa_losses']} | {item['mean_difference']:+.6g} | "
            f"[{ci[0]:+.6g}, {ci[1]:+.6g}] |"
        )
    technical = pooled["technical"]
    lines.extend(
        [
            "",
            "## Completeness and Scope",
            "",
            f"- Classical cases completed: {technical['completed_count']}/{technical['attempted_count']}.",
            f"- Classical cases with zero overflow: {technical['zero_overflow_count']}/{technical['attempted_count']}.",
            f"- Selected classical cells retained after CTS: {technical['retained_cell_count']}/{4 * technical['attempted_count']}.",
            "- This is controlled CTS/global-route sensitivity evidence, not detailed-route signoff, runtime advantage, or hardware evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def combine(
    cohorts: Sequence[tuple[str, Path, Path]], seed: int, replicates: int
) -> dict[str, object]:
    if len(cohorts) < 2:
        raise ValueError("at least two cohorts are required")
    cohort_rows: dict[str, list[dict[str, object]]] = {}
    cohort_summaries: dict[str, Mapping[str, object]] = {}
    provenance: dict[str, object] = {}
    seen_windows: set[str] = set()
    seen_designs: set[str] = set()
    for label, results_path, summary_path in cohorts:
        if label in cohort_rows or label == "pooled":
            raise ValueError(f"invalid or duplicate cohort label: {label}")
        rows = normalize_rows(load_csv(results_path))
        source = json.loads(summary_path.read_text())
        windows = {str(row["window_id"]) for row in rows}
        designs = {str(row["design"]) for row in rows}
        if windows & seen_windows or designs & seen_designs:
            raise ValueError("cohort window or design identifiers overlap")
        seen_windows.update(windows)
        seen_designs.update(designs)
        calculated = analyze_rows(rows, [source], seed, replicates)
        validate_reproduction(label, calculated, source)
        cohort_rows[label] = rows
        cohort_summaries[label] = source
        provenance[label] = {
            "comparison_results_sha256": sha256(results_path),
            "comparison_summary_sha256": sha256(summary_path),
            "classical_execution_commit": source["execution_commit"],
            "qaoa_execution_commit": source["qaoa_execution_commit"],
            "container_sha256": source["container_sha256"],
        }
    containers = {item["container_sha256"] for item in provenance.values()}  # type: ignore[union-attr]
    if len(containers) != 1:
        raise ValueError("cohorts use different routing containers")
    analyses = {
        label: analyze_rows(rows, [cohort_summaries[label]], seed, replicates)
        for label, rows in cohort_rows.items()
    }
    analyses["pooled"] = analyze_rows(
        [row for rows in cohort_rows.values() for row in rows],
        list(cohort_summaries.values()),
        seed,
        replicates,
    )
    return {
        "schema_version": "qeda-downstream-classical-cohort-combination-v1",
        "cohort_order": list(cohort_rows),
        "bootstrap": {
            "seed": seed,
            "replicates": replicates,
            "resampling": "design then window",
        },
        "provenance": provenance,
        "analyses": analyses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        nargs=3,
        action="append",
        metavar=("LABEL", "COMPARISON_RESULTS_CSV", "COMPARISON_SUMMARY_JSON"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    summary = combine(
        [(label, Path(results), Path(source)) for label, results, source in args.cohort],
        args.bootstrap_seed,
        args.bootstrap_replicates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS_REPORT.md").write_text(render_report(summary))
    pooled = summary["analyses"]["pooled"]
    print(
        json.dumps(
            {
                "designs": pooled["design_count"],
                "windows": pooled["window_count"],
                "routing_cases": pooled["routing_case_count"],
                "zero_overflow": pooled["routing_zero_overflow_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
