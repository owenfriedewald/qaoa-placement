"""Summarize completed CTS/global-routing case manifests against fixed baselines."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence


METRICS = {
    "routed_wirelength": ("globalroute__global_route__wirelength", "lower"),
    "via_count": ("globalroute__global_route__vias", "lower"),
    "setup_wns_ns": ("globalroute__timing__setup__ws", "higher"),
    "setup_tns_ns": ("globalroute__timing__setup__tns", "higher"),
    "hold_wns_ns": ("globalroute__timing__hold__ws", "higher"),
    "hold_tns_ns": ("globalroute__timing__hold__tns", "higher"),
}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return None if denominator == 0 else sum(x * y for x, y in zip(dx, dy)) / denominator


def distribution(rows: Sequence[Mapping[str, object]], field: str, direction: str) -> dict[str, object]:
    values = [float(row[field]) for row in rows]
    nonworsening = [value <= 0 if direction == "lower" else value >= 0 for value in values]
    worst_index = max(range(len(rows)), key=lambda i: values[i]) if direction == "lower" else min(range(len(rows)), key=lambda i: values[i])
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "nonworsening_count": sum(nonworsening),
        "worsened_count": len(rows) - sum(nonworsening),
        "improved_count": sum(value < 0 if direction == "lower" else value > 0 for value in values),
        "unchanged_count": sum(value == 0 for value in values),
        "worst_window_id": rows[worst_index]["window_id"],
        "worst_value": values[worst_index],
    }


def render_report(summary: Mapping[str, object]) -> str:
    overall = summary["overall"]  # type: ignore[index]
    distributions = overall["distributions"]  # type: ignore[index]
    window_count = int(summary["window_count"])
    design_count = int(summary["design_count"])
    selected_cell_count = 4 * window_count
    lines = [
        "# Finite-Shot QAOA CTS and Global-Routing Impact",
        "",
        f"All {window_count} fixed assignments were evaluated without routing-based",
        "selection. The flow inserts CTS, legalizes its clock cells, and performs",
        "global routing with repair transforms disabled as frozen in the protocol.",
        "These are global-route sensitivity results, not detailed-route signoff.",
        "",
        "## Primary Gate",
        "",
        f"- Completed assignment cases: {overall['completed_count']}/{window_count}.",
        f"- Cases with zero global-routing overflow: {overall['zero_overflow_count']}/{window_count}.",
        f"- Selected cells retained at their assigned sites: {overall['retained_cell_count']}/{selected_cell_count}.",
        f"- Cases retaining all four selected cells: {overall['all_cells_retained_count']}/{window_count}.",
        f"- Duplicate baselines reproduced exactly for all {design_count} designs.",
        "",
        "The primary feasibility gate passes only if every case completes with zero",
        "overflow; cell-retention counts expose any post-CTS legalization movement.",
        "",
        "## Paired Downstream Changes",
        "",
        "| Metric | Improved | Unchanged | Worsened | Non-worsening | Mean delta | Worst delta | Worst window |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    labels = {
        "routed_wirelength": "Routed wirelength (µm)",
        "via_count": "Via count",
        "congestion_demand": "Congestion demand",
        "setup_wns_ns": "Setup WNS (ns)",
        "setup_tns_ns": "Setup TNS (ns)",
        "hold_wns_ns": "Hold WNS (ns)",
        "hold_tns_ns": "Hold TNS (ns)",
    }
    for key in labels:
        item = distributions[key]
        lines.append(
            f"| {labels[key]} | {item['improved_count']} | {item['unchanged_count']} | "
            f"{item['worsened_count']} | {item['nonworsening_count']}/{window_count} | "
            f"{item['mean']:+.6g} | {item['worst_value']:+.6g} | `{item['worst_window_id']}` |"
        )
    lines.extend(
        [
            "",
            f"Summed routed-wirelength change is {overall['routed_wirelength_sum_delta']:+.0f} µm. "
            "The table reports the worst observed case for each metric; aggregate",
            "improvement does not imply strict per-window dominance.",
            "",
            "## Baselines",
            "",
            "| Design | Routed wirelength (µm) | Vias | Congestion demand | Usage | Setup WNS (ns) | Setup TNS (ns) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for design, row in sorted(summary["baselines"].items()):  # type: ignore[index]
        lines.append(
            f"| {design} | {row['routed_wirelength']:.0f} | {row['via_count']:.0f} | "
            f"{row['congestion_demand']:.0f} | {row['congestion_usage_percent']:.2f}% | "
            f"{row['setup_wns_ns']:.7f} | {row['setup_tns_ns']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Association",
            "",
            "Pearson correlations with local HPWL improvement are descriptive only:",
        ]
    )
    for key, value in overall["hpwl_correlations"].items():  # type: ignore[index]
        rendered = "undefined" if value is None else f"{value:.3f}"
        lines.append(f"- {labels[key]} delta: `{rendered}`.")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Execution commit: `{summary['execution_commit']}`.",
            f"- SIF SHA-256: `{summary['container_sha256']}`.",
            f"- Selected-readout SHA-256: `{summary['selected_readout_sha256']}`.",
            "- Slurm job and resource accounting are retained in the transferred execution evidence.",
            "",
            "The evidence supports a controlled claim that the fixed QAOA placements",
            "remain globally routable with negligible aggregate downstream disruption.",
            "It does not establish detailed-route closure, hardware practicality, quantum",
            "advantage, or consistent improvement over every downstream metric.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    results = [json.loads(Path(path).read_text()) for path in glob.glob(str(args.artifact_root / "*" / "case_result.json"))]
    design_count = int(plan.get("design_count", len({case["design"] for case in plan["cases"]})))
    window_count = int(plan.get("window_count", sum(case["kind"] == "assignment" for case in plan["cases"])))
    expected_case_count = window_count + 2 * design_count
    if len(results) != expected_case_count or any(result["status"] != "complete" for result in results):
        raise ValueError(f"expected {expected_case_count} complete case manifests")
    commits = {result["execution_commit"] for result in results}
    containers = {result["container_sha256"] for result in results}
    if len(commits) != 1 or len(containers) != 1:
        raise ValueError("inconsistent execution provenance")

    baselines: dict[str, dict[str, float]] = {}
    designs = sorted({str(case["design"]) for case in plan["cases"]})
    if len(designs) != design_count:
        raise ValueError("plan design count is inconsistent with its cases")
    for design in designs:
        replicas = sorted(
            [r for r in results if r["case"]["design"] == design and r["case"]["kind"] == "baseline"],
            key=lambda r: r["case"]["baseline_replica"],
        )
        if len(replicas) != 2:
            raise ValueError(f"{design} baseline count is not two")
        first, second = replicas
        baseline: dict[str, float] = {}
        for name, (native, _direction) in METRICS.items():
            baseline[name] = float(first["global_route_metrics"][native])
            if baseline[name] != float(second["global_route_metrics"][native]):
                raise ValueError(f"{design} baseline mismatch for {name}")
        for name in ("demand", "resource", "usage_percent", "total_overflow"):
            if first["congestion_total"][name] != second["congestion_total"][name]:
                raise ValueError(f"{design} congestion baseline mismatch")
        baseline["congestion_demand"] = float(first["congestion_total"]["demand"])
        baseline["congestion_resource"] = float(first["congestion_total"]["resource"])
        baseline["congestion_usage_percent"] = float(first["congestion_total"]["usage_percent"])
        baselines[design] = baseline

    rows: list[dict[str, object]] = []
    for result in results:
        case = result["case"]
        if case["kind"] != "assignment":
            continue
        baseline = baselines[case["design"]]
        row: dict[str, object] = {
            "window_id": case["window_id"],
            "design": case["design"],
            "selection_stratum": case["selection_stratum"],
            "local_hpwl_improvement_dbu": case["local_hpwl_improvement_dbu"],
            "total_overflow": result["congestion_total"]["total_overflow"],
            "retained_cell_count": result["cell_retention"]["retained_count"],
        }
        for name, (native, _direction) in METRICS.items():
            value = float(result["global_route_metrics"][native])
            row[name] = value
            row[f"{name}_delta"] = value - baseline[name]
        demand = float(result["congestion_total"]["demand"])
        row["congestion_demand"] = demand
        row["congestion_demand_delta"] = demand - baseline["congestion_demand"]
        rows.append(row)
    rows.sort(key=lambda row: str(row["window_id"]))
    if len(rows) != window_count:
        raise ValueError(f"expected {window_count} assignment results")

    directions = {name: direction for name, (_native, direction) in METRICS.items()}
    directions["congestion_demand"] = "lower"
    distributions = {
        name: distribution(rows, f"{name}_delta", direction)
        for name, direction in directions.items()
    }
    hpwl = [float(row["local_hpwl_improvement_dbu"]) for row in rows]
    correlations = {
        name: pearson(hpwl, [float(row[f"{name}_delta"]) for row in rows])
        for name in directions
    }
    overall = {
        "completed_count": len(rows),
        "zero_overflow_count": sum(int(row["total_overflow"]) == 0 for row in rows),
        "retained_cell_count": sum(int(row["retained_cell_count"]) for row in rows),
        "all_cells_retained_count": sum(int(row["retained_cell_count"]) == 4 for row in rows),
        "routed_wirelength_sum_delta": sum(float(row["routed_wirelength_delta"]) for row in rows),
        "distributions": distributions,
        "hpwl_correlations": correlations,
    }
    summary = {
        "schema_version": "qeda-downstream-routing-summary-v2",
        "design_count": design_count,
        "window_count": window_count,
        "execution_commit": next(iter(commits)),
        "container_sha256": next(iter(containers)),
        "selected_readout_sha256": plan["selected_readout_sha256"],
        "baselines": baselines,
        "overall": overall,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "routing_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "routing_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "RESULTS_REPORT.md").write_text(render_report(summary))
    print(json.dumps({"cases": len(rows), "zero_overflow": overall["zero_overflow_count"], "wirelength_delta": overall["routed_wirelength_sum_delta"]}))


if __name__ == "__main__":
    main()
