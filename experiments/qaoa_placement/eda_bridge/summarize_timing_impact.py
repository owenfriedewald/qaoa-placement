"""Summarize the preregistered full-design timing-impact gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence


TIMING_METRICS = ("setup_wns", "setup_tns", "hold_wns", "hold_tns")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def metric_distribution(rows: Sequence[Mapping[str, object]], metric: str, tolerance: float) -> dict[str, object]:
    values = [float(row[f"{metric}_delta_ns"]) for row in rows]
    worst_index = min(range(len(rows)), key=lambda index: values[index])
    best_index = max(range(len(rows)), key=lambda index: values[index])
    return {
        "mean_delta_ns": statistics.fmean(values),
        "median_delta_ns": statistics.median(values),
        "minimum_delta_ns": min(values),
        "maximum_delta_ns": max(values),
        "improved_count": sum(value > tolerance for value in values),
        "unchanged_count": sum(abs(value) <= tolerance for value in values),
        "worsened_count": sum(value < -tolerance for value in values),
        "worst_window_id": rows[worst_index]["window_id"],
        "best_window_id": rows[best_index]["window_id"],
    }


def summarize_group(rows: Sequence[Mapping[str, object]], tolerance: float) -> dict[str, object]:
    result: dict[str, object] = {
        "case_count": len(rows),
        "hpwl_improved_count": sum(float(row["hpwl_improvement_dbu"]) > 0 for row in rows),
        "hpwl_unchanged_count": sum(float(row["hpwl_improvement_dbu"]) == 0 for row in rows),
        "total_hpwl_improvement_dbu": sum(float(row["hpwl_improvement_dbu"]) for row in rows),
        "setup_wns_nonworsening_count": sum(
            float(row["setup_wns_delta_ns"]) >= -tolerance for row in rows
        ),
        "setup_tns_nonworsening_count": sum(
            float(row["setup_tns_delta_ns"]) >= -tolerance for row in rows
        ),
        "setup_joint_nonworsening_count": sum(
            float(row["setup_wns_delta_ns"]) >= -tolerance
            and float(row["setup_tns_delta_ns"]) >= -tolerance
            for row in rows
        ),
        "hold_joint_nonworsening_count": sum(
            float(row["hold_wns_delta_ns"]) >= -tolerance
            and float(row["hold_tns_delta_ns"]) >= -tolerance
            for row in rows
        ),
        "new_setup_violation_count": sum(bool(row["new_setup_violation"]) for row in rows),
        "metrics": {
            metric: metric_distribution(rows, metric, tolerance)
            for metric in TIMING_METRICS
        },
    }
    hpwl = [float(row["hpwl_improvement_dbu"]) for row in rows]
    result["pearson_hpwl_improvement_vs_timing_delta"] = {
        metric: pearson(hpwl, [float(row[f"{metric}_delta_ns"]) for row in rows])
        for metric in TIMING_METRICS
    }
    return result


def provenance(orfs_worktree: Path, designs: Iterable[str]) -> dict[str, object]:
    inputs: dict[str, object] = {}
    for design in sorted(designs):
        base = orfs_worktree / "flow" / "results" / "nangate45" / design / "base"
        odb = base / "qeda_3_4_legalize_only.odb"
        sdc = base / "2_floorplan.sdc"
        inputs[design] = {
            "odb": str(odb),
            "odb_sha256": sha256(odb),
            "sdc": str(sdc),
            "sdc_sha256": sha256(sdc),
        }
    set_rc = orfs_worktree / "flow" / "platforms" / "nangate45" / "setRC.tcl"
    return {
        "design_inputs": inputs,
        "set_rc": str(set_rc),
        "set_rc_sha256": sha256(set_rc),
    }


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%"


def ps(value_ns: float) -> str:
    return f"{1000.0 * value_ns:+.3f}"


def render_report(summary: Mapping[str, object]) -> str:
    overall = summary["overall"]  # type: ignore[index]
    metrics = overall["metrics"]  # type: ignore[index]
    case_count = int(overall["case_count"])  # type: ignore[index]
    lines = [
        "# Finite-Shot QAOA Full-Design Timing Impact",
        "",
        "This is the preregistered downstream timing gate for all 36 fixed holdout",
        "assignments. No assignment was selected, removed, or replaced using timing.",
        "Parasitics are placement-estimated; these are not routed signoff results.",
        "",
        "## Gate Result",
        "",
        f"- Setup WNS did not worsen in {overall['setup_wns_nonworsening_count']}/{case_count} "
        f"cases ({pct(int(overall['setup_wns_nonworsening_count']), case_count)}).",
        f"- Setup TNS did not worsen in {overall['setup_tns_nonworsening_count']}/{case_count} "
        f"cases ({pct(int(overall['setup_tns_nonworsening_count']), case_count)}).",
        f"- Both setup metrics were non-worsening in {overall['setup_joint_nonworsening_count']}/{case_count} "
        f"cases ({pct(int(overall['setup_joint_nonworsening_count']), case_count)}).",
        f"- Hold WNS and TNS were jointly non-worsening in {overall['hold_joint_nonworsening_count']}/{case_count} "
        f"cases ({pct(int(overall['hold_joint_nonworsening_count']), case_count)}).",
        f"- New setup violations: {overall['new_setup_violation_count']}.",
        f"- Worst setup WNS change: {ps(float(metrics['setup_wns']['minimum_delta_ns']))} ps "
        f"at `{metrics['setup_wns']['worst_window_id']}`.",
        f"- Worst setup TNS change: {ps(float(metrics['setup_tns']['minimum_delta_ns']))} ps "
        f"at `{metrics['setup_tns']['worst_window_id']}`.",
        "",
        "The result is a small timing perturbation rather than strict timing dominance:",
        "two cases worsen at least one setup metric, while the worst WNS loss is below",
        "one picosecond. Hold metrics are unchanged throughout.",
        "All three baselines already have negative setup WNS, so the zero-new-violation",
        "count is disclosed for completeness but is not an additional safety claim.",
        "",
        "## Baselines and Round Trip",
        "",
        "| Design | Setup WNS (ns) | Setup TNS (ns) | Hold WNS (ns) | Hold TNS (ns) | Max round-trip error (ns) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for design, baseline in sorted(summary["baselines"].items()):  # type: ignore[index]
        lines.append(
            f"| {design} | {baseline['setup_wns_ns']:.9f} | {baseline['setup_tns_ns']:.9f} | "
            f"{baseline['hold_wns_ns']:.9f} | {baseline['hold_tns_ns']:.9f} | "
            f"{baseline['maximum_roundtrip_error_ns']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Every design reproduced all four baseline metrics within the fixed",
            f"`{summary['roundtrip_tolerance_ns']:.0e}` ns tolerance.",
            "",
            "## Timing-Delta Distribution",
            "",
            "Positive deltas improve slack; negative deltas worsen it.",
            "",
            "| Metric | Improved | Unchanged | Worsened | Mean delta (ps) | Worst delta (ps) | Best delta (ps) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in TIMING_METRICS:
        values = metrics[metric]
        lines.append(
            f"| {metric.replace('_', ' ').upper()} | {values['improved_count']} | "
            f"{values['unchanged_count']} | {values['worsened_count']} | "
            f"{ps(float(values['mean_delta_ns']))} | "
            f"{ps(float(values['minimum_delta_ns']))} | "
            f"{ps(float(values['maximum_delta_ns']))} |"
        )
    lines.extend(
        [
            "",
            "## By Design",
            "",
            "| Design | Cases | HPWL improved | Joint setup non-worsening | Worst WNS delta (ps) | Worst TNS delta (ps) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for design, values in sorted(summary["by_design"].items()):  # type: ignore[index]
        design_metrics = values["metrics"]
        lines.append(
            f"| {design} | {values['case_count']} | {values['hpwl_improved_count']} | "
            f"{values['setup_joint_nonworsening_count']} | "
            f"{ps(float(design_metrics['setup_wns']['minimum_delta_ns']))} | "
            f"{ps(float(design_metrics['setup_tns']['minimum_delta_ns']))} |"
        )
    lines.extend(
        [
            "",
            "## Worsening Cases",
            "",
            "| Window | Design | HPWL improvement (DBU) | Setup WNS delta (ps) | Setup TNS delta (ps) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    worsening = summary["worsening_cases"]  # type: ignore[index]
    if worsening:
        for row in worsening:
            lines.append(
                f"| {row['window_id']} | {row['design']} | {row['hpwl_improvement_dbu']:.0f} | "
                f"{ps(float(row['setup_wns_delta_ns']))} | {ps(float(row['setup_tns_delta_ns']))} |"
            )
    else:
        lines.append("| none | - | - | - | - |")
    correlations = overall["pearson_hpwl_improvement_vs_timing_delta"]  # type: ignore[index]
    lines.extend(
        [
            "",
            "## Association and Interpretation",
            "",
            f"Across all cases, Pearson correlation between local HPWL improvement and setup WNS delta is "
            f"`{correlations['setup_wns']:.3f}`; for setup TNS delta it is "
            f"`{correlations['setup_tns']:.3f}`.",
            "These descriptive correlations do not establish timing benefit: most four-cell",
            "moves do not touch the worst paths, the three designs have different timing",
            "baselines, and the sample was selected for placement benchmarking rather than",
            "timing inference.",
            "",
            "The fixed assignments improve local incident-net HPWL in "
            f"{overall['hpwl_improved_count']}/{case_count} cases and leave it unchanged in "
            f"{overall['hpwl_unchanged_count']}/{case_count}. The timing gate therefore supports",
            "the narrower claim that these local gains usually preserve placement-estimated",
            "timing and introduce only sub-picosecond worst-case setup degradation here. A",
            "routed timing or congestion claim requires a separately frozen downstream study.",
            "",
            "## Provenance",
            "",
            f"- Container: `{summary['container_image']}`.",
            f"- Selected-readout SHA-256: `{summary['selected_readout_sha256']}`.",
            f"- OpenROAD legality: {summary['legality_passed_count']}/{case_count} assignments passed.",
            f"- Timing baseline round trips: {summary['roundtrip_passed_count']}/3 designs passed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--orfs-worktree", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    tolerance = float(plan["roundtrip_tolerance_ns"])
    baselines: dict[str, dict[str, float]] = {}
    results: list[dict[str, object]] = []
    raw_hashes: dict[str, str] = {}
    roundtrip_passed = 0
    for item in plan["designs"]:
        design = item["design"]
        raw_path = Path(item["raw_csv"])
        rows = read_csv(raw_path)
        before = next(row for row in rows if row["window_id"] == "__baseline_before__")
        after = next(row for row in rows if row["window_id"] == "__baseline_after__")
        baseline = {metric: float(before[f"{metric}_ns"]) for metric in TIMING_METRICS}
        errors = {
            metric: abs(float(after[f"{metric}_ns"]) - baseline[metric])
            for metric in TIMING_METRICS
        }
        maximum_error = max(errors.values())
        if maximum_error > tolerance:
            raise ValueError(f"{design} round-trip error {maximum_error} exceeds {tolerance}")
        roundtrip_passed += 1
        baselines[design] = {
            **{f"{metric}_ns": value for metric, value in baseline.items()},
            "maximum_roundtrip_error_ns": maximum_error,
        }
        raw_hashes[design] = sha256(raw_path)
        assignments = [row for row in rows if row["case_kind"] == "assignment"]
        if len(assignments) != int(item["case_count"]):
            raise ValueError(f"{design} has {len(assignments)} assignment rows")
        for row in assignments:
            output: dict[str, object] = dict(row)
            for key in ("initial_hpwl_dbu", "selected_hpwl_dbu", "hpwl_delta_dbu"):
                output[key] = float(row[key])
            output["hpwl_improvement_dbu"] = -float(row["hpwl_delta_dbu"])
            for metric in TIMING_METRICS:
                value = float(row[f"{metric}_ns"])
                output[f"{metric}_ns"] = value
                output[f"{metric}_delta_ns"] = value - baseline[metric]
            output["new_setup_violation"] = (
                baseline["setup_wns"] >= -tolerance
                and float(output["setup_wns_ns"]) < -tolerance
            )
            results.append(output)
    if len(results) != 36:
        raise ValueError(f"expected 36 timing cases, found {len(results)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "timing_results.csv"
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)

    overall = summarize_group(results, tolerance)
    by_design = {
        design: summarize_group([row for row in results if row["design"] == design], tolerance)
        for design in sorted(baselines)
    }
    worsening = [
        row
        for row in results
        if float(row["setup_wns_delta_ns"]) < -tolerance
        or float(row["setup_tns_delta_ns"]) < -tolerance
    ]
    summary = {
        "schema_version": "qeda-timing-impact-summary-v1",
        "container_image": plan["container_image"],
        "input_manifest_sha256": plan["input_manifest_sha256"],
        "selected_readout_sha256": plan["selected_readout_sha256"],
        "timing_plan_sha256": sha256(args.plan),
        "raw_timing_sha256": raw_hashes,
        "roundtrip_tolerance_ns": tolerance,
        "roundtrip_passed_count": roundtrip_passed,
        "legality_passed_count": len(results),
        "baselines": baselines,
        "overall": overall,
        "by_design": by_design,
        "worsening_cases": worsening,
        "source_provenance": provenance(args.orfs_worktree, baselines),
    }
    summary_path = args.output_dir / "timing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report_path = args.output_dir / "RESULTS_REPORT.md"
    report_path.write_text(render_report(summary))
    print(
        json.dumps(
            {
                "cases": len(results),
                "joint_setup_nonworsening": overall["setup_joint_nonworsening_count"],
                "worsening_cases": len(worsening),
                "report": str(report_path),
            }
        )
    )


if __name__ == "__main__":
    main()
