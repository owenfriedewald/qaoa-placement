"""Summarize frozen classical routing cases against QAOA and unchanged placement."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from experiments.qaoa_placement.eda_bridge.prepare_downstream_routing import sha256
from experiments.qaoa_placement.eda_bridge.summarize_downstream_routing import (
    METRICS,
    distribution,
)


METHODS = ("qaoa", "exact", "greedy", "simulated_annealing")
LABELS = {
    "qaoa": "Finite-shot QAOA",
    "exact": "Exact local HPWL",
    "greedy": "Greedy",
    "simulated_annealing": "Simulated annealing",
}
METRIC_LABELS = {
    "routed_wirelength": "Routed wirelength (µm)",
    "via_count": "Via count",
    "congestion_demand": "Congestion demand",
    "setup_wns_ns": "Setup WNS (ns)",
    "setup_tns_ns": "Setup TNS (ns)",
    "hold_wns_ns": "Hold WNS (ns)",
    "hold_tns_ns": "Hold TNS (ns)",
}
DIRECTIONS = {name: direction for name, (_native, direction) in METRICS.items()} | {
    "congestion_demand": "lower"
}


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


def paired_summary(
    qaoa_rows: Sequence[Mapping[str, object]],
    comparator_rows: Sequence[Mapping[str, object]],
    metric: str,
    direction: str,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    qaoa = {str(row["window_id"]): row for row in qaoa_rows}
    comparator = {str(row["window_id"]): row for row in comparator_rows}
    if set(qaoa) != set(comparator) or not qaoa:
        raise ValueError("paired methods must contain the same nonempty window set")
    paired: list[dict[str, object]] = []
    for window_id in sorted(qaoa):
        difference = float(qaoa[window_id][metric]) - float(comparator[window_id][metric])
        paired.append(
            {
                "window_id": window_id,
                "design": str(qaoa[window_id]["design"]),
                "difference": difference,
            }
        )
    qaoa_wins = sum(
        float(row["difference"]) < 0 if direction == "lower" else float(row["difference"]) > 0
        for row in paired
    )
    ties = sum(float(row["difference"]) == 0 for row in paired)
    return {
        "difference_definition": "qaoa_minus_comparator",
        "mean_difference": statistics.fmean(float(row["difference"]) for row in paired),
        "median_difference": statistics.median(float(row["difference"]) for row in paired),
        "hierarchical_bootstrap_95_ci": hierarchical_bootstrap_ci(
            paired, "difference", seed, replicates
        ),
        "qaoa_wins": qaoa_wins,
        "ties": ties,
        "qaoa_losses": len(paired) - qaoa_wins - ties,
    }


def render_report(summary: Mapping[str, object]) -> str:
    technical = summary["technical"]  # type: ignore[index]
    method_summaries = summary["method_summaries"]  # type: ignore[index]
    paired = summary["qaoa_pairwise"]  # type: ignore[index]
    identity = summary["assignment_identity_with_qaoa"]  # type: ignore[index]
    retained = summary["qaoa_pairwise_all_cells_retained"]  # type: ignore[index]
    window_count = int(summary["window_count"])
    design_count = int(summary["design_count"])
    classical_case_count = int(technical["attempted_count"])
    lines = [
        "# QAOA Versus Classical Downstream Routing",
        "",
        "The frozen exact, greedy, and best-of-two simulated-annealing assignments",
        "were routed without consulting downstream outcomes. They are compared with",
        "the previously completed finite-shot QAOA assignments and duplicate-verified",
        "unchanged-placement baselines. This is global-route sensitivity evidence, not",
        "detailed-route signoff.",
        "",
        "## Completeness",
        "",
        f"- New classical cases completed: {technical['completed_count']}/{classical_case_count}.",
        f"- New cases with zero global-routing overflow: {technical['zero_overflow_count']}/{classical_case_count}.",
        f"- New selected cells retained after CTS legalization: {technical['retained_cell_count']}/{4 * classical_case_count}.",
        f"- New cases retaining all four selected cells: {technical['all_cells_retained_count']}/{classical_case_count}.",
        "",
        "## Primary Outcome: Routed Wirelength",
        "",
        "Changes are paired against unchanged placement for the corresponding design.",
        "",
        "| Method | Improved | Unchanged | Worsened | Mean delta (µm) | Sum delta (µm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = method_summaries[method]["distributions"]["routed_wirelength"]
        lines.append(
            f"| {LABELS[method]} | {item['improved_count']} | {item['unchanged_count']} | "
            f"{item['worsened_count']} | {item['mean']:+.6g} | "
            f"{method_summaries[method]['routed_wirelength_sum_delta']:+.0f} |"
        )
    lines.extend(
        [
            "",
            "Paired differences below are QAOA minus comparator, so a negative routed-",
            "wirelength difference favors QAOA. Intervals use the frozen hierarchical",
            f"bootstrap over {design_count} design clusters.",
            "",
            "| Comparator | QAOA wins | Ties | QAOA losses | Mean difference (µm) | 95% interval |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for comparator in METHODS[1:]:
        item = paired[comparator]["routed_wirelength"]
        ci = item["hierarchical_bootstrap_95_ci"]
        lines.append(
            f"| {LABELS[comparator]} | {item['qaoa_wins']} | {item['ties']} | "
            f"{item['qaoa_losses']} | {item['mean_difference']:+.6g} | "
            f"[{ci[0]:+.6g}, {ci[1]:+.6g}] |"
        )
    lines.extend(
        [
            "",
            "## Assignment Identity Audit",
            "",
            "This descriptive audit was added to explain the preregistered paired",
            "routing outcomes; it does not change the primary metric or interval.",
            "",
            "| Comparator | Same assignment as QAOA | Different assignment | Different assignment, same routed wirelength |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparator in METHODS[1:]:
        item = identity[comparator]
        lines.append(
            f"| {LABELS[comparator]} | {item['identical_assignment_count']}/{window_count} | "
            f"{item['different_assignment_count']}/{window_count} | "
            f"{item['different_assignment_same_routed_wirelength_count']} |"
        )
    lines.extend(
        [
            "",
            "## Full-Retention Sensitivity",
            "",
            "This sensitivity analysis keeps only windows where both QAOA and the",
            "comparator retain all four selected cells after CTS legalization.",
            "",
            "| Comparator | Retained pairs | Mean QAOA-minus-comparator wirelength (µm) | 95% interval |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparator in METHODS[1:]:
        item = retained[comparator]["routed_wirelength"]
        ci = item["hierarchical_bootstrap_95_ci"]
        lines.append(
            f"| {LABELS[comparator]} | {retained[comparator]['pair_count']} | "
            f"{item['mean_difference']:+.6g} | [{ci[0]:+.6g}, {ci[1]:+.6g}] |"
        )
    lines.extend(
        [
            "",
            "## Secondary Outcomes",
            "",
            "Mean QAOA-minus-comparator differences are exploratory. A positive value",
            "favors QAOA only for WNS/TNS metrics; a negative value favors QAOA for",
            "wirelength, vias, and congestion demand.",
            "",
            "| Comparator | Vias | Congestion demand | Setup WNS (ns) | Setup TNS (ns) | Hold WNS (ns) | Hold TNS (ns) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparator in METHODS[1:]:
        values = [paired[comparator][metric]["mean_difference"] for metric in (
            "via_count",
            "congestion_demand",
            "setup_wns_ns",
            "setup_tns_ns",
            "hold_wns_ns",
            "hold_tns_ns",
        )]
        lines.append(
            f"| {LABELS[comparator]} | " + " | ".join(f"{value:+.6g}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Provenance and Scope",
            "",
            f"- Classical execution commit: `{summary['execution_commit']}`.",
            f"- QAOA execution commit: `{summary['qaoa_execution_commit']}`.",
            f"- SIF SHA-256: `{summary['container_sha256']}`.",
            f"- Comparator run-level SHA-256: `{summary['comparator_run_level_sha256']}`.",
            f"- QAOA routing-summary SHA-256: `{summary['qaoa_routing_summary_sha256']}`.",
        ]
    )
    if "slurm" in summary:
        slurm = summary["slurm"]
        lines.extend(
            [
                f"- Slurm array job IDs: `{', '.join(slurm['array_job_ids'])}`.",
                f"- Slurm task coverage: {slurm['task_id_min']}--{slurm['task_id_max']} "
                f"({slurm['task_id_count']} unique tasks).",
                f"- Slurm resources per task: partition `{slurm['partition']}`, "
                f"{slurm['cpus_per_task']} CPUs, {slurm['memory_per_node_mb']} MiB.",
            ]
        )
    lines.extend(
        [
            "",
            "These results compare deterministic selected assignments under a controlled",
            "CTS/global-route flow. They do not establish runtime advantage, detailed-route",
            f"closure, hardware practicality, or generalization beyond the {design_count} tested designs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--qaoa-selected-readout", type=Path, required=True)
    parser.add_argument("--qaoa-routing-results", type=Path, required=True)
    parser.add_argument("--qaoa-routing-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    if sha256(args.qaoa_selected_readout) != plan["qaoa_selected_readout_sha256"]:
        raise ValueError("QAOA selected-readout SHA-256 mismatch")
    if sha256(args.qaoa_routing_summary) != plan["qaoa_routing_summary_sha256"]:
        raise ValueError("QAOA routing-summary SHA-256 mismatch")
    qaoa_selected_rows = load_csv(args.qaoa_selected_readout)
    window_count = int(plan.get("window_count", len(qaoa_selected_rows)))
    design_count = int(
        plan.get("design_count", len({row["design"] for row in qaoa_selected_rows}))
    )
    if len(qaoa_selected_rows) != window_count or len(
        {row["window_id"] for row in qaoa_selected_rows}
    ) != window_count:
        raise ValueError(f"expected {window_count} unique QAOA selected-readout rows")
    qaoa_assignments = {
        row["window_id"]: tuple(int(site) for site in json.loads(row["solution_assignment"]))
        for row in qaoa_selected_rows
    }
    qaoa_summary = json.loads(args.qaoa_routing_summary.read_text())
    qaoa_rows_raw = load_csv(args.qaoa_routing_results)
    if len(qaoa_rows_raw) != window_count or len(
        {row["window_id"] for row in qaoa_rows_raw}
    ) != window_count:
        raise ValueError(f"expected {window_count} unique QAOA routing rows")

    result_paths = glob.glob(str(args.artifact_root / "*" / "case_result.json"))
    results = [json.loads(Path(path).read_text()) for path in result_paths]
    expected_classical_cases = len(plan["cases"])
    if len(results) != expected_classical_cases or any(
        result["status"] != "complete" for result in results
    ):
        raise ValueError(
            f"expected {expected_classical_cases} complete classical case manifests"
        )
    expected_ids = {str(case["case_id"]) for case in plan["cases"]}
    result_ids = {str(result["case"]["case_id"]) for result in results}
    if expected_ids != result_ids:
        raise ValueError("classical case manifests do not exactly match the frozen plan")
    commits = {result["execution_commit"] for result in results}
    containers = {result["container_sha256"] for result in results}
    if len(commits) != 1 or len(containers) != 1:
        raise ValueError("inconsistent classical execution provenance")
    container_sha256 = next(iter(containers))
    if container_sha256 != qaoa_summary["container_sha256"]:
        raise ValueError("classical and QAOA container hashes differ")

    slurm_records = [result.get("slurm") for result in results]
    if any(slurm_records) and not all(slurm_records):
        raise ValueError("incomplete Slurm provenance across classical cases")
    slurm_summary: dict[str, object] | None = None
    if all(slurm_records):
        task_ids = {int(record["array_task_id"]) for record in slurm_records}
        if task_ids != set(range(expected_classical_cases)):
            raise ValueError("Slurm task identifiers do not exactly cover the frozen plan")
        resources = {
            (
                str(record["partition"]),
                str(record["cpus_per_task"]),
                str(record["memory_per_node_mb"]),
            )
            for record in slurm_records
        }
        if len(resources) != 1:
            raise ValueError("inconsistent Slurm resources across classical cases")
        partition, cpus, memory_mb = next(iter(resources))
        slurm_summary = {
            "array_job_ids": sorted(
                {str(record["array_job_id"]) for record in slurm_records}
            ),
            "task_id_min": min(task_ids),
            "task_id_max": max(task_ids),
            "task_id_count": len(task_ids),
            "partition": partition,
            "cpus_per_task": int(cpus),
            "memory_per_node_mb": int(memory_mb),
        }

    baselines = qaoa_summary["baselines"]
    rows: list[dict[str, object]] = []
    for raw in qaoa_rows_raw:
        row: dict[str, object] = {key: value for key, value in raw.items()}
        row["method"] = "qaoa"
        for metric in DIRECTIONS:
            row[metric] = float(raw[metric])
            row[f"{metric}_delta"] = float(raw[f"{metric}_delta"])
        row["total_overflow"] = int(raw["total_overflow"])
        row["retained_cell_count"] = int(raw["retained_cell_count"])
        row["all_cells_retained"] = int(raw["retained_cell_count"]) == 4
        row["local_hpwl_improvement_dbu"] = int(raw["local_hpwl_improvement_dbu"])
        rows.append(row)

    for result in results:
        case = result["case"]
        baseline = baselines[case["design"]]
        row = {
            "window_id": case["window_id"],
            "design": case["design"],
            "selection_stratum": case["selection_stratum"],
            "method": case["method"],
            "comparator_seed": case["comparator_seed"],
            "local_hpwl_improvement_dbu": case["local_hpwl_improvement_dbu"],
            "total_overflow": result["congestion_total"]["total_overflow"],
            "retained_cell_count": result["cell_retention"]["retained_count"],
            "all_cells_retained": result["cell_retention"]["all_retained"],
        }
        for name, (native, _direction) in METRICS.items():
            value = float(result["global_route_metrics"][native])
            row[name] = value
            row[f"{name}_delta"] = value - float(baseline[name])
        demand = float(result["congestion_total"]["demand"])
        row["congestion_demand"] = demand
        row["congestion_demand_delta"] = demand - float(baseline["congestion_demand"])
        rows.append(row)
    rows.sort(key=lambda row: (str(row["window_id"]), METHODS.index(str(row["method"]))))

    method_rows = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    for method, group in method_rows.items():
        if len(group) != window_count or len(
            {row["window_id"] for row in group}
        ) != window_count:
            raise ValueError(f"expected {window_count} unique rows for {method}")
    comparator_assignments: dict[str, dict[str, tuple[int, ...]]] = {
        method: {} for method in METHODS[1:]
    }
    for case in plan["cases"]:
        comparator_assignments[str(case["method"])][str(case["window_id"])] = tuple(
            int(site) for site in case["solution_assignment"]
        )
    identity: dict[str, object] = {}
    qaoa_by_window = {str(row["window_id"]): row for row in method_rows["qaoa"]}
    for comparator in METHODS[1:]:
        comparator_by_window = {
            str(row["window_id"]): row for row in method_rows[comparator]
        }
        identical = {
            window_id
            for window_id in qaoa_assignments
            if qaoa_assignments[window_id] == comparator_assignments[comparator][window_id]
        }
        different_same_routed = [
            window_id
            for window_id in qaoa_assignments
            if window_id not in identical
            and float(qaoa_by_window[window_id]["routed_wirelength"])
            == float(comparator_by_window[window_id]["routed_wirelength"])
        ]
        identity[comparator] = {
            "identical_assignment_count": len(identical),
            "different_assignment_count": window_count - len(identical),
            "different_assignment_same_routed_wirelength_count": len(different_same_routed),
            "different_assignment_same_routed_wirelength_windows": different_same_routed,
        }
    method_summaries: dict[str, object] = {}
    for method, group in method_rows.items():
        method_summaries[method] = {
            "zero_overflow_count": sum(int(row["total_overflow"]) == 0 for row in group),
            "retained_cell_count": sum(int(row["retained_cell_count"]) for row in group),
            "all_cells_retained_count": sum(int(row["retained_cell_count"]) == 4 for row in group),
            "routed_wirelength_sum_delta": sum(float(row["routed_wirelength_delta"]) for row in group),
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
                int(plan["bootstrap_seed"]) + 100 * comparator_index + metric_index,
                int(plan["bootstrap_replicates"]),
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
                int(plan["bootstrap_seed"]) + 1000 + comparator_index,
                int(plan["bootstrap_replicates"]),
            ),
        }

    classical_rows = [row for row in rows if row["method"] != "qaoa"]
    summary = {
        "schema_version": "qeda-downstream-classical-comparison-summary-v1",
        "execution_commit": next(iter(commits)),
        "qaoa_execution_commit": qaoa_summary["execution_commit"],
        "container_sha256": container_sha256,
        "comparator_run_level_sha256": plan["comparator_run_level_sha256"],
        "qaoa_selected_readout_sha256": plan["qaoa_selected_readout_sha256"],
        "qaoa_routing_summary_sha256": plan["qaoa_routing_summary_sha256"],
        "window_count": window_count,
        "design_count": design_count,
        "bootstrap": {
            "seed": plan["bootstrap_seed"],
            "replicates": plan["bootstrap_replicates"],
            "design_clusters": design_count,
            "interpretation": (
                "hierarchical bootstrap over designs and windows; population scope "
                "is limited to the tested design corpus"
            ),
        },
        "technical": {
            "attempted_count": expected_classical_cases,
            "completed_count": len(classical_rows),
            "zero_overflow_count": sum(int(row["total_overflow"]) == 0 for row in classical_rows),
            "retained_cell_count": sum(int(row["retained_cell_count"]) for row in classical_rows),
            "all_cells_retained_count": sum(int(row["retained_cell_count"]) == 4 for row in classical_rows),
        },
        "method_summaries": method_summaries,
        "qaoa_pairwise": paired,
        "qaoa_pairwise_all_cells_retained": retained_paired,
        "assignment_identity_with_qaoa": identity,
    }
    if slurm_summary is not None:
        summary["slurm"] = slurm_summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison_results.csv").open("w", newline="") as handle:
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(str(field))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULTS_REPORT.md").write_text(render_report(summary))
    print(
        json.dumps(
            {
                "classical_cases": len(classical_rows),
                "zero_overflow": summary["technical"]["zero_overflow_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
