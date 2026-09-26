"""Prepare frozen exact, greedy, and annealing cases for downstream routing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.qaoa_placement.eda_bridge.prepare_downstream_routing import (
    CONTAINER_IMAGE,
    expected_cells,
    render_apply_tcl,
    render_inspect_tcl,
    sha256,
)
from experiments.qaoa_placement.eda_bridge.prepare_timing_impact import assignment_cost


METHODS = ("exact", "greedy", "simulated_annealing")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_comparator_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        method = str(row["method"])
        if method in METHODS:
            groups.setdefault((str(row["window_id"]), method), []).append(row)
    selected = [
        min(
            group,
            key=lambda row: (
                int(float(row["best_hpwl_dbu"])),
                str(row["seed"]),
                str(row["solution_assignment"]),
            ),
        )
        for _key, group in sorted(groups.items())
    ]
    expected_windows = {
        str(row["window_id"]) for row in rows if str(row["method"]) in METHODS
    }
    for method in METHODS:
        method_rows = [row for row in selected if row["method"] == method]
        method_windows = {str(row["window_id"]) for row in method_rows}
        if method_windows != expected_windows:
            raise ValueError(
                f"selected {method} windows do not match the comparator corpus"
            )
        if len(method_rows) != len(expected_windows):
            raise ValueError(f"selected {method} rows do not have unique windows")
    return selected


def prepare(
    manifest_path: Path,
    run_level_path: Path,
    qaoa_selected_readout: Path,
    qaoa_routing_summary: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {output_dir}")
    manifest = json.loads(manifest_path.read_text())
    windows = {str(window["window_id"]): window for window in manifest["windows"]}
    rows = select_comparator_rows(load_rows(run_level_path))
    selected_windows = {str(row["window_id"]) for row in rows}
    if selected_windows != set(windows):
        raise ValueError("comparator windows do not exactly match the benchmark manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["design"]),
            str(item["window_id"]),
            METHODS.index(str(item["method"])),
        ),
    ):
        method = str(row["method"])
        window_id = str(row["window_id"])
        window = windows[window_id]
        assignment = [int(site) for site in json.loads(row["solution_assignment"])]
        selected_hpwl = assignment_cost(window, assignment)
        reported_hpwl = int(float(row["best_hpwl_dbu"]))
        if selected_hpwl != reported_hpwl:
            raise ValueError(
                f"{method} {window_id} assignment cost {selected_hpwl} != {reported_hpwl}"
            )
        case_id = f"{method}__{window_id}"
        apply_path = output_dir / f"apply_{case_id}.tcl"
        inspect_path = output_dir / f"inspect_{case_id}.tcl"
        expected = expected_cells(window, assignment)
        apply_path.write_text(render_apply_tcl(window, assignment))
        inspect_path.write_text(render_inspect_tcl(expected))
        cases.append(
            {
                "case_id": case_id,
                "design": str(row["design"]),
                "kind": "classical_comparator",
                "method": method,
                "comparator_seed": str(row["seed"]),
                "variant": f"qeda_classical_routing_{method}_{window_id}",
                "apply_tcl": str(apply_path),
                "inspect_tcl": str(inspect_path),
                "expected_cells": expected,
                "window_id": window_id,
                "selection_stratum": str(row["selection_stratum"]),
                "objective_evaluation_budget": int(row["objective_evaluation_budget"]),
                "objective_evaluations": int(row["objective_evaluations"]),
                "initial_hpwl_dbu": int(float(row["initial_hpwl_dbu"])),
                "selected_hpwl_dbu": selected_hpwl,
                "local_hpwl_improvement_dbu": int(float(row["initial_hpwl_dbu"]))
                - selected_hpwl,
                "solution_assignment": assignment,
            }
        )
    expected_case_count = len(METHODS) * len(windows)
    if len(cases) != expected_case_count:
        raise ValueError(f"expected {expected_case_count} cases, got {len(cases)}")
    plan = {
        "schema_version": "qeda-downstream-classical-comparison-v1",
        "container_image": CONTAINER_IMAGE,
        "orfs_commit": "be0dca0b1fd41df54792b3012350cd52bccd99bb",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "comparator_run_level": str(run_level_path),
        "comparator_run_level_sha256": sha256(run_level_path),
        "qaoa_selected_readout": str(qaoa_selected_readout),
        "qaoa_selected_readout_sha256": sha256(qaoa_selected_readout),
        "qaoa_routing_summary": str(qaoa_routing_summary),
        "qaoa_routing_summary_sha256": sha256(qaoa_routing_summary),
        "selection_rule": (
            "minimum best_hpwl_dbu, then seed string, then serialized assignment, "
            "within each frozen window/method group"
        ),
        "methods": list(METHODS),
        "num_cores": 2,
        "grt_seed": 20260827,
        "bootstrap_seed": 20260827,
        "bootstrap_replicates": 10000,
        "design_count": len({str(case["design"]) for case in cases}),
        "window_count": len(windows),
        "cases": cases,
    }
    plan_path = output_dir / "routing_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-level", type=Path, required=True)
    parser.add_argument("--qaoa-selected-readout", type=Path, required=True)
    parser.add_argument("--qaoa-routing-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path = prepare(
        args.manifest,
        args.run_level,
        args.qaoa_selected_readout,
        args.qaoa_routing_summary,
        args.output_dir,
    )
    plan = json.loads(plan_path.read_text())
    print(json.dumps({"cases": len(plan["cases"]), "plan": str(plan_path)}))


if __name__ == "__main__":
    main()
