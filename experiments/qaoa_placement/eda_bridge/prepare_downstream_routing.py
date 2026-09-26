"""Prepare fixed baseline and QAOA cases for the downstream ORFS routing gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.qaoa_placement.eda_bridge.prepare_openroad_validation import (
    ORIENTATION,
    tcl_word,
)


CONTAINER_IMAGE = (
    "openroad/orfs@sha256:"
    "d995618be9f2bcdfa5538b885123463070dfbf178bea1818716d4652fe0fa380"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_apply_tcl(
    window: Mapping[str, object] | None, assignment: Sequence[int] | None
) -> str:
    lines = [
        "foreach variable {QEDA_INPUT_ODB QEDA_OUTPUT_ODB} {",
        "  if {![info exists ::env($variable)]} { error \"$variable is required\" }",
        "}",
        "read_db $::env(QEDA_INPUT_ODB)",
        "set block [ord::get_db_block]",
    ]
    if window is not None and assignment is not None:
        sites = {int(site["site_id"]): site for site in window["candidate_sites"]}  # type: ignore[index]
        cells = [str(cell["cell_id"]) for cell in window["movable_cells"]]  # type: ignore[index]
        for cell, site_id in zip(cells, assignment):
            site = sites[int(site_id)]
            orient = ORIENTATION[str(site["orient"])]
            lines.extend(
                [
                    f"set inst [$block findInst {tcl_word(cell)}]",
                    f"$inst setOrient {orient}",
                    f"$inst setLocation {int(site['x'])} {int(site['y'])}",
                ]
            )
    lines.extend(
        [
            "check_placement -verbose",
            "write_db $::env(QEDA_OUTPUT_ODB)",
            "puts \"QEDA_DOWNSTREAM_INPUT_READY\"",
        ]
    )
    return "\n".join(lines) + "\n"


def render_inspect_tcl(expected_cells: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "foreach variable {QEDA_INPUT_ODB QEDA_OUTPUT_CSV} {",
        "  if {![info exists ::env($variable)]} { error \"$variable is required\" }",
        "}",
        "read_db $::env(QEDA_INPUT_ODB)",
        "set block [ord::get_db_block]",
        "set output [open $::env(QEDA_OUTPUT_CSV) w]",
        "puts $output {cell_id,expected_x,expected_y,expected_orient,final_x,final_y,final_orient}",
    ]
    for item in expected_cells:
        lines.extend(
            [
                f"set inst [$block findInst {tcl_word(item['cell_id'])}]",
                "set location [$inst getLocation]",
                f"puts $output [join [list {tcl_word(item['cell_id'])} {int(item['x'])} {int(item['y'])} {ORIENTATION[str(item['orient'])]} [lindex $location 0] [lindex $location 1] [$inst getOrient]] ,]",
            ]
        )
    lines.extend(["close $output", "puts \"QEDA_DOWNSTREAM_INSPECTED\""])
    return "\n".join(lines) + "\n"


def expected_cells(window: Mapping[str, object], assignment: Sequence[int]) -> list[dict[str, object]]:
    sites = {int(site["site_id"]): site for site in window["candidate_sites"]}  # type: ignore[index]
    return [
        {
            "cell_id": str(cell["cell_id"]),
            "site_id": int(site_id),
            "x": int(sites[int(site_id)]["x"]),
            "y": int(sites[int(site_id)]["y"]),
            "orient": str(sites[int(site_id)]["orient"]),
        }
        for cell, site_id in zip(window["movable_cells"], assignment)  # type: ignore[index]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    windows = {str(window["window_id"]): window for window in manifest["windows"]}
    with args.selected_readout.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_window_ids = set(windows)
    selected_window_ids = {row["window_id"] for row in rows}
    if len(rows) != len(expected_window_ids) or selected_window_ids != expected_window_ids:
        raise ValueError("selected readout must contain exactly one row per manifest window")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for design in sorted({row["design"] for row in rows}):
        for suffix in ("a", "b"):
            case_id = f"{design}__baseline_{suffix}"
            apply_path = args.output_dir / f"apply_{case_id}.tcl"
            inspect_path = args.output_dir / f"inspect_{case_id}.tcl"
            apply_path.write_text(render_apply_tcl(None, None))
            inspect_path.write_text(render_inspect_tcl([]))
            cases.append(
                {
                    "case_id": case_id,
                    "design": design,
                    "kind": "baseline",
                    "baseline_replica": suffix,
                    "variant": f"qeda_routing_{case_id}",
                    "apply_tcl": str(apply_path),
                    "inspect_tcl": str(inspect_path),
                    "expected_cells": [],
                    "window_id": "",
                    "local_hpwl_improvement_dbu": 0,
                }
            )
        for row in sorted((item for item in rows if item["design"] == design), key=lambda item: item["window_id"]):
            window = windows[row["window_id"]]
            assignment = [int(site) for site in json.loads(row["solution_assignment"])]
            case_id = row["window_id"]
            apply_path = args.output_dir / f"apply_{case_id}.tcl"
            inspect_path = args.output_dir / f"inspect_{case_id}.tcl"
            expected = expected_cells(window, assignment)
            apply_path.write_text(render_apply_tcl(window, assignment))
            inspect_path.write_text(render_inspect_tcl(expected))
            cases.append(
                {
                    "case_id": case_id,
                    "design": design,
                    "kind": "assignment",
                    "baseline_replica": "",
                    "variant": f"qeda_routing_{case_id}",
                    "apply_tcl": str(apply_path),
                    "inspect_tcl": str(inspect_path),
                    "expected_cells": expected,
                    "window_id": row["window_id"],
                    "selection_stratum": row["selection_stratum"],
                    "seed": int(row["seed"]),
                    "initial_hpwl_dbu": int(float(row["initial_hpwl_dbu"])),
                    "selected_hpwl_dbu": int(float(row["best_hpwl_dbu"])),
                    "local_hpwl_improvement_dbu": int(
                        float(row["initial_hpwl_dbu"]) - float(row["best_hpwl_dbu"])
                    ),
                }
            )
    plan = {
        "schema_version": "qeda-downstream-routing-v1",
        "container_image": CONTAINER_IMAGE,
        "orfs_commit": "be0dca0b1fd41df54792b3012350cd52bccd99bb",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "selected_readout": str(args.selected_readout),
        "selected_readout_sha256": sha256(args.selected_readout),
        "num_cores": 2,
        "grt_seed": 20260827,
        "design_count": len({row["design"] for row in rows}),
        "window_count": len(rows),
        "cases": cases,
    }
    plan_path = args.output_dir / "routing_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(cases), "plan": str(plan_path)}))


if __name__ == "__main__":
    main()
