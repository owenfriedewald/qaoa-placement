"""Prepare reproducible OpenROAD timing-impact cases for fixed assignments."""

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
OPENROAD_BINARY = "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
LIBERTY = (
    "/OpenROAD-flow-scripts/flow/platforms/nangate45/lib/"
    "NangateOpenCellLibrary_typical.lib"
)
CSV_HEADER = (
    "design,window_id,case_kind,selection_stratum,seed,initial_hpwl_dbu,"
    "selected_hpwl_dbu,hpwl_delta_dbu,setup_wns_ns,setup_tns_ns,"
    "hold_wns_ns,hold_tns_ns"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selected_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36:
        raise ValueError(f"expected 36 selected rows, found {len(rows)}")
    window_ids = [row["window_id"] for row in rows]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("selected readout contains duplicate window IDs")
    return rows


def assignment_cost(window: Mapping[str, object], assignment: Sequence[int]) -> int:
    target = tuple(int(site) for site in assignment)
    for encoded_assignment, cost in window["cost_landscape"]:  # type: ignore[index]
        if tuple(int(site) for site in encoded_assignment) == target:
            return int(cost)
    raise ValueError(f"assignment is absent from {window['window_id']} landscape")


def site_commands(
    window: Mapping[str, object], assignment: Sequence[int]
) -> list[str]:
    sites = {
        int(site["site_id"]): site
        for site in window["candidate_sites"]  # type: ignore[index]
    }
    cells = [
        str(cell["cell_id"])
        for cell in window["movable_cells"]  # type: ignore[index]
    ]
    if len(cells) != len(assignment):
        raise ValueError(f"assignment length mismatch for {window['window_id']}")
    lines: list[str] = []
    for cell, site_id in zip(cells, assignment):
        site = sites[int(site_id)]
        orient = ORIENTATION.get(str(site["orient"]))
        if orient is None:
            raise ValueError(f"unsupported orientation {site['orient']}")
        lines.extend(
            [
                f"set inst [$block findInst {tcl_word(cell)}]",
                f"$inst setOrient {orient}",
                f"$inst setLocation {int(site['x'])} {int(site['y'])}",
            ]
        )
    return lines


def initial_assignment(window: Mapping[str, object]) -> list[int]:
    initial = window["initial_assignment"]
    return [
        int(initial[str(cell["cell_id"])])  # type: ignore[index]
        for cell in window["movable_cells"]  # type: ignore[index]
    ]


def render_tcl(
    design: str,
    windows: Mapping[str, Mapping[str, object]],
    rows: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "foreach variable {QEDA_INPUT_ODB QEDA_SDC QEDA_SET_RC QEDA_OUTPUT_CSV} {",
        "  if {![info exists ::env($variable)]} { error \"$variable is required\" }",
        "}",
        f"read_liberty {tcl_word(LIBERTY)}",
        "read_db $::env(QEDA_INPUT_ODB)",
        "read_sdc $::env(QEDA_SDC)",
        "source $::env(QEDA_SET_RC)",
        "set block [ord::get_db_block]",
        "set output [open $::env(QEDA_OUTPUT_CSV) w]",
        f"puts $output {tcl_word(CSV_HEADER)}",
        "proc qeda_measure {output design window_id kind stratum seed initial_hpwl selected_hpwl} {",
        "  estimate_parasitics -placement",
        "  set setup_wns [sta::time_sta_ui [sta::worst_slack_cmd max]]",
        "  set setup_tns [sta::time_sta_ui [sta::total_negative_slack_cmd max]]",
        "  set hold_wns [sta::time_sta_ui [sta::worst_slack_cmd min]]",
        "  set hold_tns [sta::time_sta_ui [sta::total_negative_slack_cmd min]]",
        "  set hpwl_delta [expr {$selected_hpwl - $initial_hpwl}]",
        "  puts $output [join [list $design $window_id $kind $stratum $seed $initial_hpwl $selected_hpwl $hpwl_delta $setup_wns $setup_tns $hold_wns $hold_tns] ,]",
        "  flush $output",
        "}",
        f"qeda_measure $output {design} __baseline_before__ baseline __all__ none 0 0",
    ]
    for row in sorted(rows, key=lambda item: item["window_id"]):
        window = windows[row["window_id"]]
        assignment = [int(site) for site in json.loads(row["solution_assignment"])]
        selected_hpwl = assignment_cost(window, assignment)
        reported_hpwl = int(float(row["best_hpwl_dbu"]))
        if selected_hpwl != reported_hpwl:
            raise ValueError(
                f"{row['window_id']} assignment cost {selected_hpwl} != reported {reported_hpwl}"
            )
        lines.append(f"# Apply {row['window_id']} without timing-based selection.")
        lines.extend(site_commands(window, assignment))
        lines.append("check_placement -verbose")
        lines.append(
            "qeda_measure $output "
            f"{design} {row['window_id']} assignment {row['selection_stratum']} "
            f"{row['seed']} {int(float(row['initial_hpwl_dbu']))} {selected_hpwl}"
        )
        lines.append(f"# Restore {row['window_id']} exactly.")
        lines.extend(site_commands(window, initial_assignment(window)))
    lines.extend(
        [
            "check_placement -verbose",
            f"qeda_measure $output {design} __baseline_after__ baseline __all__ none 0 0",
            "close $output",
            f"puts \"QEDA_TIMING_CASES {design} {len(rows)}\"",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    windows = {window["window_id"]: window for window in manifest["windows"]}
    rows = load_selected_rows(args.selected_readout)
    missing = sorted(set(row["window_id"] for row in rows) - set(windows))
    if missing:
        raise ValueError(f"selected windows absent from manifest: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan: dict[str, object] = {
        "schema_version": "qeda-timing-impact-v1",
        "container_image": CONTAINER_IMAGE,
        "openroad_binary": OPENROAD_BINARY,
        "input_manifest": str(args.manifest),
        "input_manifest_sha256": sha256(args.manifest),
        "selected_readout": str(args.selected_readout),
        "selected_readout_sha256": sha256(args.selected_readout),
        "roundtrip_tolerance_ns": 1e-12,
        "designs": [],
    }
    for design in sorted({row["design"] for row in rows}):
        design_rows = [row for row in rows if row["design"] == design]
        if len(design_rows) != 12:
            raise ValueError(f"expected 12 {design} rows, found {len(design_rows)}")
        tcl_path = args.output_dir / f"evaluate_timing_{design}.tcl"
        raw_path = args.output_dir / f"raw_timing_{design}.csv"
        tcl_path.write_text(render_tcl(design, windows, design_rows))
        plan["designs"].append(  # type: ignore[union-attr]
            {
                "design": design,
                "case_count": len(design_rows),
                "tcl": str(tcl_path),
                "raw_csv": str(raw_path),
                "input_odb": (
                    f"/orfs-work/flow/results/nangate45/{design}/base/"
                    "qeda_3_4_legalize_only.odb"
                ),
                "sdc": (
                    f"/orfs-work/flow/results/nangate45/{design}/base/"
                    "2_floorplan.sdc"
                ),
                "set_rc": "/orfs-work/flow/platforms/nangate45/setRC.tcl",
            }
        )
    plan_path = args.output_dir / "timing_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"designs": 3, "cases": len(rows), "plan": str(plan_path)}))


if __name__ == "__main__":
    main()
