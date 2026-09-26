"""Prepare compact OpenROAD Tcl bundles for benchmark winner legality checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


METHODS = (
    "exact",
    "greedy",
    "simulated_annealing",
    "token_qaoa",
    "selected_token_qaoa",
    "finite_shot_token_qaoa",
)
ORIENTATION = {"N": "R0", "S": "R180", "FN": "MY", "FS": "MX"}


def tcl_word(value: object) -> str:
    # Keep the single DEF escape used by OpenDB names. Tcl braces protect '['
    # and '$' without requiring another layer of backslashes.
    text = str(value).replace("}", "\\}")
    return "{" + text + "}"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def selected_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        if row["method"] in METHODS:
            groups.setdefault((row["window_id"], row["method"]), []).append(row)
    return [
        min(group, key=lambda row: (int(float(row["best_hpwl_dbu"])), str(row["seed"])))
        for _key, group in sorted(groups.items())
    ]


def render_tcl(design: str, windows: Mapping[str, Mapping[str, object]], rows: Sequence[Mapping[str, str]]) -> str:
    lines = [
        "if {![info exists ::env(QEDA_INPUT_ODB)]} { error \"QEDA_INPUT_ODB is required\" }",
        "read_db $::env(QEDA_INPUT_ODB)",
        "set block [ord::get_db_block]",
        "set validated 0",
    ]
    for row in rows:
        if row["design"] != design:
            continue
        window = windows[row["window_id"]]
        sites = {int(site["site_id"]): site for site in window["candidate_sites"]}
        cells = [str(cell["cell_id"]) for cell in window["movable_cells"]]
        assignment = [int(site) for site in json.loads(row["solution_assignment"])]
        case_id = f"{row['window_id']}__{row['method']}__seed_{row['seed'] or 'none'}"
        lines.append(f"# {case_id}")
        for cell, site_id in zip(cells, assignment):
            site = sites[site_id]
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
        lines.append("check_placement -verbose")
        lines.append(f"puts \"QEDA_VALIDATED {case_id}\"")
        for cell_index, cell in enumerate(cells):
            site = sites[cell_index]
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
        lines.append("incr validated")
    lines.extend(["check_placement -verbose", "puts \"QEDA_VALIDATION_COUNT $validated\""])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-level", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    windows = {window["window_id"]: window for window in payload["windows"]}
    rows = selected_rows(load_rows(args.run_level))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan: list[dict[str, object]] = []
    for design in sorted({row["design"] for row in rows}):
        design_rows = [row for row in rows if row["design"] == design]
        tcl_path = args.output_dir / f"validate_{design}.tcl"
        tcl_path.write_text(render_tcl(design, windows, design_rows))
        plan.append(
            {
                "design": design,
                "tcl": str(tcl_path),
                "case_count": len(design_rows),
                "methods": sorted({str(row["method"]) for row in design_rows}),
            }
        )
    (args.output_dir / "validation_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"designs": len(plan), "cases": len(rows)}))


if __name__ == "__main__":
    main()
