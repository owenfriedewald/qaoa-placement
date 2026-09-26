"""Extract exact-solvable placement windows from placed LEF/DEF designs.

The bridge intentionally has no OpenROAD Python dependency. OpenROAD exports a
placed ODB to DEF, then this module parses the small LEF/DEF subset needed for
deterministic window extraction, exact local HPWL evaluation, and reinsertion.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence


SCHEMA_VERSION = "qeda-real-placement-windows-v1"


@dataclass(frozen=True)
class Macro:
    name: str
    width: int
    height: int
    pins: Mapping[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class Component:
    name: str
    master: str
    x: int
    y: int
    orient: str
    status: str


@dataclass(frozen=True)
class Row:
    name: str
    site: str
    x: int
    y: int
    orient: str
    count_x: int
    count_y: int
    step_x: int
    step_y: int


@dataclass(frozen=True)
class Pin:
    name: str
    x: int
    y: int


@dataclass(frozen=True)
class Net:
    name: str
    connections: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DefDesign:
    name: str
    dbu_per_micron: int
    diearea: tuple[int, int, int, int]
    rows: tuple[Row, ...]
    components: Mapping[str, Component]
    pins: Mapping[str, Pin]
    nets: tuple[Net, ...]
    source_text: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_dbu(value: str, dbu_per_micron: int) -> int:
    return int(round(float(value) * dbu_per_micron))


def parse_lef(path: Path, dbu_per_micron: int) -> dict[str, Macro]:
    """Parse macro sizes and representative pin centers from a cell LEF."""

    macros: dict[str, Macro] = {}
    macro_name: str | None = None
    macro_size: tuple[int, int] | None = None
    pin_name: str | None = None
    pin_rects: list[tuple[float, float, float, float]] = []
    pin_centers: dict[str, tuple[int, int]] = {}

    def finish_pin() -> None:
        nonlocal pin_name, pin_rects
        if pin_name is None:
            return
        if pin_rects:
            areas = [max((x2 - x1) * (y2 - y1), 1e-18) for x1, y1, x2, y2 in pin_rects]
            area_sum = sum(areas)
            x = sum(area * (x1 + x2) / 2 for area, (x1, y1, x2, y2) in zip(areas, pin_rects)) / area_sum
            y = sum(area * (y1 + y2) / 2 for area, (x1, y1, x2, y2) in zip(areas, pin_rects)) / area_sum
            pin_centers[pin_name] = (_round_dbu(str(x), dbu_per_micron), _round_dbu(str(y), dbu_per_micron))
        pin_name = None
        pin_rects = []

    def finish_macro() -> None:
        nonlocal macro_name, macro_size, pin_centers
        finish_pin()
        if macro_name is not None and macro_size is not None:
            macros[macro_name] = Macro(macro_name, macro_size[0], macro_size[1], dict(pin_centers))
        macro_name = None
        macro_size = None
        pin_centers = {}

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        match = re.match(r"MACRO\s+(\S+)", line)
        if match:
            finish_macro()
            macro_name = match.group(1)
            continue
        if macro_name is None:
            continue
        match = re.match(r"SIZE\s+([\d.eE+-]+)\s+BY\s+([\d.eE+-]+)\s*;", line)
        if match:
            macro_size = (_round_dbu(match.group(1), dbu_per_micron), _round_dbu(match.group(2), dbu_per_micron))
            continue
        match = re.match(r"PIN\s+(\S+)", line)
        if match:
            finish_pin()
            pin_name = match.group(1)
            continue
        match = re.match(
            r"RECT\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*;",
            line,
        )
        if match and pin_name is not None:
            pin_rects.append(tuple(float(match.group(index)) for index in range(1, 5)))
            continue
        if line.startswith("END "):
            ended = line.split(maxsplit=1)[1]
            if pin_name is not None and ended == pin_name:
                finish_pin()
            elif ended == macro_name:
                finish_macro()
    finish_macro()
    return macros


def _section_records(lines: Sequence[str], section: str) -> list[str]:
    records: list[str] = []
    active = False
    current: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith(f"{section} "):
            active = True
            continue
        if stripped == f"END {section}":
            if current:
                records.append(" ".join(current))
            break
        if not active:
            continue
        if stripped.startswith("-"):
            if current:
                records.append(" ".join(current))
            current = [stripped]
        elif current:
            current.append(stripped)
        if current and ";" in stripped:
            records.append(" ".join(current))
            current = []
    return records


def parse_def(path: Path) -> DefDesign:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    name_match = re.search(r"^DESIGN\s+(\S+)\s*;", text, re.MULTILINE)
    units_match = re.search(r"^UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;", text, re.MULTILINE)
    die_match = re.search(
        r"^DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;",
        text,
        re.MULTILINE,
    )
    if not name_match or not units_match or not die_match:
        raise ValueError(f"DEF header is incomplete: {path}")

    rows: list[Row] = []
    row_pattern = re.compile(
        r"^ROW\s+(\S+)\s+(\S+)\s+(-?\d+)\s+(-?\d+)\s+(\S+)\s+DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(-?\d+)\s+(-?\d+)\s*;"
    )
    for line in lines:
        match = row_pattern.match(line.strip())
        if match:
            rows.append(
                Row(
                    name=match.group(1),
                    site=match.group(2),
                    x=int(match.group(3)),
                    y=int(match.group(4)),
                    orient=match.group(5),
                    count_x=int(match.group(6)),
                    count_y=int(match.group(7)),
                    step_x=int(match.group(8)),
                    step_y=int(match.group(9)),
                )
            )

    components: dict[str, Component] = {}
    for record in _section_records(lines, "COMPONENTS"):
        head = re.match(r"-\s+(\S+)\s+(\S+)", record)
        placement = re.search(
            r"\+\s+(PLACED|FIXED|COVER)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)", record
        )
        if head and placement:
            component = Component(
                name=head.group(1), master=head.group(2), x=int(placement.group(2)), y=int(placement.group(3)),
                orient=placement.group(4), status=placement.group(1),
            )
            components[component.name] = component

    pins: dict[str, Pin] = {}
    for record in _section_records(lines, "PINS"):
        head = re.match(r"-\s+(\S+)", record)
        placement = re.search(
            r"\+\s+(?:PLACED|FIXED|COVER)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)", record
        )
        if head and placement:
            pins[head.group(1)] = Pin(head.group(1), int(placement.group(1)), int(placement.group(2)))

    nets: list[Net] = []
    for record in _section_records(lines, "NETS"):
        head = re.match(r"-\s+(\S+)", record)
        if not head:
            continue
        connections = tuple(re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", record))
        nets.append(Net(head.group(1), connections))

    return DefDesign(
        name=name_match.group(1), dbu_per_micron=int(units_match.group(1)),
        diearea=tuple(int(die_match.group(index)) for index in range(1, 5)), rows=tuple(rows),
        components=components, pins=pins, nets=tuple(nets), source_text=text,
    )


def _oriented_pin(component: Component, macro: Macro, pin_name: str, site: Mapping[str, object] | None = None) -> tuple[int, int]:
    px, py = macro.pins.get(pin_name, (macro.width // 2, macro.height // 2))
    x = int(site["x"]) if site is not None else component.x
    y = int(site["y"]) if site is not None else component.y
    orient = str(site["orient"]) if site is not None else component.orient
    if orient == "N":
        dx, dy = px, py
    elif orient == "S":
        dx, dy = macro.width - px, macro.height - py
    elif orient == "FN":
        dx, dy = macro.width - px, py
    elif orient == "FS":
        dx, dy = px, macro.height - py
    else:
        dx, dy = macro.width // 2, macro.height // 2
    return x + dx, y + dy


def _bbox(component: Component, macro: Macro) -> tuple[int, int, int, int]:
    width, height = macro.width, macro.height
    if component.orient in {"E", "W", "FE", "FW"}:
        width, height = height, width
    return component.x, component.y, component.x + width, component.y + height


def _overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return left[0] < right[2] and right[0] < left[2] and left[1] < right[3] and right[1] < left[3]


def _incident_nets(design: DefDesign, selected: set[str]) -> list[Net]:
    return [net for net in design.nets if any(instance in selected for instance, _pin in net.connections)]


def assignment_hpwl(
    design: DefDesign,
    macros: Mapping[str, Macro],
    selected_cells: Sequence[str],
    sites: Sequence[Mapping[str, object]],
    assignment: Sequence[int],
    incident_nets: Sequence[Net] | None = None,
) -> int:
    assigned = {cell: sites[assignment[index]] for index, cell in enumerate(selected_cells)}
    nets = list(incident_nets) if incident_nets is not None else _incident_nets(design, set(selected_cells))
    total = 0
    for net in nets:
        terminals: list[tuple[int, int]] = []
        for instance, pin_name in net.connections:
            if instance == "PIN":
                pin = design.pins.get(pin_name)
                if pin is not None:
                    terminals.append((pin.x, pin.y))
                continue
            component = design.components.get(instance)
            if component is None or component.master not in macros:
                continue
            terminals.append(_oriented_pin(component, macros[component.master], pin_name, assigned.get(instance)))
        if len(terminals) >= 2:
            xs = [point[0] for point in terminals]
            ys = [point[1] for point in terminals]
            total += max(xs) - min(xs) + max(ys) - min(ys)
    return total


def total_design_hpwl(design: DefDesign, macros: Mapping[str, Macro]) -> int:
    """Return the bridge's pin-aware HPWL proxy over every ordinary DEF net."""

    return assignment_hpwl(design, macros, (), (), (), design.nets)


def _candidate_empty_sites(
    design: DefDesign,
    macros: Mapping[str, Macro],
    selected: Sequence[Component],
    width: int,
    height: int,
    count: int,
) -> list[dict[str, object]]:
    selected_names = {component.name for component in selected}
    occupied = [
        (component.name, _bbox(component, macros[component.master]))
        for component in design.components.values()
        if component.name not in selected_names and component.master in macros
    ]
    center_x = sum(component.x + width // 2 for component in selected) / len(selected)
    center_y = sum(component.y + height // 2 for component in selected) / len(selected)
    usable_rows = sorted(
        (row for row in design.rows if row.count_y == 1),
        key=lambda row: (row.y, row.x, row.name),
    )
    row_ys = sorted({row.y for row in usable_rows})
    occupied_by_y: dict[int, list[tuple[int, int]]] = {y: [] for y in row_ys}
    for _name, box in occupied:
        # A candidate [row_y, row_y + height) overlaps this component exactly
        # when row_y is in (box_y1 - height, box_y2).  Indexing intervals by
        # row avoids the previous all-components scan at every site, which is
        # prohibitive for benchmark-scale placed designs.
        first = bisect_right(row_ys, box[1] - height)
        last = bisect_left(row_ys, box[3])
        for row_y in row_ys[first:last]:
            occupied_by_y[row_y].append((box[0], box[2]))
    for intervals in occupied_by_y.values():
        intervals.sort()

    original_boxes = [
        (component.x, component.y, component.x + width, component.y + height)
        for component in selected
    ]
    candidates: list[tuple[float, dict[str, object]]] = []
    # Only the nearest few vacancies on any one row can contribute to the
    # globally nearest `count` sites.  We still scan outward until that many
    # legal vacancies are found, so dense rows remain supported.
    per_row_limit = count + 4
    for row in usable_rows:
        if row.step_x <= 0:
            continue
        end_x = row.x + row.count_x * row.step_x
        max_index = min(row.count_x - 1, (end_x - width - row.x) // row.step_x)
        if max_index < 0:
            continue
        target = (center_x - width / 2 - row.x) / row.step_x
        nearest_index = min(max(int(round(target)), 0), max_index)
        index_order: list[int] = [nearest_index]
        for delta in range(1, max_index + 1):
            left = nearest_index - delta
            right = nearest_index + delta
            if left >= 0:
                index_order.append(left)
            if right <= max_index:
                index_order.append(right)
            if left < 0 and right > max_index:
                break
        found_on_row = 0
        for index in index_order:
            x = row.x + index * row.step_x
            y = row.y
            box = (x, y, x + width, y + height)
            if any(_overlap(box, other) for other in original_boxes):
                continue
            if any(x < right and left < x + width for left, right in occupied_by_y.get(y, ())):
                continue
            distance = abs((x + width / 2) - center_x) + abs((y + height / 2) - center_y)
            candidates.append((distance, {"x": x, "y": y, "orient": row.orient, "source": "vacant_row_site"}))
            found_on_row += 1
            if found_on_row == per_row_limit:
                break
    candidates.sort(key=lambda item: (item[0], item[1]["y"], item[1]["x"]))
    chosen: list[dict[str, object]] = []
    for _distance, site in candidates:
        box = (int(site["x"]), int(site["y"]), int(site["x"]) + width, int(site["y"]) + height)
        if any(
            _overlap(box, (int(other["x"]), int(other["y"]), int(other["x"]) + width, int(other["y"]) + height))
            for other in chosen
        ):
            continue
        chosen.append(site)
        if len(chosen) == count:
            return chosen
    return chosen


def _window_from_cells(
    design: DefDesign,
    macros: Mapping[str, Macro],
    selected: Sequence[Component],
    extra_sites: int,
    window_index: int,
) -> dict[str, object] | None:
    macro = macros[selected[0].master]
    original_sites = [
        {"site_id": index, "x": component.x, "y": component.y, "orient": component.orient, "source": "initial_cell_site"}
        for index, component in enumerate(selected)
    ]
    vacant = _candidate_empty_sites(design, macros, selected, macro.width, macro.height, extra_sites)
    if len(vacant) < extra_sites:
        return None
    sites = original_sites + [
        {"site_id": len(original_sites) + index, **site} for index, site in enumerate(vacant)
    ]
    cells = [component.name for component in selected]
    incident = _incident_nets(design, set(cells))
    if not incident:
        return None
    scored: list[tuple[int, tuple[int, ...]]] = []
    for assignment in permutations(range(len(sites)), len(cells)):
        scored.append((assignment_hpwl(design, macros, cells, sites, assignment, incident), assignment))
    scored.sort(key=lambda row: (row[0], row[1]))
    initial_assignment = tuple(range(len(cells)))
    initial_hpwl = assignment_hpwl(design, macros, cells, sites, initial_assignment, incident)
    optimum = scored[0][0]
    optimum_rows = [assignment for cost, assignment in scored if cost == optimum]
    best_assignment = optimum_rows[0]
    return {
        "window_id": f"{design.name}_w{window_index:04d}",
        "selection_policy": "spatial_same_size_cells_plus_nearest_legal_vacancies",
        "objective": "incident-net HPWL using LEF representative pin centers and fixed external anchors",
        "cell_count": len(cells),
        "site_count": len(sites),
        "feasible_assignment_count": math.perm(len(sites), len(cells)),
        "movable_cells": [
            {
                "cell_id": component.name, "master": component.master, "width_dbu": macro.width,
                "height_dbu": macro.height, "initial_site_id": index,
            }
            for index, component in enumerate(selected)
        ],
        "candidate_sites": sites,
        "incident_net_count": len(incident),
        "incident_nets": [net.name for net in incident],
        "initial_assignment": {cell: initial_assignment[index] for index, cell in enumerate(cells)},
        "exact_best_assignment": {cell: best_assignment[index] for index, cell in enumerate(cells)},
        "initial_hpwl_dbu": initial_hpwl,
        "exact_hpwl_dbu": optimum,
        "exact_optimum_degeneracy": len(optimum_rows),
        "exact_improvement_dbu": initial_hpwl - optimum,
        "exact_improvement_percent": 100.0 * (initial_hpwl - optimum) / initial_hpwl if initial_hpwl else 0.0,
    }


def extract_windows(
    design: DefDesign,
    macros: Mapping[str, Macro],
    count: int = 5,
    cells_per_window: int = 4,
    sites_per_window: int = 6,
    selection_seed: int = 20260827,
) -> list[dict[str, object]]:
    if sites_per_window < cells_per_window:
        raise ValueError("sites_per_window must be at least cells_per_window")
    row_y = sorted({row.y for row in design.rows})
    row_pitches = [right - left for left, right in zip(row_y, row_y[1:]) if right > left]
    single_row_height = min(row_pitches) if row_pitches else None
    eligible = [
        component for component in design.components.values()
        if component.status == "PLACED"
        and component.master in macros
        and not component.name.startswith("PHY_")
        and (single_row_height is None or macros[component.master].height == single_row_height)
    ]
    groups: dict[tuple[int, int], list[Component]] = {}
    for component in eligible:
        macro = macros[component.master]
        groups.setdefault((macro.width, macro.height), []).append(component)
    groups = {key: value for key, value in groups.items() if len(value) >= cells_per_window}
    if not groups:
        raise ValueError("no same-size movable-cell group can form a window")
    ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    anchors: list[tuple[tuple[int, int], Component]] = []
    for key, components in ordered_groups:
        anchors.extend((key, component) for component in components)
    anchors.sort(key=lambda item: (item[1].y, item[1].x, item[1].name))
    if anchors:
        rotation = selection_seed % len(anchors)
        anchors = anchors[rotation:] + anchors[:rotation]

    windows: list[dict[str, object]] = []
    used_signatures: set[tuple[str, ...]] = set()
    for key, anchor in anchors:
        pool = groups[key]
        width, height = key
        nearby = sorted(
            pool,
            key=lambda component: (
                abs((component.x + width // 2) - (anchor.x + width // 2))
                + abs((component.y + height // 2) - (anchor.y + height // 2)),
                component.name,
            ),
        )
        selected = nearby[:cells_per_window]
        signature = tuple(sorted(component.name for component in selected))
        if signature in used_signatures:
            continue
        window = _window_from_cells(
            design, macros, selected, sites_per_window - cells_per_window, len(windows)
        )
        if window is None:
            continue
        windows.append(window)
        used_signatures.add(signature)
        if len(windows) == count:
            break
    if len(windows) < count:
        raise ValueError(f"extracted only {len(windows)} of {count} requested windows")
    return windows


def canonical_manifest_hash(payload: Mapping[str, object]) -> str:
    clean = dict(payload)
    clean.pop("manifest_hash", None)
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    def_path: Path,
    lef_path: Path,
    source: Mapping[str, object],
    count: int = 5,
    cells_per_window: int = 4,
    sites_per_window: int = 6,
    selection_seed: int = 20260827,
) -> dict[str, object]:
    design = parse_def(def_path)
    macros = parse_lef(lef_path, design.dbu_per_micron)
    windows = extract_windows(design, macros, count, cells_per_window, sites_per_window, selection_seed)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            **source,
            "design_name": design.name,
            "def_sha256": sha256(def_path),
            "lef_sha256": sha256(lef_path),
        },
        "units": {"distance": "database_units", "dbu_per_micron": design.dbu_per_micron},
        "extraction_policy": {
            "window_count": count, "cells_per_window": cells_per_window,
            "sites_per_window": sites_per_window, "selection_seed": selection_seed,
            "selection_is_qaoa_blind": True,
            "cell_compatibility": "same width and height",
            "empty_site_policy": "nearest nonoverlapping row-aligned vacancy",
            "external_context": "all incident nets with non-window terminals fixed as anchors",
        },
        "windows": windows,
    }
    payload["manifest_hash"] = canonical_manifest_hash(payload)
    return payload


def reinsert_window(def_text: str, window: Mapping[str, object]) -> str:
    sites = {int(site["site_id"]): site for site in window["candidate_sites"]}
    cells = {str(cell["cell_id"]): str(cell["master"]) for cell in window["movable_cells"]}
    assignment = {str(cell): int(site_id) for cell, site_id in window["exact_best_assignment"].items()}
    updated = def_text
    for cell_name, master in cells.items():
        site = sites[assignment[cell_name]]
        pattern = re.compile(
            rf"(^\s*-\s+{re.escape(cell_name)}\s+{re.escape(master)}\b[^;]*?\+\s+)"
            rf"(?:PLACED|FIXED|COVER)(\s+\(\s*)-?\d+\s+-?\d+(\s*\)\s+)\S+",
            re.MULTILINE,
        )
        def replacement(match: re.Match[str]) -> str:
            return (
                f"{match.group(1)}PLACED{match.group(2)}"
                f"{int(site['x'])} {int(site['y'])}{match.group(3)}{site['orient']}"
            )

        updated, replacements = pattern.subn(replacement, updated, count=1)
        if replacements != 1:
            raise ValueError(f"could not reinsert {cell_name}")
    return updated


def write_gate_outputs(
    manifest: Mapping[str, object], def_path: Path, output_dir: Path, lef_path: Path | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reinserted = output_dir / "reinserted"
    reinserted.mkdir(exist_ok=True)
    manifest_path = output_dir / "window_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_text = def_path.read_text(encoding="utf-8", errors="replace")
    checks: list[dict[str, object]] = []
    baseline_total_hpwl: int | None = None
    if lef_path is not None:
        baseline_design = parse_def(def_path)
        baseline_macros = parse_lef(lef_path, baseline_design.dbu_per_micron)
        baseline_total_hpwl = total_design_hpwl(baseline_design, baseline_macros)
    for window in manifest["windows"]:
        output_def = reinserted / f"{window['window_id']}.def"
        output_def.write_text(reinsert_window(source_text, window), encoding="utf-8")
        if lef_path is not None:
            reparsed = parse_def(output_def)
            macros = parse_lef(lef_path, reparsed.dbu_per_micron)
            cells = [str(cell["cell_id"]) for cell in window["movable_cells"]]
            actual_sites = [
                {
                    "x": reparsed.components[cell].x,
                    "y": reparsed.components[cell].y,
                    "orient": reparsed.components[cell].orient,
                }
                for cell in cells
            ]
            actual_hpwl = assignment_hpwl(reparsed, macros, cells, actual_sites, tuple(range(len(cells))))
            reinserted_total_hpwl = total_design_hpwl(reparsed, macros)
            expected_delta = int(window["initial_hpwl_dbu"]) - int(window["exact_hpwl_dbu"])
            measured_delta = int(baseline_total_hpwl) - reinserted_total_hpwl
            checks.append(
                {
                    "window_id": window["window_id"],
                    "reinserted_hpwl_dbu": actual_hpwl,
                    "exact_hpwl_dbu": int(window["exact_hpwl_dbu"]),
                    "matches_exact_objective": actual_hpwl == int(window["exact_hpwl_dbu"]),
                    "baseline_total_hpwl_dbu": baseline_total_hpwl,
                    "reinserted_total_hpwl_dbu": reinserted_total_hpwl,
                    "expected_full_design_delta_dbu": expected_delta,
                    "measured_full_design_delta_dbu": measured_delta,
                    "full_design_delta_matches_local": measured_delta == expected_delta,
                }
            )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "window_count": len(manifest["windows"]),
        "all_exactly_solved": True,
        "all_have_reinserted_def": True,
        "all_reinserted_objectives_match": all(bool(check["matches_exact_objective"]) for check in checks) if checks else None,
        "all_full_design_deltas_match_local": all(bool(check["full_design_delta_matches_local"]) for check in checks) if checks else None,
        "reinsertion_objective_checks": checks,
        "total_initial_hpwl_dbu": sum(int(window["initial_hpwl_dbu"]) for window in manifest["windows"]),
        "total_exact_hpwl_dbu": sum(int(window["exact_hpwl_dbu"]) for window in manifest["windows"]),
        "improving_window_count": sum(int(window["exact_hpwl_dbu"]) < int(window["initial_hpwl_dbu"]) for window in manifest["windows"]),
    }
    (output_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--def", dest="def_path", type=Path, required=True)
    parser.add_argument("--lef", dest="lef_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-repository", default="")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--source-platform", default="")
    parser.add_argument("--source-design", default="")
    parser.add_argument("--source-stage", default="detailed_place")
    parser.add_argument("--openroad-version", default="")
    parser.add_argument("--container-digest", default="")
    parser.add_argument("--window-count", type=int, default=5)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--sites", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=20260827)
    args = parser.parse_args()
    source = {
        "repository": args.source_repository, "commit": args.source_commit,
        "platform": args.source_platform, "design": args.source_design,
        "stage": args.source_stage, "openroad_version": args.openroad_version,
        "container_digest": args.container_digest,
    }
    manifest = build_manifest(
        args.def_path, args.lef_path, source, args.window_count, args.cells, args.sites, args.selection_seed
    )
    write_gate_outputs(manifest, args.def_path, args.output_dir, args.lef_path)
    print(json.dumps({"manifest_hash": manifest["manifest_hash"], "windows": len(manifest["windows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
