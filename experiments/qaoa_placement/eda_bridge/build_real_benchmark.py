"""Build a frozen, stratified benchmark from real placed-design windows.

Selection is deliberately independent of exact improvement and QAOA outcome.
Exact enumeration is retained in the manifest so every solver sees the same
pin-aware incident-net HPWL objective, including fixed external anchors.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path
from typing import Mapping, Sequence

from .placement_windows import (
    _incident_nets,
    assignment_hpwl,
    canonical_manifest_hash,
    extract_windows,
    parse_def,
    parse_lef,
    sha256,
)


SCHEMA_VERSION = "qeda-real-placement-benchmark-v1"
STRATA = ("low_connectivity", "medium_connectivity", "high_connectivity")


def _hash_key(seed: int, window: Mapping[str, object]) -> str:
    cells = ",".join(sorted(str(cell["cell_id"]) for cell in window["movable_cells"]))
    return hashlib.sha256(f"{seed}|{cells}".encode()).hexdigest()


def _assign_strata(candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        (dict(window) for window in candidates),
        key=lambda window: (int(window["incident_net_count"]), str(window["window_id"])),
    )
    total = len(ordered)
    for index, window in enumerate(ordered):
        stratum_index = min(2, (3 * index) // max(total, 1))
        window["selection_stratum"] = STRATA[stratum_index]
        window["candidate_pool_connectivity_rank"] = index
    return ordered


def select_stratified_windows(
    candidates: Sequence[Mapping[str, object]],
    count: int,
    selection_seed: int,
) -> list[dict[str, object]]:
    """Select balanced connectivity strata without consulting exact quality."""

    if count % len(STRATA):
        raise ValueError(f"window count must be divisible by {len(STRATA)}")
    quota = count // len(STRATA)
    stratified = _assign_strata(candidates)
    chosen: list[dict[str, object]] = []
    used_cells: set[str] = set()
    for stratum in STRATA:
        rows = sorted(
            (window for window in stratified if window["selection_stratum"] == stratum),
            key=lambda window: (_hash_key(selection_seed, window), str(window["window_id"])),
        )
        accepted = 0
        for window in rows:
            cells = {str(cell["cell_id"]) for cell in window["movable_cells"]}
            if cells & used_cells:
                continue
            chosen.append(window)
            used_cells.update(cells)
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            raise ValueError(
                f"only selected {accepted}/{quota} disjoint windows for {stratum}; "
                "increase --candidate-pool"
            )
    return chosen


def add_cost_landscape(
    design,
    macros: Mapping[str, object],
    window: Mapping[str, object],
) -> dict[str, object]:
    output = dict(window)
    cells = [str(cell["cell_id"]) for cell in window["movable_cells"]]
    sites = list(window["candidate_sites"])
    incident = _incident_nets(design, set(cells))
    landscape: list[list[object]] = []
    for assignment in permutations(range(len(sites)), len(cells)):
        cost = assignment_hpwl(design, macros, cells, sites, assignment, incident)
        landscape.append([list(assignment), int(cost)])
    landscape.sort(key=lambda row: tuple(row[0]))
    costs = [int(row[1]) for row in landscape]
    if min(costs) != int(window["exact_hpwl_dbu"]):
        raise ValueError(f"cost landscape disagrees with exact result for {window['window_id']}")
    output["cost_landscape_encoding"] = "[cell-site permutation, incident-net HPWL DBU]"
    output["cost_landscape"] = landscape
    output["maximum_hpwl_dbu"] = max(costs)
    return output


def build_design_benchmark(
    def_path: Path,
    lef_path: Path,
    source: Mapping[str, object],
    window_count: int = 12,
    candidate_pool: int = 120,
    selection_seed: int = 20260827,
) -> dict[str, object]:
    design = parse_def(def_path)
    macros = parse_lef(lef_path, design.dbu_per_micron)
    candidates = extract_windows(
        design,
        macros,
        count=candidate_pool,
        cells_per_window=4,
        sites_per_window=6,
        selection_seed=selection_seed,
    )
    selected = select_stratified_windows(candidates, window_count, selection_seed)
    design_key = str(source["design"])
    windows: list[dict[str, object]] = []
    for index, selected_window in enumerate(selected):
        window = add_cost_landscape(design, macros, selected_window)
        window["source_candidate_window_id"] = window["window_id"]
        window["window_id"] = f"{design_key}_w{index:04d}"
        windows.append(window)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            **source,
            "def_sha256": sha256(def_path),
            "lef_sha256": sha256(lef_path),
            "def_design_name": design.name,
            "component_count": len(design.components),
            "net_count": len(design.nets),
        },
        "units": {"distance": "database_units", "dbu_per_micron": design.dbu_per_micron},
        "selection_policy": {
            "selection_seed": selection_seed,
            "candidate_pool_size": candidate_pool,
            "selected_window_count": window_count,
            "cells_per_window": 4,
            "sites_per_window": 6,
            "strata": list(STRATA),
            "stratum_feature": "incident_net_count rank tertile",
            "windows_per_stratum": window_count // len(STRATA),
            "cell_disjoint_within_design": True,
            "consults_exact_improvement": False,
            "consults_qaoa_outcomes": False,
        },
        "windows": windows,
    }
    payload["manifest_hash"] = canonical_manifest_hash(payload)
    return payload


def combine_design_benchmarks(
    paths: Sequence[Path],
    benchmark_id: str = "orfs_nangate45_three_design_36window_v2",
) -> dict[str, object]:
    designs = [json.loads(path.read_text()) for path in paths]
    for path, design in zip(paths, designs):
        if design.get("manifest_hash") != canonical_manifest_hash(design):
            raise ValueError(f"invalid design manifest hash: {path}")
    windows = [window for design in designs for window in design["windows"]]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "design_count": len(designs),
        "window_count": len(windows),
        "design_manifests": [
            {
                "path": str(path),
                "manifest_hash": design["manifest_hash"],
                "source": design["source"],
                "selection_policy": design["selection_policy"],
            }
            for path, design in zip(paths, designs)
        ],
        "windows": windows,
    }
    payload["manifest_hash"] = canonical_manifest_hash(payload)
    return payload


def _source(args: argparse.Namespace) -> dict[str, object]:
    return {
        "repository": args.source_repository,
        "commit": args.source_commit,
        "platform": args.source_platform,
        "design": args.source_design,
        "stage": args.source_stage,
        "openroad_version": args.openroad_version,
        "container_digest": args.container_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-design")
    build.add_argument("--def", dest="def_path", type=Path, required=True)
    build.add_argument("--lef", dest="lef_path", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--source-repository", required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--source-platform", required=True)
    build.add_argument("--source-design", required=True)
    build.add_argument("--source-stage", default="legalize_only_from_3_4_place_resized")
    build.add_argument("--openroad-version", required=True)
    build.add_argument("--container-digest", required=True)
    build.add_argument("--window-count", type=int, default=12)
    build.add_argument("--candidate-pool", type=int, default=120)
    build.add_argument("--selection-seed", type=int, default=20260827)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--design-manifest", action="append", type=Path, required=True)
    combine.add_argument(
        "--benchmark-id",
        default="orfs_nangate45_three_design_36window_v2",
        help="stable identifier for the combined corpus",
    )
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-design":
        payload = build_design_benchmark(
            args.def_path,
            args.lef_path,
            _source(args),
            args.window_count,
            args.candidate_pool,
            args.selection_seed,
        )
    else:
        payload = combine_design_benchmarks(args.design_manifest, args.benchmark_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_hash": payload["manifest_hash"], "windows": len(payload["windows"])}))


if __name__ == "__main__":
    main()
