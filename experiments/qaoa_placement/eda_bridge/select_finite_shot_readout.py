"""Freeze one routed finite-shot readout per benchmark window.

This replication-stage utility deliberately performs no comparison against an
ideal QAOA run or a classical solver.  It applies the predeclared best-of-4096,
then lowest-seed tie break to the four fixed finite-shot runs for each window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from .placement_windows import canonical_manifest_hash, sha256
from .summarize_finite_shot_optimization import selected_validation_rows


OBJECTIVE_SHOTS = 8192
SEEDS = {11, 17, 23, 29}


def load_and_validate(
    manifest: Mapping[str, object], rows: Sequence[Mapping[str, str]]
) -> list[dict[str, object]]:
    expected_ids = {str(window["window_id"]) for window in manifest["windows"]}
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["window_id"]), []).append(row)
    if set(grouped) != expected_ids:
        raise ValueError("finite-shot rows do not exactly cover the manifest windows")
    for window_id, group in grouped.items():
        if len(group) != len(SEEDS):
            raise ValueError(f"{window_id} does not have four finite-shot runs")
        if {int(row["seed"]) for row in group} != SEEDS:
            raise ValueError(f"{window_id} does not have the frozen seed set")
        if {int(row["objective_shots"]) for row in group} != {OBJECTIVE_SHOTS}:
            raise ValueError(f"{window_id} does not use {OBJECTIVE_SHOTS} objective shots")
        if any(
            row["graph"] != "ring"
            or row["schedule"] != "reversed"
            or int(row["p"]) != 3
            or row["objective"] != "beat_initial"
            for row in group
        ):
            raise ValueError(f"{window_id} does not use the frozen QAOA protocol")
    return selected_validation_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--finite-run-level", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("manifest_hash") != canonical_manifest_hash(manifest):
        raise ValueError("benchmark manifest hash is invalid")
    with args.finite_run_level.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = load_and_validate(manifest, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "selected_readout.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    exact_hits = sum(str(row["sampled_exact_hit_4096"]).lower() == "true" for row in selected)
    summary = {
        "schema_version": "qeda-finite-shot-readout-selection-v1",
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_manifest_hash": manifest["manifest_hash"],
        "window_count": len(selected),
        "selection_rule": "minimum sampled_best_4096_hpwl_dbu, then minimum seed",
        "objective_shots": OBJECTIVE_SHOTS,
        "seeds": sorted(SEEDS),
        "sampled_exact_hit_count": exact_hits,
        "sampled_exact_hit_fraction": exact_hits / len(selected),
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "finite_run_level_sha256": sha256(args.finite_run_level),
            "selected_readout_sha256": sha256(output),
        },
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
