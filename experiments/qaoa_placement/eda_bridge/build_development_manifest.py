"""Freeze a balanced train/validation split for QAOA protocol development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .build_real_benchmark import STRATA
from .placement_windows import canonical_manifest_hash


SCHEMA_VERSION = "qeda-real-placement-development-v1"


def _split_key(seed: int, window: Mapping[str, object]) -> str:
    return hashlib.sha256(f"{seed}|{window['window_id']}".encode()).hexdigest()


def build_development_manifest(
    paths: Sequence[Path],
    benchmark_id: str,
    split_seed: int,
) -> dict[str, object]:
    designs = [json.loads(path.read_text()) for path in paths]
    for path, design in zip(paths, designs):
        if design.get("manifest_hash") != canonical_manifest_hash(design):
            raise ValueError(f"invalid design manifest hash: {path}")

    windows: list[dict[str, object]] = []
    counts: dict[str, int] = {"train": 0, "validation": 0}
    for design in designs:
        design_key = str(design["source"]["design"])
        for stratum in STRATA:
            group = [
                dict(window)
                for window in design["windows"]
                if window["selection_stratum"] == stratum
            ]
            if len(group) % 2:
                raise ValueError(f"odd {design_key}/{stratum} window count: {len(group)}")
            group.sort(key=lambda window: (_split_key(split_seed, window), window["window_id"]))
            midpoint = len(group) // 2
            for index, window in enumerate(group):
                split = "train" if index < midpoint else "validation"
                window["development_split"] = split
                window["development_split_hash_rank"] = index
                counts[split] += 1
                windows.append(window)

    windows.sort(key=lambda window: str(window["window_id"]))
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "design_count": len(designs),
        "window_count": len(windows),
        "split_counts": counts,
        "split_policy": {
            "split_seed": split_seed,
            "balanced_within_design_and_connectivity_stratum": True,
            "consults_exact_improvement": False,
            "consults_classical_outcomes": False,
            "consults_qaoa_outcomes": False,
        },
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-manifest", action="append", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_development_manifest(
        args.design_manifest,
        args.benchmark_id,
        args.split_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "windows": payload["window_count"],
                "splits": payload["split_counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
