"""Audit token-oracle geometry, padding, and optimization-context effects.

This is a deterministic construction/transpilation study.  It reconciles the
canonical routed circuit with the isolated-component scaling convention, then
sweeps the variables that can actually change Walsh support: site geometry and
the association between site labels and binary codewords.  No statevector,
noise model, or hardware execution is involved.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import qiskit
from qiskit import transpile

from benchmark_suite import load_manifest, problem_from_manifest
from occupant_circuit import two_qubit_depth
from placement_core import PlacementProblem
from run_token_permutation_p3_quality_uplift import scheduled_qaoa_circuit
from run_token_permutation_quality_validation import MANIFEST
from token_permutation_encoding import (
    token_mixer_circuit,
    token_phase_circuit,
    token_phase_terms,
    token_register_width,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/qaoa_placement/mixer_reduction/token_geometry_padding_audit_v2"
BASIS = ("rz", "sx", "x", "cx")
SITES = (6, 8, 9, 12, 16)
GEOMETRY_FAMILIES = ("compact", "line", "l_shape", "sparse_scatter")
REPLICATES = 5
TRANSPILER_SEED = 123
GAMMA = 0.17
BETA = 0.23


def canonical_problem() -> tuple[str, PlacementProblem]:
    manifest = load_manifest(MANIFEST)
    return str(manifest[0]["instance_id"]), problem_from_manifest(manifest[0])


def fixed_six_net_problem(sites: Sequence[tuple[int, int]]) -> PlacementProblem:
    cells = ("A", "B", "C", "D")
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    weights = (0.71, 0.83, 0.97, 1.09, 1.23, 1.37)
    nets = tuple(
        (cells[left], cells[right], weight)
        for (left, right), weight in zip(pairs, weights)
    )
    return PlacementProblem(cells, tuple(sites), nets, 40.0)


def compact_geometry(sites: int) -> tuple[tuple[int, int], ...]:
    widths = {6: 3, 8: 4, 9: 3, 12: 4, 16: 4}
    width = widths[sites]
    return tuple((index % width, index // width) for index in range(sites))


def line_geometry(sites: int) -> tuple[tuple[int, int], ...]:
    return tuple((index, 0) for index in range(sites))


def l_geometry(sites: int) -> tuple[tuple[int, int], ...]:
    horizontal = (sites + 1) // 2
    return tuple((index, 0) for index in range(horizontal)) + tuple(
        (0, index) for index in range(1, sites - horizontal + 1)
    )


def scatter_geometry(sites: int, seed: int) -> tuple[tuple[int, int], ...]:
    rng = np.random.default_rng(seed)
    extent = 3 * sites
    candidates = [(x, y) for x in range(extent) for y in range(extent)]
    indices = rng.choice(len(candidates), size=sites, replace=False)
    return tuple(candidates[int(index)] for index in indices)


def geometry(family: str, sites: int, replicate: int) -> tuple[tuple[int, int], ...]:
    seed = 70_000 + 100 * sites + replicate
    if family == "compact":
        base = compact_geometry(sites)
    elif family == "line":
        base = line_geometry(sites)
    elif family == "l_shape":
        base = l_geometry(sites)
    elif family == "sparse_scatter":
        return scatter_geometry(sites, seed)
    else:  # pragma: no cover
        raise ValueError(family)
    if replicate == 0:
        return base
    rng = np.random.default_rng(seed)
    return tuple(base[int(index)] for index in rng.permutation(sites))


def lowered_metrics(circuit, optimization_level: int) -> dict[str, object]:
    started = time.time()
    lowered = transpile(
        circuit,
        basis_gates=list(BASIS),
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILER_SEED,
    )
    operations = lowered.count_ops()
    return {
        "cx": int(operations.get("cx", 0)),
        "depth": int(lowered.depth()),
        "two_qubit_depth": int(two_qubit_depth(lowered)),
        "seconds": time.time() - started,
    }


def canonical_reconciliation() -> list[dict[str, object]]:
    instance_id, problem = canonical_problem()
    circuits = {
        "phase_beam_p1": token_phase_circuit(problem, GAMMA, ordering_policy="beam_search"),
        "phase_repository_p1": token_phase_circuit(
            problem, GAMMA, ordering_policy="repository_order"
        ),
        "mixer_line_p1": token_mixer_circuit(problem, BETA, "line", reps=1),
        "mixer_ring_p1": token_mixer_circuit(problem, BETA, "ring", reps=1),
        "integrated_line_rotating_p3": scheduled_qaoa_circuit(
            problem, "line", "rotating", 3
        ),
        "integrated_ring_reversed_p3": scheduled_qaoa_circuit(
            problem, "ring", "reversed", 3
        ),
    }
    rows = []
    for name, circuit in circuits.items():
        for level in (1, 3):
            metrics = lowered_metrics(circuit, level)
            rows.append(
                {
                    "instance_id": instance_id,
                    "cells": len(problem.cells),
                    "sites": len(problem.sites),
                    "net_count": len(problem.nets),
                    "site_geometry": json.dumps(problem.sites),
                    "component": name,
                    "optimization_level": level,
                    "logical_cx": metrics["cx"],
                    "logical_depth": metrics["depth"],
                    "logical_two_qubit_depth": metrics["two_qubit_depth"],
                    "transpile_seconds": metrics["seconds"],
                }
            )
    return rows


def geometry_rows() -> list[dict[str, object]]:
    rows = []
    for sites in SITES:
        for family in GEOMETRY_FAMILIES:
            for replicate in range(REPLICATES):
                site_coordinates = geometry(family, sites, replicate)
                problem = fixed_six_net_problem(site_coordinates)
                terms, _global_phase = token_phase_terms(problem, GAMMA)
                phase = token_phase_circuit(
                    problem, GAMMA, ordering_policy="repository_order"
                )
                level1 = lowered_metrics(phase, 1)
                rows.append(
                    {
                        "cells": len(problem.cells),
                        "sites": sites,
                        "register_width": token_register_width(problem),
                        "invalid_codewords": 2 ** token_register_width(problem) - sites,
                        "net_count": len(problem.nets),
                        "geometry_family": family,
                        "replicate": replicate,
                        "replicate_seed": 70_000 + 100 * sites + replicate,
                        "site_geometry": json.dumps(site_coordinates),
                        "walsh_terms": len(terms),
                        "phase_cx_level1": level1["cx"],
                        "phase_depth_level1": level1["depth"],
                        "phase_two_qubit_depth_level1": level1["two_qubit_depth"],
                        "transpile_seconds": level1["seconds"],
                    }
                )
                print(f"counted {sites}s {family} replicate {replicate}", flush=True)
    return rows


def mixer_rows() -> list[dict[str, object]]:
    rows = []
    for sites in SITES:
        problem = fixed_six_net_problem(compact_geometry(sites))
        width = token_register_width(problem)
        for topology in ("line", "ring"):
            circuit = token_mixer_circuit(problem, BETA, topology, reps=1)
            level1 = lowered_metrics(circuit, 1)
            expected = 14 * width * (sites - 1 if topology == "line" else sites)
            if level1["cx"] != expected:
                raise RuntimeError(
                    f"{sites}s {topology}: observed {level1['cx']} != formula {expected}"
                )
            level3 = lowered_metrics(circuit, 3)
            rows.append(
                {
                    "sites": sites,
                    "register_width": width,
                    "topology": topology,
                    "formula_cx": expected,
                    "mixer_cx_level1": level1["cx"],
                    "mixer_cx_level3": level3["cx"],
                    "mixer_depth_level1": level1["depth"],
                    "mixer_depth_level3": level3["depth"],
                }
            )
    return rows


def percentile_summary(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": int(np.min(array)),
        "max": int(np.max(array)),
    }


def geometry_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for sites in SITES:
        for family in (*GEOMETRY_FAMILIES, "all"):
            group = [
                row
                for row in rows
                if row["sites"] == sites
                and (family == "all" or row["geometry_family"] == family)
            ]
            level1 = percentile_summary([int(row["phase_cx_level1"]) for row in group])
            support = percentile_summary([int(row["walsh_terms"]) for row in group])
            output.append(
                {
                    "cells": 4,
                    "sites": sites,
                    "register_width": int(group[0]["register_width"]),
                    "invalid_codewords": int(group[0]["invalid_codewords"]),
                    "net_count": 6,
                    "geometry_family": family,
                    "instances": len(group),
                    "phase_cx_median": level1["median"],
                    "phase_cx_q1": level1["q1"],
                    "phase_cx_q3": level1["q3"],
                    "phase_cx_min": level1["min"],
                    "phase_cx_max": level1["max"],
                    "walsh_terms_median": support["median"],
                    "walsh_terms_min": support["min"],
                    "walsh_terms_max": support["max"],
                }
            )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(
    canonical: Sequence[Mapping[str, object]],
    summary: Sequence[Mapping[str, object]],
) -> str:
    canonical_level1 = {
        str(row["component"]): row
        for row in canonical
        if row["optimization_level"] == 1
    }
    canonical_level3 = {
        str(row["component"]): row
        for row in canonical
        if row["optimization_level"] == 3
    }
    lines = [
        "# Token Geometry and Padding Audit",
        "",
        "This compiler-only audit separates isolated level-1 component counts",
        "from integrated level-3 complete-circuit counts. It also fixes the",
        "four-cell complete six-net graph while sweeping site geometry and the",
        "site-label/codeword association. Continuous weights are fixed because",
        "they alter rotation angles, not generic Walsh support.",
        "",
        "## Canonical reconciliation",
        "",
        f"The frozen canonical 4c/6s instance has three nets and sites "
        f"`{canonical[0]['site_geometry']}`. Its beam-ordered isolated phase is "
        f"{canonical_level1['phase_beam_p1']['logical_cx']} CX/layer at level 1; "
        f"the isolated line and ring mixers are "
        f"{canonical_level1['mixer_line_p1']['logical_cx']} and "
        f"{canonical_level1['mixer_ring_p1']['logical_cx']} CX/layer. The complete "
        f"p=3 line/ring circuits are "
        f"{canonical_level1['integrated_line_rotating_p3']['logical_cx']}/"
        f"{canonical_level1['integrated_ring_reversed_p3']['logical_cx']} CX at "
        f"level 1 and {canonical_level3['integrated_line_rotating_p3']['logical_cx']}/"
        f"{canonical_level3['integrated_ring_reversed_p3']['logical_cx']} CX at "
        "level 3. Thus the exact isolated-mixer difference is 126 CX at p=3; "
        "the 117-CX integrated level-3 difference reflects whole-circuit "
        "optimization context, not a failure of the component formula.",
        "",
        "## Geometry and padding sweep",
        "",
        "| Sites | k | Dead codes | Geometries | Phase CX median [IQR] | Range | Walsh-term range |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["geometry_family"] != "all":
            continue
        lines.append(
            f"| {row['sites']} | {row['register_width']} | {row['invalid_codewords']} | "
            f"{row['instances']} | {row['phase_cx_median']:.0f} "
            f"[{row['phase_cx_q1']:.0f}, {row['phase_cx_q3']:.0f}] | "
            f"{row['phase_cx_min']}--{row['phase_cx_max']} | "
            f"{row['walsh_terms_min']}--{row['walsh_terms_max']} |"
        )
    lines.extend(
        [
            "",
            "The isolated mixer identities are `14 p k (m-1)` CX for a line",
            "and `14 p k m` CX for a ring at optimization level 1. The loose",
            "phase bound is `O(p |E| m^2 log m)`. The empirical phase spread is",
            "caused by geometry and site/codeword alignment; it must not be",
            "represented as variation over weights or graph labels alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.time()
    canonical = canonical_reconciliation()
    rows = geometry_rows()
    mixers = mixer_rows()
    summary = geometry_summary(rows)
    OUT.mkdir(parents=True)
    write_csv(OUT / "canonical_reconciliation.csv", canonical)
    write_csv(OUT / "geometry_run_level.csv", rows)
    write_csv(OUT / "geometry_summary.csv", summary)
    write_csv(OUT / "mixer_summary.csv", mixers)
    (OUT / "RESULTS_REPORT.md").write_text(report(canonical, summary))
    (OUT / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "qiskit": qiskit.__version__,
                "basis_gates": list(BASIS),
                "transpiler_seed": TRANSPILER_SEED,
                "phase_ordering": "repository_order",
                "gamma": GAMMA,
                "beta": BETA,
                "site_counts": list(SITES),
                "geometry_families": list(GEOMETRY_FAMILIES),
                "replicates_per_family": REPLICATES,
                "fixed_net_weights": [0.71, 0.83, 0.97, 1.09, 1.23, 1.37],
                "elapsed_seconds": time.time() - started,
                "evidence_kind": "logical construction/transpilation; no hardware execution",
                "command": (
                    "MPLCONFIGDIR=/tmp/qeda-geometry-mpl "
                    "PYTHONPATH=experiments/qaoa_placement:experiments/qaoa_placement/mixer_reduction "
                    ".venv/bin/python experiments/qaoa_placement/mixer_reduction/"
                    "run_token_geometry_padding_audit.py"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
