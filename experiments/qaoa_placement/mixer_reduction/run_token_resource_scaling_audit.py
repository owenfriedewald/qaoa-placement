"""Construct and count exact token/permutation QAOA components across sizes."""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Mapping, Sequence

import qiskit
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from occupant_circuit import two_qubit_depth
from placement_core import PlacementProblem, grid_sites
from token_permutation_encoding import (
    token_data_qubits,
    token_mixer_circuit,
    token_phase_circuit,
    token_register_width,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/qaoa_placement/mixer_reduction/token_resource_scaling_v2"
SPECS = ((4, 6, 3, 2), (5, 8, 4, 2), (6, 9, 3, 3), (8, 12, 4, 3))
BASIS = ("rz", "sx", "x", "cx")


def representative_problem(cells: int, sites: int, width: int, height: int) -> PlacementProblem:
    names = tuple(f"C{index}" for index in range(cells))
    target_edges = min(2 * cells, cells * (cells - 1) // 2)
    pairs: list[tuple[int, int]] = []
    for distance in range(1, cells):
        for left in range(cells):
            right = (left + distance) % cells
            pair = tuple(sorted((left, right)))
            if pair not in pairs:
                pairs.append(pair)
            if len(pairs) == target_edges:
                break
        if len(pairs) == target_edges:
            break
    nets = tuple(
        (names[left], names[right], float(1 + index % 3))
        for index, (left, right) in enumerate(pairs)
    )
    return PlacementProblem(names, grid_sites(width, height), nets, 40.0)


def metrics(circuit) -> dict[str, object]:
    started = time.time()
    lowered = transpile(
        circuit,
        basis_gates=list(BASIS),
        optimization_level=1,
        seed_transpiler=123,
    )
    ops = lowered.count_ops()
    return {
        "logical_cx": int(ops.get("cx", 0)),
        "logical_depth": int(lowered.depth()),
        "logical_two_qubit_depth": two_qubit_depth(lowered),
        "logical_one_qubit_gates": int(sum(count for name, count in ops.items() if name != "cx")),
        "logical_transpilation_seconds": time.time() - started,
    }


def routed_metrics(circuit) -> dict[str, object]:
    started = time.time()
    routed = transpile(
        circuit,
        basis_gates=list(BASIS),
        coupling_map=CouplingMap.from_heavy_hex(5),
        optimization_level=1,
        seed_transpiler=118,
        layout_method="sabre",
        routing_method="sabre",
    )
    ops = routed.count_ops()
    return {
        "routing_status": "ok",
        "routed_cx": int(ops.get("cx", 0)),
        "routed_depth": int(routed.depth()),
        "routed_two_qubit_depth": two_qubit_depth(routed),
        "physical_qubits": int(routed.num_qubits),
        "routing_seconds": time.time() - started,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report(rows: Sequence[Mapping[str, object]]) -> str:
    by_size: dict[tuple[int, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        by_size.setdefault((int(row["cells"]), int(row["sites"])), {})[str(row["component"])] = row
    lines = [
        "# Token/Permutation Resource Scaling Audit",
        "",
        "All rows are directly constructed and transpiled exact p=1 circuits.  The",
        "representative net graphs have average cell degree approximately four",
        "(six nets at 4c, then 2n nets). Counts use a fixed basis and transpiler",
        "seed. The canonical p=3 table separately",
        "reports direct synthetic-heavy-hex routing.",
        "",
        "| Size | k | |E| | Active qubits | Phase logical CX | Mixer logical CX | Full logical CX | Phase share (logical) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (cells, sites), group in sorted(by_size.items()):
        phase = group["phase"]
        mixer = group["mixer_line"]
        full = group["full_line"]
        share = float(phase["logical_cx"]) / (float(phase["logical_cx"]) + float(mixer["logical_cx"]))
        lines.append(
            f"| {cells}c/{sites}s | {phase['register_width']} | {phase['nets']} | {full['active_qubits']} | "
            f"{phase['logical_cx']} | {mixer['logical_cx']} | {full['logical_cx']} | {100 * share:.1f}% |"
        )
    lines.extend(
        [
            "",
            "For m sites and register width k=ceil(log2 m), the token encoding uses",
            "mk data qubits plus one reusable mixer ancilla.  The line mixer has",
            "m-1 register-swap blocks per layer; under the audited decomposition its",
            "logical cost is 14kp(m-1) CX under this basis.  The exact phase circuit is net dependent.",
            "Before parity reuse it has at most |E|(4^k-1) nonconstant Walsh terms",
            "per layer, each supported on at most 2k qubits.  The table reports",
            "measured compiler counts rather than presenting that loose bound as a",
            "prediction.",
            "",
            "The dominant component is not monotone: Walsh sparsity makes the mixer",
            "slightly larger at 5c/8s, while the phase dominates the other three",
            "representatives. Counts are resource scaling evidence, not quality scaling,",
            "noise robustness, or hardware execution.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.time()
    rows = []
    for cells, sites, width, height in SPECS:
        problem = representative_problem(cells, sites, width, height)
        data = token_data_qubits(problem)
        phase = token_phase_circuit(problem, 0.17, ordering_policy="repository_order")
        mixer = token_mixer_circuit(problem, 0.23, "line", reps=1)
        complete = QuantumCircuit(data + 1)
        complete.compose(phase, qubits=range(data), inplace=True)
        complete.compose(mixer, qubits=range(data + 1), inplace=True)
        circuits = {
            "phase": phase,
            "mixer_line": mixer,
            "full_line": complete,
        }
        for component, circuit in circuits.items():
            print(f"counting {cells}c/{sites}s {component}", flush=True)
            routed = {
                "routing_status": "not_routed_scaling_count_only",
                "routed_cx": "",
                "routed_depth": "",
                "routed_two_qubit_depth": "",
                "physical_qubits": "",
                "routing_seconds": 0.0,
            }
            row = {
                "cells": cells,
                "sites": sites,
                "nets": len(problem.nets),
                "average_cell_degree": 2.0 * len(problem.nets) / cells,
                "register_width": token_register_width(problem),
                "data_qubits": token_data_qubits(problem),
                "active_qubits": int(circuit.num_qubits),
                "p": 1,
                "component": component,
                **metrics(circuit),
                **routed,
            }
            rows.append(row)
    OUT.mkdir(parents=True)
    write_csv(OUT / "resource_scaling.csv", rows)
    (OUT / "RESULTS_REPORT.md").write_text(report(rows))
    (OUT / "environment.json").write_text(
        json.dumps(
            {
                "command": (
                    "PYTHONPATH=experiments/qaoa_placement:experiments/qaoa_placement/"
                    "mixer_reduction .venv/bin/python experiments/qaoa_placement/"
                    "mixer_reduction/run_token_resource_scaling_audit.py"
                ),
                "python": platform.python_version(),
                "qiskit": qiskit.__version__,
                "basis_gates": list(BASIS),
                "coupling_map": "CouplingMap.from_heavy_hex(5), 57 qubits",
                "logical_seed_transpiler": 123,
                "routing_seed_transpiler": 118,
                "optimization_level": 1,
                "parity_ordering": "repository_order",
                "wall_clock_seconds": time.time() - started,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
