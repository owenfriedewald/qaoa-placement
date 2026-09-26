"""Bounded study for explicit-EMPTY token/permutation encoding."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import qiskit
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/qaoa_placement/mixer_reduction"
for path in (ROOT / "experiments/qaoa_placement", OUT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from occupant_circuit import two_qubit_depth  # noqa: E402
from placement_core import hpwl, random_problem  # noqa: E402
from token_permutation_encoding import (  # noqa: E402
    aggregate_probabilities_by_assignment,
    assignment_to_token_sites,
    empty_token_count,
    expected_empty_degeneracy,
    legal_token_permutations,
    real_assignment_key,
    token_data_qubits,
    token_graph_edges,
    token_mixer_circuit,
    token_permutation_graph_metrics,
    token_phase_circuit,
    token_phase_terms,
    token_qaoa_circuit,
    token_register_width,
    token_sites_to_assignment,
    token_sites_to_bits,
    valid_permutation_probability,
)


BASIS_GATES = ("rz", "sx", "x", "cx")
BACKEND = "synthetic_heavy_hex_d5"
GAMMA = 0.17
BETA = 0.23
ROUTE_SEED = 118
GRAPHS = ("complete", "line", "ring", "empty_star", "real_empty_priority")
OCCUPANT_P1_CX = 3768
OCCUPANT_P1_DEPTH = 2707
OCCUPANT_P1_DEPTH_PARETO_CX = 3838
OCCUPANT_P1_DEPTH_PARETO_DEPTH = 2530
OCCUPANT_P2_CX = 7619
OCCUPANT_P2_DEPTH = 4905


def canonical_problem():
    return random_problem(
        seed=12000 + 100 * 4 + 6,
        num_cells=4,
        width=3,
        height=2,
        edge_probability=0.7,
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coupling() -> CouplingMap:
    return CouplingMap.from_heavy_hex(5)


def circuit_metrics(circuit) -> dict[str, object]:
    lowered = transpile(circuit, basis_gates=["u3", "cx"], optimization_level=3, seed_transpiler=123)
    ops = lowered.count_ops()
    return {
        "logical_qubits": circuit.num_qubits,
        "unconstrained_cx": int(ops.get("cx", 0)),
        "unconstrained_depth": int(lowered.depth()),
        "unconstrained_two_qubit_depth": two_qubit_depth(lowered),
    }


def route(circuit) -> dict[str, object]:
    start = time.time()
    routed = transpile(
        circuit,
        basis_gates=list(BASIS_GATES),
        coupling_map=coupling(),
        optimization_level=3,
        seed_transpiler=ROUTE_SEED,
        layout_method="sabre",
        routing_method="sabre",
    )
    seconds = time.time() - start
    ops = routed.count_ops()
    return {
        "status": "ok",
        "routed_cx": int(ops.get("cx", 0)),
        "routed_depth": int(routed.depth()),
        "routed_two_qubit_depth": two_qubit_depth(routed),
        "routed_one_qubit_count": int(sum(count for name, count in ops.items() if name != "cx")),
        "physical_qubits_used": int(routed.num_qubits),
        "runtime_seconds": seconds,
    }


def connectivity_rows(problem) -> list[dict[str, object]]:
    return [
        {
            **token_permutation_graph_metrics(problem, graph),
            "cells": len(problem.cells),
            "sites": len(problem.sites),
            "real_placements": math.perm(len(problem.sites), len(problem.cells)),
        }
        for graph in GRAPHS
    ]


def equivalence_rows(problem) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    terms, global_phase = token_phase_terms(problem, GAMMA)
    checked = 0
    max_error = 0.0
    for token_sites in legal_token_permutations(problem):
        assignment = token_sites_to_assignment(problem, token_sites)
        assert assignment is not None
        bits = token_sites_to_bits(problem, token_sites)
        phase = float(global_phase)
        for term in terms:
            sign = 1.0
            for qubit in term["support"]:
                sign *= -1.0 if bits[int(qubit)] else 1.0
            phase += float(term["coefficient"]) * sign
        expected_phase = -float(GAMMA) * float(hpwl(problem, assignment))
        max_error = max(max_error, float(abs(phase - expected_phase)))
        checked += 1
    rows.append(
        {
            "validation": "phase_coefficients_match_hpwl_all_legal_token_permutations",
            "token_permutations_checked": checked,
            "max_phase_error": max_error,
            "passes": bool(max_error < 1e-10),
        }
    )
    return rows


def resource_and_routing_rows(problem) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    resources: list[dict[str, object]] = []
    routing: list[dict[str, object]] = []
    terms, _global = token_phase_terms(problem, GAMMA)
    phase = token_phase_circuit(problem, GAMMA)
    phase_metrics = circuit_metrics(phase)
    phase_route = route(phase)
    resources.append(
        {
            "component": "phase",
            "graph": "",
            "p": 1,
            "data_qubits": token_data_qubits(problem),
            "ancillas": 0,
            "total_qubits": phase.num_qubits,
            "parity_terms": len(terms),
            "token_edges": "",
            **phase_metrics,
        }
    )
    routing.append({"component": "phase", "graph": "", "p": 1, **phase_route})

    best_p1: tuple[str, int] | None = None
    for graph in GRAPHS:
        mixer = token_mixer_circuit(problem, BETA, graph, reps=1)
        mixer_metrics = circuit_metrics(mixer)
        mixer_route = route(mixer)
        resources.append(
            {
                "component": "mixer",
                "graph": graph,
                "p": 1,
                "data_qubits": token_data_qubits(problem),
                "ancillas": 1,
                "total_qubits": mixer.num_qubits,
                "parity_terms": "",
                "token_edges": len(token_graph_edges(problem, graph)),
                **mixer_metrics,
            }
        )
        routing.append({"component": "mixer", "graph": graph, "p": 1, **mixer_route})

        full = token_qaoa_circuit(problem, GAMMA, BETA, graph, p=1)
        full_metrics = circuit_metrics(full)
        full_route = route(full)
        resources.append(
            {
                "component": "full_qaoa",
                "graph": graph,
                "p": 1,
                "data_qubits": token_data_qubits(problem),
                "ancillas": 1,
                "total_qubits": full.num_qubits,
                "parity_terms": len(terms),
                "token_edges": len(token_graph_edges(problem, graph)),
                **full_metrics,
            }
        )
        routing.append({"component": "full_qaoa", "graph": graph, "p": 1, **full_route})
        if best_p1 is None or int(full_route["routed_cx"]) < best_p1[1]:
            best_p1 = (graph, int(full_route["routed_cx"]))

    # p=2 only for the best p=1 graph when it beats the current p=1 occupant route.
    if best_p1 and best_p1[1] < OCCUPANT_P1_CX:
        graph = best_p1[0]
        full = token_qaoa_circuit(problem, GAMMA, BETA, graph, p=2)
        full_metrics = circuit_metrics(full)
        full_route = route(full)
        resources.append(
            {
                "component": "full_qaoa",
                "graph": graph,
                "p": 2,
                "data_qubits": token_data_qubits(problem),
                "ancillas": 1,
                "total_qubits": full.num_qubits,
                "parity_terms": len(terms) * 2,
                "token_edges": len(token_graph_edges(problem, graph)) * 2,
                **full_metrics,
            }
        )
        routing.append({"component": "full_qaoa", "graph": graph, "p": 2, **full_route})
    return resources, routing


def quality_rows(problem) -> list[dict[str, object]]:
    assignments = []
    for token_sites in legal_token_permutations(problem):
        assignment = token_sites_to_assignment(problem, token_sites)
        if assignment is None:
            continue
        key = real_assignment_key(problem, assignment)
        if key not in {real_assignment_key(problem, item) for item in assignments}:
            assignments.append(assignment)
    costs = {real_assignment_key(problem, assignment): hpwl(problem, assignment) for assignment in assignments}
    opt = min(costs.values())
    init_assignment = {cell: idx for idx, cell in enumerate(problem.cells)}
    init_cost = hpwl(problem, init_assignment)
    rows: list[dict[str, object]] = []
    for graph in GRAPHS:
        circuit = token_qaoa_circuit(problem, GAMMA, BETA, graph, p=1, include_initial_state=True, initial_assignment=init_assignment)
        state = Statevector.from_instruction(circuit)
        probs = aggregate_probabilities_by_assignment(problem, state)
        feasibility = valid_permutation_probability(problem, state)
        optimal_probability = sum(prob for key, prob in probs.items() if abs(costs[key] - opt) < 1e-9)
        beat_initial_probability = sum(prob for key, prob in probs.items() if costs[key] < init_cost - 1e-9)
        useful = 1000.0 * beat_initial_probability
        sorted_costs = sorted((costs[key], prob) for key, prob in probs.items())
        cvar_mass = 0.0
        cvar_cost = 0.0
        for cost, prob in sorted_costs:
            take = min(prob, 0.25 - cvar_mass)
            if take <= 0:
                break
            cvar_cost += take * cost
            cvar_mass += take
        cvar = cvar_cost / cvar_mass if cvar_mass > 0 else float("nan")
        nonzero_per_assignment = sum(1 for prob in probs.values() if prob > 1e-12)
        rows.append(
            {
                "graph": graph,
                "p": 1,
                "parameters": f"gamma={GAMMA}, beta={BETA}",
                "feasibility_probability": feasibility,
                "real_assignment_probability_mass": sum(probs.values()),
                "optimal_probability": optimal_probability,
                "probability_beating_initial": beat_initial_probability,
                "useful_feasible_samples_per_1000": useful,
                "best_shot_ratio": min(cost for key, cost in costs.items() if probs.get(key, 0.0) > 1e-12) / opt,
                "cvar_025_cost": cvar,
                "exact_optimum_cost": opt,
                "initial_cost": init_cost,
                "empty_token_degeneracy": expected_empty_degeneracy(problem),
                "aggregated_real_assignments_with_support": nonzero_per_assignment,
                "optimum_preserved_after_empty_aggregation": optimal_probability > 1e-12,
            }
        )
    rows.append(
        {
            "graph": "occupant_register_reference",
            "p": 1,
            "parameters": "not rerun; current architecture too large for full statevector in this pass",
            "feasibility_probability": 1.0,
            "real_assignment_probability_mass": "",
            "optimal_probability": "",
            "probability_beating_initial": "",
            "useful_feasible_samples_per_1000": "",
            "best_shot_ratio": "",
            "cvar_025_cost": "",
            "exact_optimum_cost": opt,
            "initial_cost": init_cost,
            "empty_token_degeneracy": 1,
            "aggregated_real_assignments_with_support": "",
            "optimum_preserved_after_empty_aggregation": "",
        }
    )
    return rows


def write_reports(connectivity, resources, routing, quality, equivalence) -> None:
    full_routes = [row for row in routing if row["component"] == "full_qaoa" and int(row["p"]) == 1]
    best_p1 = min(full_routes, key=lambda row: int(row["routed_cx"]))
    best_quality = max((row for row in quality if row["graph"] != "occupant_register_reference"), key=lambda row: float(row["optimal_probability"]))
    connected = all(row["connected"] for row in connectivity)
    exact = all(row["passes"] for row in equivalence)
    p2_rows = [row for row in routing if row["component"] == "full_qaoa" and int(row["p"]) == 2]
    p2_text = "not attempted" if not p2_rows else f"{p2_rows[0]['graph']}: {p2_rows[0]['routed_cx']} CX / depth {p2_rows[0]['routed_depth']}"
    if connected and exact and int(best_p1["routed_cx"]) < 3000 and float(best_quality["optimal_probability"]) > 1e-6:
        decision = "Outcome A: Full-placement token/permutation encoding is promising"
        recommendation = "Promote token/permutation encoding to a bounded quality and parameter-validation branch."
    elif connected and exact and int(best_p1["routed_cx"]) < OCCUPANT_P1_CX:
        decision = "Outcome B: Cheap but quality weak"
        recommendation = "Investigate mixer graph, initialization, and EMPTY-token symmetry handling before promotion."
    elif connected and exact:
        decision = "Outcome C: Connected but routes poorly"
        recommendation = "Keep fixed-site cell-to-site as the cheaper branch and do not promote token/permutation yet."
    else:
        decision = "Outcome D: Exactness/connectivity issue"
        recommendation = "Fix permutation encoding or phase exactness before any routing claims."
    summary = [
        "# Token/Permutation Encoding Summary",
        "",
        "Architecture: explicit real-cell and distinguishable EMPTY tokens, one site-index register per token.",
        f"Canonical data qubits: {token_data_qubits(canonical_problem())}; total with mixer ancilla: {token_data_qubits(canonical_problem()) + 1}.",
        f"EMPTY-token degeneracy: {expected_empty_degeneracy(canonical_problem())}.",
        "",
        "## Best Routed Results",
        "",
        f"- best p=1: `{best_p1['graph']}` with {best_p1['routed_cx']} CX / depth {best_p1['routed_depth']}.",
        f"- p=2: {p2_text}.",
        f"- occupant p=1 reference: {OCCUPANT_P1_CX} CX / depth {OCCUPANT_P1_DEPTH}.",
        f"- occupant p=1 low-depth reference: {OCCUPANT_P1_DEPTH_PARETO_CX} CX / depth {OCCUPANT_P1_DEPTH_PARETO_DEPTH}.",
        "",
        "## Quality Smoke",
        "",
        f"- best fixed-parameter optimal probability: `{best_quality['graph']}` with {float(best_quality['optimal_probability']):.6f}.",
        f"- best fixed-parameter useful feasible / 1000: {float(best_quality['useful_feasible_samples_per_1000']):.2f}.",
        "- occupant-register reference was not rerun by full statevector because the current primary circuit is 50 qubits.",
        "",
        f"Decision: **{decision}**.",
        "",
        recommendation,
    ]
    (OUT / "token_permutation_summary.md").write_text("\n".join(summary) + "\n")
    (OUT / "token_permutation_decision.md").write_text(
        f"""# Token/Permutation Encoding Decision

Decision: **{decision}**.

{recommendation}

## Interpretation

The token/permutation encoding fixes the disconnected occupied-site-set problem
of swap-only cell-to-site encoding by including explicit EMPTY tokens.  Legal
permutation subspace connectivity is restored for connected token-swap graphs,
and the exact phase separator remains cheap because real-cell token registers
directly hold site labels.

## Exact Next Step

Run a bounded parameter and mixer-graph validation for token/permutation QAOA:
use the same six frozen one-per-family instances, p=1 and p=2 only, exact
statevector/reduced permutation simulation, and compare aggregated real-placement
quality against the occupant-register bounded reference.
"""
    )


def environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "qiskit": qiskit.__version__,
        "numpy": np.__version__,
        "backend": BACKEND,
        "coupling": "CouplingMap.from_heavy_hex(5)",
        "physical_qubits": coupling().size(),
        "basis_gates": BASIS_GATES,
        "route_seed": ROUTE_SEED,
        "graphs": GRAPHS,
        "gamma": GAMMA,
        "beta": BETA,
        "command": "PYTHONPATH=experiments/qaoa_placement:experiments/qaoa_placement/mixer_reduction MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python experiments/qaoa_placement/mixer_reduction/run_token_permutation_study.py",
    }


def main() -> None:
    problem = canonical_problem()
    connectivity = connectivity_rows(problem)
    equivalence = equivalence_rows(problem)
    resources, routing = resource_and_routing_rows(problem)
    quality = quality_rows(problem)
    write_csv(OUT / "token_permutation_connectivity.csv", connectivity)
    write_csv(OUT / "token_permutation_equivalence.csv", equivalence)
    write_csv(OUT / "token_permutation_resource_estimates.csv", resources)
    write_csv(OUT / "token_permutation_routing_results.csv", routing)
    write_csv(OUT / "token_permutation_quality_results.csv", quality)
    (OUT / "token_permutation_environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    write_reports(connectivity, resources, routing, quality, equivalence)


if __name__ == "__main__":
    main()
