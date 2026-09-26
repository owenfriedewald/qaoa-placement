"""Prospectively frozen confirmation and symmetric compilation for ACM TQC.

Run freeze before any quality/route/phase command. Each command writes new
files and refuses to overwrite results. No provider connection is made.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import qiskit
import scipy
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
MIXER = Path(__file__).resolve().parent
QAOA = MIXER.parent
for directory in (MIXER, QAOA):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmark_suite import (BenchmarkSpec, manifest_payload, manifest_row, load_manifest,
                             problem_from_manifest)
from invariant_placement import ReducedPlacement, circuit_hash, placement_circuit, token_phase, distance_coefficients
from objective_functions import cvar_cost
from placement_core import PlacementProblem, hpwl, random_problem, brute_force
from run_robust_schedule_experiment import initialization_classes
from run_token_permutation_quality_validation import init_assignment
from run_token_permutation_p3_quality_uplift import initial_parameters

DEFAULT = MIXER / "acm_confirmation_20260904"
FROZEN = QAOA / "paper_suite_results/audit_061026/benchmark_manifest.json"
TOKEN_DEV = MIXER / "token_permutation_p3_parameter_results.csv"
PENALTY_DEV = MIXER / "penalty_lambda_sweep_v1/run_level.csv"
METHODS = ("token", "penalty_row_xy")
READOUTS = (1, 8, 32, 128, 512)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__), MIXER / "invariant_placement.py", QAOA / "benchmark_suite.py",
             QAOA / "placement_core.py", QAOA / "objective_functions.py",
             MIXER / "run_token_permutation_p3_quality_uplift.py",
             MIXER / "phase_separator_optimization.py", MIXER / "register_partial_swap.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def development_choice():
    candidates = itertools.product(("line", "ring"), ("rotating", "reversed", "empty_prioritized"))
    rows = [r for r in read_csv(TOKEN_DEV) if r["p"] == "3" and r["optimization_objective"] == "cvar_0.25"
            and r["parameter_protocol"] == "optimized"]
    scores = []
    for graph, schedule in candidates:
        group = [r for r in rows if r["graph"] == graph and r["mixer_schedule"] == schedule]
        scores.append((np.mean([float(r["optimal_probability"]) for r in group]),
                       np.mean([float(r["probability_beating_initial"]) for r in group]), graph, schedule))
    best = max(x[:2] for x in scores)
    graph, schedule = min(x[2:] for x in scores if x[:2] == best)
    penalty = read_csv(PENALTY_DEV)
    lambdas = sorted({float(r["penalty_lambda"]) for r in penalty})
    def score(value):
        group = [r for r in penalty if float(r["penalty_lambda"]) == value]
        return (np.mean([float(r["optimal_probability"]) for r in group]),
                np.mean([float(r["feasible_probability"]) for r in group]), -value)
    return graph, schedule, max(lambdas, key=score)


def freeze(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    graph, schedule, penalty = development_choice()
    families = [("sparse_grid", .35, 3, 2), ("dense_grid", .95, 3, 2),
                ("sparse_grid_line", .45, 6, 1), ("dense_grid_line", .95, 6, 1),
                ("sparse_grid_lshape", .45, 4, 3), ("dense_grid_lshape", .90, 4, 3)]
    rows = []
    for f, (family, density, width, height) in enumerate(families):
        for repeat in range(6):
            seed = 904000 + 100 * f + repeat
            row = manifest_row(BenchmarkSpec(family, seed, 4, width, height, density),
                               f"fresh4_{family}_{seed}")
            # Timings are provenance, not part of the deterministic instance specification.
            row.pop("exact_runtime_seconds", None)
            rows.append(row)
        for repeat in range(2):
            seed = 905000 + 100 * f + repeat
            base = random_problem(seed, 5, 7, 1, density)
            if family.endswith("_line"):
                sites = tuple((i, 0) for i in range(7))
            elif family.endswith("_lshape"):
                sites = ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3))
            else:
                sites = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2))
            problem = replace(base, sites=sites)
            exact = brute_force(problem)
            inits = [dict(init_mode=mode, init_seed=s, assignment=a, hpwl=hpwl(problem, a),
                          is_optimal=hpwl(problem, a) == exact[0][0])
                     for mode, s, a in initialization_classes(problem, 9000 + seed)]
            rows.append(dict(instance_id=f"fresh5_{family}_{seed}", family=family, seed=seed,
                             num_cells=5, num_sites=7, cells=problem.cells, geometry=sites,
                             nets=problem.nets, net_count=len(problem.nets), initializations=inits,
                             exact_hpwl=exact[0][0], degenerate_optima=sum(x[0] == exact[0][0] for x in exact)))
    protocol = dict(schema="acm-confirmation-v1", frozen_before_evaluation=True,
                    selection_data={str(TOKEN_DEV.relative_to(ROOT)): sha(TOKEN_DEV),
                                    str(PENALTY_DEV.relative_to(ROOT)): sha(PENALTY_DEV)},
                    token_graph=graph, token_schedule=schedule, penalty_lambda=penalty,
                    objective="analytical CVaR 0.25 of cost plus collision penalties", p=3,
                    optimizer="COBYLA", budget=200, rhobeg=.2, tol=.001,
                    optimizer_seeds=[11, 17, 23, 31], initializations=["deterministic", "poor", "random"],
                    parameter_policy="2p-1: beta0,gamma1,beta1,gamma2,beta2; matched starts",
                    circuit_policy="both symbolic, prepared, no first phase, no row penalties, no measurements",
                    primary_endpoint="instance mean unconditional optimum probability, token minus penalty",
                    secondary_endpoints=["legal mass", "improvement mass", "expected best feasible cost with initial fallback"],
                    readout_budgets=READOUTS, routing_seeds=list(range(101, 121)),
                    instances_4c6s=36, instances_5c7s=12, outcome_filters="none",
                    routing_cohorts=["frozen 32 4c6s line/rotating", "fresh 36 4c6s development-selected protocol"],
                    code_sha256=source_hashes())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest_payload(rows), indent=2) + "\n")
    print(json.dumps({k: protocol[k] for k in ["token_graph", "token_schedule", "penalty_lambda",
                                              "instances_4c6s", "instances_5c7s"]}), flush=True)


def greedy(problem, initial, budget):
    """Same strict best-neighbor policy as the paper; include initial query in the budget."""
    current = tuple(initial[c] for c in problem.cells)
    cost = hpwl(problem, initial)
    evaluations = 1
    n, m = len(problem.cells), len(problem.sites)
    while evaluations < budget:
        neighbors = set()
        for u in range(n):
            for site in range(m):
                moved = list(current)
                if site in current:
                    v = current.index(site)
                    moved[u], moved[v] = moved[v], moved[u]
                else:
                    moved[u] = site
                if tuple(moved) != current:
                    neighbors.add(tuple(moved))
        best, best_cost = current, cost
        for candidate in sorted(neighbors):
            if evaluations == budget:
                break
            value = hpwl(problem, dict(zip(problem.cells, candidate)))
            evaluations += 1
            if value < best_cost - 1e-12:
                best, best_cost = candidate, value
        if best_cost >= cost - 1e-12:
            break
        current, cost = best, best_cost
    return float(cost), evaluations


def expected_best(probs, costs, legal, initial_cost, shots):
    # The initial assignment is always available; invalid/worse samples cannot harm it.
    effective = np.where(legal, np.minimum(costs, initial_cost), initial_cost)
    values, reverse = np.unique(effective, return_inverse=True)
    mass = np.bincount(reverse, weights=probs, minlength=len(values))
    tail = np.r_[1.0, np.maximum(0, 1 - np.cumsum(mass))]
    return float(np.sum(values * (tail[:-1] ** shots - tail[1:] ** shots)))


def quality_case(instance, protocol):
    problem = replace(problem_from_manifest(instance), penalty=protocol["penalty_lambda"])
    models = {method: ReducedPlacement(problem, method, protocol["token_graph"], protocol["token_schedule"])
              for method in METHODS}
    rows, curves = [], []
    for init_mode in protocol["initializations"]:
        _, initial = init_assignment(instance, init_mode)
        initial_cost = hpwl(problem, initial)
        for method, model in models.items():
            for seed in protocol["optimizer_seeds"]:
                start = initial_parameters(3, seed)[1:]  # same five live coordinates on both sides
                started = time.monotonic()
                objective = lambda theta: cvar_cost(model.probabilities(theta, initial), model.energies, alpha=.25)
                result = minimize(objective, start, method="COBYLA",
                                  options=dict(maxiter=protocol["budget"], rhobeg=.2, tol=.001))
                probs = model.probabilities(result.x, initial)
                base = dict(instance_id=instance["instance_id"], family=instance["family"],
                            num_cells=len(problem.cells), num_sites=len(problem.sites), method=method,
                            init_mode=init_mode, optimizer_seed=seed)
                rows.append(dict(**base, exact_hpwl=model.exact, initial_hpwl=initial_cost,
                                 optimal_probability=float(probs[model.optimal].sum()),
                                 feasible_probability=float(probs[model.legal].sum()),
                                 improvement_probability=float(probs[model.legal & (model.costs < initial_cost - 1e-12)].sum()),
                                 objective_value=float(result.fun), evaluations=int(result.nfev),
                                 optimizer_success=bool(result.success), theta=json.dumps(result.x.tolist()),
                                 runtime_seconds=time.monotonic() - started))
                for shots in READOUTS:
                    curves.append(dict(**base, budget=shots,
                                       expected_best_hpwl=expected_best(probs, model.costs, model.legal, initial_cost, shots)))
        model = models["token"]
        uniform = np.ones(len(model.states)) / len(model.states)
        for budget in READOUTS:
            cost, actual = greedy(problem, initial, budget)
            for method, value in [("greedy", cost), ("uniform_feasible", expected_best(
                    uniform, model.costs, model.legal, initial_cost, budget))]:
                curves.append(dict(instance_id=instance["instance_id"], family=instance["family"],
                                   num_cells=len(problem.cells), num_sites=len(problem.sites), method=method,
                                   init_mode=init_mode, optimizer_seed="", budget=budget, expected_best_hpwl=value))
    return rows, curves


def route_case(instance, protocol, cohort, seeds):
    from qiskit import transpile, qpy
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    problem = replace(problem_from_manifest(instance), penalty=protocol["penalty_lambda"])
    graph, schedule = (("line", "rotating") if cohort == "frozen" else
                       (protocol["token_graph"], protocol["token_schedule"]))
    backend = FakeSherbrooke()
    rows = []
    for method in METHODS:
        circuit = placement_circuit(problem, method, graph=graph, schedule=schedule)
        logical = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3,
                            seed_transpiler=123)
        digest = circuit_hash(circuit)
        for seed in seeds:
            started = time.monotonic()
            routed = transpile(circuit, backend=backend, optimization_level=3, seed_transpiler=seed,
                               layout_method="sabre", routing_method="sabre")
            rows.append(dict(instance_id=instance["instance_id"], family=instance["family"],
                             num_cells=len(problem.cells), num_sites=len(problem.sites),
                             net_count=len(problem.nets), method=method, cohort=cohort,
                             graph=graph if method == "token" else "complete_per_row",
                             schedule=schedule if method == "token" else "lexicographic_site_pairs",
                             active_qubits=circuit.num_qubits, symbolic_parameters=circuit.num_parameters,
                             preparation=True, first_phase=False, row_penalties=False, measurements=False,
                             logical_cx=logical.count_ops().get("cx", 0), logical_depth=logical.depth(),
                             routing_seed=seed, routed_ecr=routed.count_ops().get("ecr", 0),
                             routed_depth=routed.depth(),
                             routed_two_qubit_depth=routed.depth(lambda inst: len(inst.qubits) == 2),
                             circuit_sha256=digest, routed_sha256=circuit_hash(routed),
                             operations=json.dumps(dict(routed.count_ops()), sort_keys=True),
                             runtime_seconds=time.monotonic() - started))
    return rows


def environment(out, command, protocol):
    import importlib.metadata as metadata
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    backend = FakeSherbrooke()
    edges = sorted(tuple(map(int, edge)) for edge in backend.coupling_map.get_edges())
    value = dict(command=sys.argv, python=platform.python_version(), platform=platform.platform(),
                 versions={pkg: metadata.version(pkg) for pkg in ["qiskit", "qiskit-ibm-runtime", "numpy", "scipy", "matplotlib"]},
                 code_sha256=source_hashes(), protocol_sha256=sha(out / "protocol.json"),
                 manifest_sha256=sha(out / "benchmark_manifest.json"), backend="FakeSherbrooke",
                 coupling_edges_sha256=hashlib.sha256(json.dumps(edges, separators=(",", ":")).encode()).hexdigest(),
                 backend_snapshot_sha256={name: sha(Path(backend.dirname) / name)
                                          for name in (backend.conf_filename, backend.props_filename)},
                 note="offline archived topology; no provider connection or hardware execution")
    (out / f"{command}_environment.json").write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "quality", "route"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cohort", choices=("frozen", "fresh"), default="frozen")
    parser.add_argument("--limit", type=int, help="smoke only; use a separate output directory")
    parser.add_argument("--seeds", type=int, nargs="+", help="smoke routing seeds; full default is frozen in protocol")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
        return
    protocol = json.loads((args.out / "protocol.json").read_text())
    # Freeze records the actual source; refuse silent implementation drift in a campaign.
    if protocol["code_sha256"] != source_hashes():
        raise RuntimeError("Code changed since freeze; freeze a new campaign instead of mixing results")
    rows = load_manifest(args.out / "benchmark_manifest.json")
    suffix = "quality" if args.command == "quality" else f"routing_{args.cohort}"
    path = args.out / f"{suffix}_run_level.csv"
    if path.exists():
        raise FileExistsError(path)
    if args.command == "route":
        rows = load_manifest(FROZEN) if args.cohort == "frozen" else rows
        rows = [r for r in rows if int(r["num_cells"]) == 4 and int(r["num_sites"]) == 6]
    if args.limit:
        rows = rows[:args.limit]
    environment(args.out, suffix, protocol)
    all_rows, curves = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(quality_case, row, protocol) if args.command == "quality" else
                 pool.submit(route_case, row, protocol, args.cohort, args.seeds or protocol["routing_seeds"]): row
                 for row in rows}
        for task in as_completed(tasks):
            result = task.result()
            if args.command == "quality":
                all_rows.extend(result[0]); curves.extend(result[1])
                write_csv(args.out / "quality_budget_curves.csv", sorted(curves, key=lambda r: (
                    r["instance_id"], r["method"], r["init_mode"], str(r["optimizer_seed"]), r["budget"])))
            else:
                all_rows.extend(result)
            all_rows.sort(key=lambda r: (r["instance_id"], r["method"], r.get("init_mode", ""),
                                        r.get("optimizer_seed", 0), r.get("routing_seed", 0)))
            write_csv(path, all_rows)
            print(f"finished {suffix} {tasks[task]['instance_id']} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
