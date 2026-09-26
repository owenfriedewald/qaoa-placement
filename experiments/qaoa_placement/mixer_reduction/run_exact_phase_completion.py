"""Controlled exact phase-completion audit; no new QAOA optimization or hardware.

All 100 geometry/alignment cases from the archived resource study are reused.
Two policies are fixed before construction: zero padding and weighted L1
completion. Both receive identical symbolic gamma and compiler settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent):
    sys.path.insert(0, str(path))

from invariant_placement import distance_coefficients, token_phase, circuit_hash
from run_acm_confirmation import write_csv, sha
from run_token_geometry_padding_audit import geometry, fixed_six_net_problem


def repository_phase(problem, gamma, policy):
    """Deterministic mask order on both sides, with adjacent CX cancellation.

    Ordering is deliberately fixed; no geometry-specific synthesis search.
    """
    width = (len(problem.sites) - 1).bit_length()
    circuit = QuantumCircuit(len(problem.sites) * width)
    coeff = distance_coefficients(problem.sites, policy)
    labels = {cell: i for i, cell in enumerate(problem.cells)}
    tokens = []
    for u, v, weight in problem.nets:
        circuit.global_phase -= weight * coeff[0] * gamma
        qubits = tuple(range(labels[u] * width, (labels[u]+1) * width)) + tuple(
            range(labels[v] * width, (labels[v]+1) * width))
        for mask in range(1, len(coeff)):
            if abs(coeff[mask]) < 1e-10:
                continue
            support = [q for bit, q in enumerate(qubits) if (mask >> bit) & 1]
            target = support[-1]
            for control in support[:-1]:
                op = ("cx", control, target)
                if tokens and tokens[-1] == op: tokens.pop()
                else: tokens.append(op)
            tokens.append(("rz", target, 2 * weight * coeff[mask] * gamma))
            for control in reversed(support[:-1]):
                op = ("cx", control, target)
                if tokens and tokens[-1] == op: tokens.pop()
                else: tokens.append(op)
                
    for name, left, right in tokens:
        if name == "cx": circuit.cx(left, right)
        else: circuit.rz(right, left)
    return circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "exact_phase_completion_20260904")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    cases = [(m, family, r) for m in (6, 8, 9, 12, 16)
             for family in ("compact", "line", "l_shape", "sparse_scatter") for r in range(5)]
    if args.smoke:
        cases = cases[:1]
    protocol = dict(cases=cases, policies=["zero", "l1"], gamma="symbolic", ordering="repository mask order with adjacent CX cancellation",
                    compiler=dict(basis=["rz", "sx", "x", "cx"], optimization_level=3, seed=123),
                    equality="all m*m valid site pairs, including equal sites",
                    objective="sum_mask (0.01+2*max(popcount(mask)-1,0))*abs(c_mask); constant free",
                    claim="weighted L1 surrogate, not an optimum gate-count guarantee",
                    source_sha256={p.name: sha(p) for p in [Path(__file__), HERE / "invariant_placement.py",
                        HERE / "run_token_geometry_padding_audit.py"]})
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rows = []
    for m, family, repeat in cases:
        sites = geometry(family, m, repeat)
        problem = fixed_six_net_problem(sites)
        width = (m - 1).bit_length()
        valid = [a | (b << width) for b in range(m) for a in range(m)]
        coefficients = {}
        for policy in ("zero", "l1"):
            started = time.monotonic()
            coeff = distance_coefficients(sites, policy)
            coefficients[policy] = coeff.tolist()
            circuit = repository_phase(problem, Parameter("gamma"), policy)
            lowered = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"],
                                optimization_level=3, seed_transpiler=123)
            exact = np.array([abs(sites[a][0]-sites[b][0]) + abs(sites[a][1]-sites[b][1])
                              for b in range(m) for a in range(m)])
            residual = float(np.max(np.abs((hadamard(len(coeff)) @ coeff)[valid] - exact)))
            rows.append(dict(sites=m, family=family, repeat=repeat, policy=policy,
                             symbolic_parameters=circuit.num_parameters,
                             nonconstant_terms_per_net=int(np.count_nonzero(coeff[1:])),
                             raw_cx=circuit.count_ops().get("cx", 0),
                             logical_cx=lowered.count_ops().get("cx", 0),
                             logical_depth=lowered.depth(), legal_distance_max_error=residual,
                             circuit_sha256=circuit_hash(circuit), runtime_seconds=time.monotonic()-started))
        (args.out / f"coefficients_{m}_{family}_{repeat}.json").write_text(
            json.dumps(dict(sites=sites, coefficients=coefficients), indent=2) + "\n")
        write_csv(args.out / "run_level.csv", rows)
        print(f"finished phase {m} {family} {repeat}", flush=True)
    import importlib.metadata as metadata
    (args.out / "environment.json").write_text(json.dumps(dict(
        command=sys.argv, versions={p: metadata.version(p) for p in ["qiskit", "numpy", "scipy"]},
        protocol_sha256=sha(args.out / "protocol.json")), indent=2) + "\n")


if __name__ == "__main__":
    main()
