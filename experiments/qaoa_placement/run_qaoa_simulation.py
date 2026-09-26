"""Run an ideal-simulator QAOA placement experiment.

This script intentionally avoids IBM hardware/Runtime. It produces local
simulation artifacts that can be shown as the first QEDA algorithm-design
result:

- exact brute-force placement baseline,
- random feasible-placement baseline,
- QAOA p=1 and p=2 convergence/sampling results,
- circuit metrics and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from placement_core import (
    Assignment,
    PlacementProblem,
    assignment_to_bits,
    bits_to_assignment,
    bits_to_int,
    brute_force,
    build_qubo,
    default_problem,
    feasible_assignments,
    hpwl,
    int_to_bits,
    ising_terms,
    qubo_energy,
    random_baseline,
    variable_index,
)


def sparse_pauli_from_ising(num_qubits: int, h: Mapping[int, float], j: Mapping[Tuple[int, int], float]) -> SparsePauliOp:
    terms = []

    for idx, coeff in h.items():
        label = ["I"] * num_qubits
        label[num_qubits - 1 - idx] = "Z"
        terms.append(("".join(label), coeff))

    for (left, right), coeff in j.items():
        label = ["I"] * num_qubits
        label[num_qubits - 1 - left] = "Z"
        label[num_qubits - 1 - right] = "Z"
        terms.append(("".join(label), coeff))

    return SparsePauliOp.from_list(terms)


def cell_xy_mixer_operator(problem: PlacementProblem) -> SparsePauliOp:
    """Build sum_c sum_s<t (XX + YY) over each cell's one-hot site register."""

    terms = []
    num_sites = len(problem.sites)
    num_qubits = problem.num_variables

    for cell_idx in range(len(problem.cells)):
        for left_site in range(num_sites):
            for right_site in range(left_site + 1, num_sites):
                left = variable_index(cell_idx, left_site, num_sites)
                right = variable_index(cell_idx, right_site, num_sites)
                for pauli in ("X", "Y"):
                    label = ["I"] * num_qubits
                    label[num_qubits - 1 - left] = pauli
                    label[num_qubits - 1 - right] = pauli
                    terms.append(("".join(label), 1.0))

    return SparsePauliOp.from_list(terms)


def feasible_initial_state(problem: PlacementProblem) -> QuantumCircuit:
    """Prepare the first brute-force feasible assignment as a basis state."""

    assignment = next(feasible_assignments(problem))
    bits = assignment_to_bits(problem, assignment)
    circuit = QuantumCircuit(problem.num_variables)
    for idx, bit in enumerate(bits):
        if bit:
            circuit.x(idx)
    return circuit


def ising_energy_from_state_index(state_index: int, constant: float, h: Mapping[int, float], j: Mapping[Tuple[int, int], float], num_qubits: int) -> float:
    bits = int_to_bits(state_index, num_qubits)
    z = [1 - 2 * bit for bit in bits]
    total = constant
    for idx, coeff in h.items():
        total += coeff * z[idx]
    for (left, right), coeff in j.items():
        total += coeff * z[left] * z[right]
    return float(total)


def energy_spectrum(constant: float, h: Mapping[int, float], j: Mapping[Tuple[int, int], float], num_qubits: int) -> np.ndarray:
    return np.asarray(
        [
            ising_energy_from_state_index(state_index, constant, h, j, num_qubits)
            for state_index in range(2**num_qubits)
        ],
        dtype=float,
    )


def apply_mixer(state: np.ndarray, beta: float, num_qubits: int) -> None:
    """Apply exp(-i beta sum X_i) in place."""

    cos = np.cos(beta)
    minus_i_sin = -1j * np.sin(beta)

    for qubit in range(num_qubits):
        step = 1 << qubit
        block = step << 1
        for start in range(0, state.size, block):
            left = slice(start, start + step)
            right = slice(start + step, start + block)
            a = state[left].copy()
            b = state[right].copy()
            state[left] = cos * a + minus_i_sin * b
            state[right] = minus_i_sin * a + cos * b


def qaoa_probabilities(params: Sequence[float], energies: np.ndarray, reps: int, num_qubits: int) -> np.ndarray:
    state = np.full(2**num_qubits, 1 / np.sqrt(2**num_qubits), dtype=np.complex128)
    for layer in range(reps):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        state *= np.exp(-1j * gamma * energies)
        apply_mixer(state, beta, num_qubits)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def constrained_basis(problem: PlacementProblem) -> Tuple[List[Assignment], np.ndarray, Dict[int, int]]:
    assignments = []
    state_indices = []
    for assignment in feasible_cell_assignments(problem):
        bits = assignment_to_bits(problem, assignment)
        state_indices.append(bits_to_int(bits))
        assignments.append(assignment)
    index_by_state = {state_index: idx for idx, state_index in enumerate(state_indices)}
    return assignments, np.asarray(state_indices, dtype=int), index_by_state


def assignment_key(problem: PlacementProblem, assignment: Mapping[str, int]) -> Tuple[int, ...]:
    return tuple(assignment[cell] for cell in problem.cells)


def feasible_cell_assignments(problem: PlacementProblem) -> Iterable[Assignment]:
    """Assignments with exactly one site per cell, allowing site collisions."""

    num_sites = len(problem.sites)
    total = num_sites ** len(problem.cells)
    for value in range(total):
        remaining = value
        assignment: Assignment = {}
        for cell in problem.cells:
            assignment[cell] = remaining % num_sites
            remaining //= num_sites
        yield assignment


def constrained_cost_energies(problem: PlacementProblem, model) -> Tuple[List[Assignment], np.ndarray, np.ndarray, Dict[int, int]]:
    assignments, state_indices, index_by_state = constrained_basis(problem)
    energies = np.asarray(
        [qubo_energy(model, assignment_to_bits(problem, assignment)) for assignment in assignments],
        dtype=float,
    )
    return assignments, state_indices, energies, index_by_state


def collision_free_basis(problem: PlacementProblem) -> Tuple[List[Assignment], Dict[Tuple[int, ...], int]]:
    assignments = list(feasible_assignments(problem))
    index_by_key = {
        assignment_key(problem, assignment): idx
        for idx, assignment in enumerate(assignments)
    }
    return assignments, index_by_key


def collision_free_cost_energies(problem: PlacementProblem, model) -> Tuple[List[Assignment], np.ndarray, Dict[Tuple[int, ...], int]]:
    assignments, index_by_key = collision_free_basis(problem)
    energies = np.asarray(
        [qubo_energy(model, assignment_to_bits(problem, assignment)) for assignment in assignments],
        dtype=float,
    )
    return assignments, energies, index_by_key


def apply_cell_xy_mixer_subspace(state: np.ndarray, beta: float, problem: PlacementProblem, index_by_state: Mapping[int, int]) -> None:
    """Apply an XY-style site-swap mixer inside each cell register.

    Each term mixes two one-hot site states for the same cell. The pairwise
    update is the two-level action of exp(-i beta (XX + YY)) on the constrained
    subspace. Terms are applied in a fixed product order, matching a practical
    Trotterized custom mixer layer.
    """

    num_sites = len(problem.sites)
    theta = 2.0 * beta
    cos = np.cos(theta)
    minus_i_sin = -1j * np.sin(theta)

    for cell_idx, cell in enumerate(problem.cells):
        for left_site in range(num_sites):
            for right_site in range(left_site + 1, num_sites):
                visited = set()
                for basis_pos, assignment in enumerate(feasible_cell_assignments(problem)):
                    if basis_pos in visited:
                        continue
                    if assignment[cell] != left_site:
                        continue
                    swapped = dict(assignment)
                    swapped[cell] = right_site
                    left_state = bits_to_int(assignment_to_bits(problem, assignment))
                    right_state = bits_to_int(assignment_to_bits(problem, swapped))
                    left_pos = index_by_state[left_state]
                    right_pos = index_by_state[right_state]
                    visited.add(left_pos)
                    visited.add(right_pos)
                    a = state[left_pos]
                    b = state[right_pos]
                    state[left_pos] = cos * a + minus_i_sin * b
                    state[right_pos] = minus_i_sin * a + cos * b


def collision_free_swap_edges(problem: PlacementProblem, assignments: Sequence[Assignment], index_by_key: Mapping[Tuple[int, ...], int]) -> List[Tuple[int, int]]:
    """Return unique feasible-placement transitions induced by assignment swaps.

    A placement is treated as a matching from real cells to sites. If there are
    more sites than cells, one dummy empty occupant is included so swaps can move
    a real cell into an empty site while preserving collision freedom.
    """

    edges = set()
    cells = list(problem.cells)
    num_sites = len(problem.sites)

    for left_idx, assignment in enumerate(assignments):
        occupied = {assignment[cell]: cell for cell in cells}
        empty_sites = [site for site in range(num_sites) if site not in occupied]
        occupants = cells + [f"EMPTY_{site}" for site in empty_sites]

        for pos, left_occ in enumerate(occupants):
            for right_occ in occupants[pos + 1 :]:
                moved = dict(assignment)

                if left_occ.startswith("EMPTY_") and right_occ.startswith("EMPTY_"):
                    continue
                if left_occ.startswith("EMPTY_"):
                    empty_site = int(left_occ.split("_", 1)[1])
                    moved[right_occ] = empty_site
                elif right_occ.startswith("EMPTY_"):
                    empty_site = int(right_occ.split("_", 1)[1])
                    moved[left_occ] = empty_site
                else:
                    moved[left_occ], moved[right_occ] = moved[right_occ], moved[left_occ]

                right_idx = index_by_key[assignment_key(problem, moved)]
                if left_idx != right_idx:
                    edges.add((left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx))

    return sorted(edges)


def apply_swap_mixer_subspace(state: np.ndarray, beta: float, edges: Sequence[Tuple[int, int]]) -> None:
    """Apply a product of two-level swaps over feasible placement states."""

    cos = np.cos(beta)
    minus_i_sin = -1j * np.sin(beta)

    for left_pos, right_pos in edges:
        a = state[left_pos]
        b = state[right_pos]
        state[left_pos] = cos * a + minus_i_sin * b
        state[right_pos] = minus_i_sin * a + cos * b


def constrained_qaoa_probabilities(params: Sequence[float], energies: np.ndarray, reps: int, problem: PlacementProblem, index_by_state: Mapping[int, int]) -> np.ndarray:
    assignments = list(feasible_cell_assignments(problem))
    initial_assignment = next(feasible_assignments(problem))
    initial_state_index = bits_to_int(assignment_to_bits(problem, initial_assignment))
    initial_pos = index_by_state[initial_state_index]

    state = np.zeros(len(assignments), dtype=np.complex128)
    state[initial_pos] = 1.0
    for layer in range(reps):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        state *= np.exp(-1j * gamma * energies)
        apply_cell_xy_mixer_subspace(state, beta, problem, index_by_state)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def collision_free_qaoa_probabilities(params: Sequence[float], energies: np.ndarray, reps: int, initial_pos: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    state = np.zeros(len(energies), dtype=np.complex128)
    state[initial_pos] = 1.0
    for layer in range(reps):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]
        state *= np.exp(-1j * gamma * energies)
        apply_swap_mixer_subspace(state, beta, edges)
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def expectation_from_probs(probs: np.ndarray, energies: np.ndarray) -> float:
    return float(np.dot(probs, energies))


def sampled_distribution(probs: np.ndarray, shots: int, seed: int) -> Dict[int, int]:
    rng = np.random.default_rng(seed)
    states = np.arange(len(probs))
    samples = rng.choice(states, size=shots, p=probs)
    values, counts = np.unique(samples, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def best_feasible_sample(problem: PlacementProblem, counts: Mapping[int, int]) -> Dict[str, object]:
    model = build_qubo(problem)
    best = None
    feasible_count = 0

    for state_index, count in counts.items():
        bits = int_to_bits(state_index, problem.num_variables)
        assignment = bits_to_assignment(problem, bits)
        if assignment is None:
            continue
        feasible_count += count
        cost = hpwl(problem, assignment)
        energy = qubo_energy(model, bits)
        candidate = {
            "state_index": state_index,
            "count": count,
            "assignment": assignment,
            "hpwl": cost,
            "qubo_energy": energy,
        }
        if best is None or cost < best["hpwl"] or (cost == best["hpwl"] and count > best["count"]):
            best = candidate

    return {
        "best": best,
        "feasible_shots": feasible_count,
        "total_shots": sum(counts.values()),
        "feasible_fraction": feasible_count / max(sum(counts.values()), 1),
    }


def best_feasible_subspace_sample(problem: PlacementProblem, assignments: Sequence[Assignment], counts: Mapping[int, int]) -> Dict[str, object]:
    model = build_qubo(problem)
    best = None
    feasible_count = 0

    for subspace_idx, count in counts.items():
        assignment = assignments[subspace_idx]
        if len(set(assignment.values())) != len(problem.cells):
            continue
        feasible_count += count
        bits = assignment_to_bits(problem, assignment)
        cost = hpwl(problem, assignment)
        candidate = {
            "subspace_index": subspace_idx,
            "count": count,
            "assignment": assignment,
            "hpwl": cost,
            "qubo_energy": qubo_energy(model, bits),
        }
        if best is None or cost < best["hpwl"] or (cost == best["hpwl"] and count > best["count"]):
            best = candidate

    return {
        "best": best,
        "feasible_shots": feasible_count,
        "total_shots": sum(counts.values()),
        "feasible_fraction": feasible_count / max(sum(counts.values()), 1),
    }


def best_collision_free_sample(problem: PlacementProblem, assignments: Sequence[Assignment], counts: Mapping[int, int]) -> Dict[str, object]:
    model = build_qubo(problem)
    best = None
    feasible_count = sum(counts.values())

    for subspace_idx, count in counts.items():
        assignment = assignments[subspace_idx]
        bits = assignment_to_bits(problem, assignment)
        cost = hpwl(problem, assignment)
        candidate = {
            "subspace_index": subspace_idx,
            "count": count,
            "assignment": assignment,
            "hpwl": cost,
            "qubo_energy": qubo_energy(model, bits),
        }
        if best is None or cost < best["hpwl"] or (cost == best["hpwl"] and count > best["count"]):
            best = candidate

    return {
        "best": best,
        "feasible_shots": feasible_count,
        "total_shots": sum(counts.values()),
        "feasible_fraction": 1.0,
    }


def circuit_metrics(circuit: QuantumCircuit) -> Dict[str, object]:
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "num_parameters": len(circuit.parameters),
        "operations": {name: int(count) for name, count in circuit.count_ops().items()},
    }


def transpiled_metrics(circuit: QuantumCircuit) -> Dict[str, object]:
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=42)
    transpiled = pm.run(circuit)
    metrics = circuit_metrics(transpiled)
    metrics["transpilation_backend"] = backend.name
    metrics["optimization_level"] = 3
    return metrics


def run_qaoa_reps(problem: PlacementProblem, reps: int, maxiter: int, shots: int, seed: int, include_transpile: bool) -> Dict[str, object]:
    model = build_qubo(problem)
    constant, h, j = ising_terms(model)
    energies = energy_spectrum(constant, h, j, model.num_variables)
    cost_operator = sparse_pauli_from_ising(model.num_variables, h, j)
    ansatz = QAOAAnsatz(cost_operator=cost_operator, reps=reps)

    history: List[float] = []

    def objective(params: Sequence[float]) -> float:
        probs = qaoa_probabilities(params, energies, reps, model.num_variables)
        value = expectation_from_probs(probs, energies)
        history.append(value)
        return value

    init_params = []
    for layer in range(reps):
        init_params.extend([math.pi / (layer + 1), math.pi / (2 * (layer + 1))])

    start = time.time()
    result = minimize(
        objective,
        np.asarray(init_params, dtype=float),
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.4, "tol": 1e-3},
    )
    elapsed = time.time() - start

    final_probs = qaoa_probabilities(result.x, energies, reps, model.num_variables)
    counts = sampled_distribution(final_probs, shots=shots, seed=seed + reps)
    best_sample = best_feasible_sample(problem, counts)

    return {
        "reps": reps,
        "success": bool(result.success),
        "message": str(result.message),
        "optimized_value": float(result.fun),
        "optimized_params": [float(value) for value in result.x],
        "iterations_recorded": len(history),
        "elapsed_seconds": elapsed,
        "history": history,
        "shots": shots,
        "sampling": best_sample,
        "logical_circuit": circuit_metrics(ansatz),
        "transpiled_circuit": transpiled_metrics(ansatz) if include_transpile else None,
    }


def run_constrained_qaoa_reps(problem: PlacementProblem, reps: int, maxiter: int, shots: int, seed: int, include_transpile: bool) -> Dict[str, object]:
    model = build_qubo(problem)
    constant, h, j = ising_terms(model)
    cost_operator = sparse_pauli_from_ising(model.num_variables, h, j)
    mixer_operator = cell_xy_mixer_operator(problem)
    initial_state = feasible_initial_state(problem)
    ansatz = QAOAAnsatz(
        cost_operator=cost_operator,
        mixer_operator=mixer_operator,
        initial_state=initial_state,
        reps=reps,
    )
    assignments, _state_indices, energies, index_by_state = constrained_cost_energies(problem, model)

    history: List[float] = []

    def objective(params: Sequence[float]) -> float:
        probs = constrained_qaoa_probabilities(params, energies, reps, problem, index_by_state)
        value = expectation_from_probs(probs, energies)
        history.append(value)
        return value

    init_params = []
    for layer in range(reps):
        init_params.extend([math.pi / (layer + 1), math.pi / (8 * (layer + 1))])

    start = time.time()
    result = minimize(
        objective,
        np.asarray(init_params, dtype=float),
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.2, "tol": 1e-3},
    )
    elapsed = time.time() - start

    final_probs = constrained_qaoa_probabilities(result.x, energies, reps, problem, index_by_state)
    counts = sampled_distribution(final_probs, shots=shots, seed=seed + 100 + reps)
    best_sample = best_feasible_subspace_sample(problem, assignments, counts)

    return {
        "method": "cell-local XY constrained mixer",
        "reps": reps,
        "success": bool(result.success),
        "message": str(result.message),
        "optimized_value": float(result.fun),
        "optimized_params": [float(value) for value in result.x],
        "iterations_recorded": len(history),
        "elapsed_seconds": elapsed,
        "history": history,
        "shots": shots,
        "sampling": best_sample,
        "subspace_size": len(assignments),
        "logical_circuit": circuit_metrics(ansatz),
        "transpiled_circuit": transpiled_metrics(ansatz) if include_transpile else None,
    }


def run_collision_free_qaoa_reps(problem: PlacementProblem, reps: int, maxiter: int, shots: int, seed: int) -> Dict[str, object]:
    model = build_qubo(problem)
    assignments, energies, index_by_key = collision_free_cost_energies(problem, model)
    edges = collision_free_swap_edges(problem, assignments, index_by_key)
    initial_assignment = next(feasible_assignments(problem))
    initial_pos = index_by_key[assignment_key(problem, initial_assignment)]

    history: List[float] = []

    def objective(params: Sequence[float]) -> float:
        probs = collision_free_qaoa_probabilities(params, energies, reps, initial_pos, edges)
        value = expectation_from_probs(probs, energies)
        history.append(value)
        return value

    init_params = []
    for layer in range(reps):
        init_params.extend([math.pi / (layer + 1), math.pi / (8 * (layer + 1))])

    start = time.time()
    result = minimize(
        objective,
        np.asarray(init_params, dtype=float),
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.2, "tol": 1e-3},
    )
    elapsed = time.time() - start

    final_probs = collision_free_qaoa_probabilities(result.x, energies, reps, initial_pos, edges)
    counts = sampled_distribution(final_probs, shots=shots, seed=seed + 200 + reps)
    best_sample = best_collision_free_sample(problem, assignments, counts)

    return {
        "method": "collision-free swap mixer",
        "reps": reps,
        "success": bool(result.success),
        "message": str(result.message),
        "optimized_value": float(result.fun),
        "optimized_params": [float(value) for value in result.x],
        "iterations_recorded": len(history),
        "elapsed_seconds": elapsed,
        "history": history,
        "shots": shots,
        "sampling": best_sample,
        "subspace_size": len(assignments),
        "mixer_edges": len(edges),
        "logical_circuit": {
            "num_qubits": problem.num_variables,
            "depth": None,
            "size": None,
            "num_parameters": 2 * reps,
            "operations": {
                "feasible-state-swap-edge": len(edges) * reps,
            },
            "note": "Feasible-subspace mixer simulation; gate-level decomposition is future work.",
        },
        "transpiled_circuit": None,
    }


def plot_convergence(results: Sequence[Mapping[str, object]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for result in results:
        history = result["history"]
        method = result.get("method", "standard X mixer")
        plt.plot(
            range(1, len(history) + 1),
            history,
            marker="o",
            linewidth=1.4,
            markersize=3,
            label=f"{method}, p={result['reps']}",
        )
    plt.xlabel("COBYLA objective evaluation")
    plt.ylabel("Expected QUBO energy")
    plt.title("QAOA convergence on toy placement QUBO")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_placement(problem: PlacementProblem, assignment: Mapping[str, int], title: str, out_path: Path) -> None:
    plt.figure(figsize=(4.5, 4.5))
    xs = [site[0] for site in problem.sites]
    ys = [site[1] for site in problem.sites]
    plt.scatter(xs, ys, s=500, facecolor="#f6f6f6", edgecolor="#222222", linewidth=1.2)

    for cell, site_idx in assignment.items():
        x, y = problem.sites[site_idx]
        plt.text(x, y, cell, ha="center", va="center", fontsize=16, weight="bold")

    for left, right, weight in problem.nets:
        left_site = problem.sites[assignment[left]]
        right_site = problem.sites[assignment[right]]
        plt.plot([left_site[0], right_site[0]], [left_site[1], right_site[1]], color="#4c78a8", alpha=0.4 + 0.08 * weight, linewidth=weight)

    plt.title(title)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlim(-0.35, 1.35)
    plt.ylim(-0.35, 1.35)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_summary_csv(rows: Iterable[Mapping[str, object]], out_path: Path) -> None:
    rows = list(rows)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="experiments/qaoa_placement/results")
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--transpile-metrics", action="store_true")
    parser.add_argument("--include-constrained", action="store_true")
    parser.add_argument("--include-collision-free", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problem = default_problem()
    exact = brute_force(problem)
    random_stats = random_baseline(problem)
    qaoa_results = [
        run_qaoa_reps(
            problem,
            reps=reps,
            maxiter=args.maxiter,
            shots=args.shots,
            seed=args.seed,
            include_transpile=args.transpile_metrics,
        )
        for reps in args.reps
    ]
    constrained_results = (
        [
            run_constrained_qaoa_reps(
                problem,
                reps=reps,
                maxiter=args.maxiter,
                shots=args.shots,
                seed=args.seed,
                include_transpile=args.transpile_metrics,
            )
            for reps in args.reps
        ]
        if args.include_constrained
        else []
    )
    collision_free_results = (
        [
            run_collision_free_qaoa_reps(
                problem,
                reps=reps,
                maxiter=args.maxiter,
                shots=args.shots,
                seed=args.seed,
            )
            for reps in args.reps
        ]
        if args.include_collision_free
        else []
    )

    exact_hpwl, exact_assignment = exact[0]

    summary_rows = [
        {
            "method": "Exact brute force",
            "hpwl": exact_hpwl,
            "feasible_fraction": 1.0,
            "details": str(exact_assignment),
        },
        {
            "method": "Random feasible baseline mean",
            "hpwl": random_stats["mean"],
            "feasible_fraction": 1.0,
            "details": f"best={random_stats['best']}, worst={random_stats['worst']}, samples={int(random_stats['samples'])}",
        },
    ]

    for result in qaoa_results:
        sample = result["sampling"]
        best = sample["best"]
        summary_rows.append(
            {
                "method": f"QAOA p={result['reps']} best feasible sample",
                "hpwl": None if best is None else best["hpwl"],
                "feasible_fraction": sample["feasible_fraction"],
                "details": "no feasible sample" if best is None else str(best["assignment"]),
            }
        )

    for result in constrained_results:
        sample = result["sampling"]
        best = sample["best"]
        summary_rows.append(
            {
                "method": f"Constrained XY QAOA p={result['reps']} best feasible sample",
                "hpwl": None if best is None else best["hpwl"],
                "feasible_fraction": sample["feasible_fraction"],
                "details": "no feasible sample" if best is None else str(best["assignment"]),
            }
        )

    for result in collision_free_results:
        sample = result["sampling"]
        best = sample["best"]
        summary_rows.append(
            {
                "method": f"Collision-free swap QAOA p={result['reps']} best feasible sample",
                "hpwl": None if best is None else best["hpwl"],
                "feasible_fraction": sample["feasible_fraction"],
                "details": "no feasible sample" if best is None else str(best["assignment"]),
            }
        )

    write_summary_csv(summary_rows, out_dir / "summary.csv")
    all_qaoa_results = qaoa_results + constrained_results + collision_free_results
    plot_convergence(all_qaoa_results, out_dir / "qaoa_convergence.png")
    plot_placement(problem, exact_assignment, f"Exact optimum HPWL={exact_hpwl}", out_dir / "placement_exact.png")

    for result in all_qaoa_results:
        best = result["sampling"]["best"]
        if best is not None:
            if result.get("method") == "collision-free swap mixer":
                prefix = "placement_collision_free_swap"
            elif result.get("method"):
                prefix = "placement_constrained_xy"
            else:
                prefix = "placement_qaoa"
            plot_placement(
                problem,
                best["assignment"],
                f"{result.get('method', 'standard X mixer')} p={result['reps']} best sample HPWL={best['hpwl']}",
                out_dir / f"{prefix}_p{result['reps']}.png",
            )

    full_report = {
        "problem": {
            "cells": problem.cells,
            "sites": problem.sites,
            "nets": problem.nets,
            "penalty": problem.penalty,
            "num_variables": problem.num_variables,
            "num_feasible_assignments": len(exact),
        },
        "exact": {"hpwl": exact_hpwl, "assignment": exact_assignment},
        "random_baseline": random_stats,
        "qaoa": qaoa_results,
        "constrained_qaoa": constrained_results,
        "collision_free_qaoa": collision_free_results,
        "summary_rows": summary_rows,
    }
    with (out_dir / "report.json").open("w") as handle:
        json.dump(full_report, handle, indent=2)

    print(f"Wrote results to {out_dir}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
