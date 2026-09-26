"""Input-aware placement circuits for the September ACM correction.

Historical builders are deliberately unchanged. These circuits are equivalent
on a declared basis initialization, not as full-space unitaries. Both methods
use beta[0], gamma[0], beta[1], ... (2p-1 parameters), include preparation,
omit the first cost phase, and exclude measurement.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from functools import lru_cache

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from scipy.linalg import hadamard
from scipy.optimize import linprog

from placement_core import PlacementProblem, QuboModel, build_qubo, ising_terms
from phase_separator_optimization import order_parity_terms
from token_permutation_encoding import prepare_token_permutation_state, token_data_qubits
from run_token_permutation_p3_quality_uplift import append_scheduled_mixer


def collision_only_qubo(problem: PlacementProblem) -> QuboModel:
    """Remove exactly-one-row penalties; retain the archive's 2 lambda collision convention."""
    original = build_qubo(problem)
    m, n = len(problem.sites), len(problem.cells)
    linear = dict(original.linear)
    quadratic = dict(original.quadratic)
    for u in range(n):
        for s in range(m):
            linear[u * m + s] += problem.penalty
        for s, t in itertools.combinations(range(m), 2):
            quadratic[u * m + s, u * m + t] -= 2 * problem.penalty
    return QuboModel(linear, quadratic, original.offset - n * problem.penalty, n * m)


@lru_cache(maxsize=256)
def distance_coefficients(sites: tuple[tuple[int, int], ...], completion: str = "zero") -> np.ndarray:
    """Walsh coefficients of distance, preserving every valid pair (including equal sites).

    l1 completion minimizes a weighted coefficient norm, NOT gate count.
    All invalid-code entries are free; legal entries are hard equality constraints.
    The result is independent of net weight and QAOA gamma.
    """
    m = len(sites)
    width = (m - 1).bit_length()
    dim = 1 << (2 * width)
    walsh = hadamard(dim).astype(float)
    indices = np.asarray([a | (b << width) for b in range(m) for a in range(m)])
    values = np.asarray([abs(sites[a][0] - sites[b][0]) + abs(sites[a][1] - sites[b][1])
                         for b in range(m) for a in range(m)], dtype=float)
    padded = np.zeros(dim)
    padded[indices] = values
    if completion == "zero" or m == 1 << width:
        coeff = walsh @ padded / dim
    elif completion == "l1":
        # Positive/negative coefficient parts; epsilon also penalizes one-body support.
        costs = np.asarray([0.01 + 2 * max(mask.bit_count() - 1, 0) for mask in range(dim)])
        costs[0] = 0
        a = walsh[indices]
        result = linprog(np.r_[costs, costs], A_eq=np.c_[a, -a], b_eq=values,
                         bounds=(0, None), method="highs")
        if not result.success:
            raise RuntimeError(f"Phase completion failed: {result.message}")
        coeff = result.x[:dim] - result.x[dim:]
    else:
        raise ValueError(completion)
    coeff[np.abs(coeff) < 1e-10] = 0
    if np.max(np.abs(walsh[indices] @ coeff - values)) > 1e-8:
        raise ValueError("Completed phase does not match legal distances")
    coeff.setflags(write=False)
    return coeff


def token_phase(problem: PlacementProblem, gamma, completion: str = "zero") -> QuantumCircuit:
    """Symbolic exact phase with the archive's deterministic beam ordering."""
    circuit = QuantumCircuit(token_data_qubits(problem))
    width = (len(problem.sites) - 1).bit_length()
    coeff = distance_coefficients(problem.sites, completion)
    labels = {cell: i for i, cell in enumerate(problem.cells)}
    terms = []
    constant = 0.0
    for net_id, (u, v, weight) in enumerate(problem.nets):
        qubits = tuple(range(labels[u] * width, (labels[u] + 1) * width)) + tuple(
            range(labels[v] * width, (labels[v] + 1) * width))
        constant += weight * coeff[0]
        for mask in range(1, len(coeff)):
            if coeff[mask] == 0:
                continue
            support = tuple(q for bit, q in enumerate(qubits) if (mask >> bit) & 1)
            terms.append(dict(term_id=len(terms), net_id=net_id, mask=mask, support=support,
                              support_size=len(support), coefficient=weight * coeff[mask]))
    ordered = order_parity_terms(terms, "beam_search")
    tokens = []
    for term in ordered:
        support = term["support"]
        target = support[-1]
        for control in support[:-1]:
            op = ("cx", control, target)
            if tokens and tokens[-1] == op:
                tokens.pop()
            else:
                tokens.append(op)
        tokens.append(("rz", target, 2 * term["coefficient"] * gamma))
        for control in reversed(support[:-1]):
            op = ("cx", control, target)
            if tokens and tokens[-1] == op:
                tokens.pop()
            else:
                tokens.append(op)
    for name, left, right in tokens:
        if name == "cx":
            circuit.cx(left, right)
        else:
            circuit.rz(right, left)
    circuit.global_phase = -constant * gamma
    return circuit


def placement_circuit(problem: PlacementProblem, method: str, reps: int = 3,
                      graph: str = "line", schedule: str = "rotating",
                      initial: dict[str, int] | None = None,
                      theta=None, completion: str = "zero") -> QuantumCircuit:
    if reps < 1:
        raise ValueError("reps must be positive")
    if initial is None:
        initial = dict(zip(problem.cells, range(len(problem.cells))))
    if (set(initial) != set(problem.cells) or len(set(initial.values())) != len(problem.cells)
            or any(not 0 <= s < len(problem.sites) for s in initial.values())):
        raise ValueError("initial must be an injective valid placement")
    theta = ParameterVector("theta", 2 * reps - 1) if theta is None else theta
    if len(theta) != 2 * reps - 1:
        raise ValueError("expected 2p-1 parameters")
    if method == "token":
        data = token_data_qubits(problem)
        circuit = QuantumCircuit(data + 1)
        prepare_token_permutation_state(circuit, problem, initial)
    elif method == "penalty_row_xy":
        circuit = QuantumCircuit(problem.num_variables)
        for u, cell in enumerate(problem.cells):
            circuit.x(u * len(problem.sites) + initial[cell])
        constant, h, j = ising_terms(collision_only_qubo(problem))
    else:
        raise ValueError(method)
    for layer in range(reps):
        beta = theta[2 * layer]
        if layer:
            gamma = theta[2 * layer - 1]
            if method == "token":
                circuit.compose(token_phase(problem, gamma, completion), range(data), inplace=True)
            else:
                circuit.global_phase -= constant * gamma
                for qubit, coeff in sorted(h.items()):
                    circuit.rz(2 * coeff * gamma, qubit)
                for (left, right), coeff in sorted(j.items()):
                    circuit.rzz(2 * coeff * gamma, left, right)
        if method == "token":
            append_scheduled_mixer(circuit, problem, beta, graph, schedule, layer)
        else:
            m = len(problem.sites)
            for u in range(len(problem.cells)):
                for left, right in itertools.combinations(range(m), 2):
                    circuit.rxx(2 * beta, u * m + left, u * m + right)
                    circuit.ryy(2 * beta, u * m + left, u * m + right)
    return circuit


def circuit_hash(circuit: QuantumCircuit) -> str:
    """Stable operation/parameter-name hash; random Parameter UUIDs are intentionally excluded."""
    payload = [circuit.num_qubits, str(circuit.global_phase), [
        [item.operation.name, [circuit.find_bit(q).index for q in item.qubits],
         [str(x) for x in item.operation.params]] for item in circuit.data]]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


class ReducedPlacement:
    """Vectorized ideal simulation; token m! states and Row-XY m**n states.

    Pair maps are cached once per instance. No quotient of EMPTY amplitudes is used.
    """
    def __init__(self, problem: PlacementProblem, method: str, graph="line", schedule="rotating"):
        self.problem, self.method, self.graph, self.schedule = problem, method, graph, schedule
        n, m = len(problem.cells), len(problem.sites)
        self.states = np.asarray(list(itertools.permutations(range(m))) if method == "token"
                                 else list(itertools.product(range(m), repeat=n)), dtype=np.int16)
        self.index = {tuple(row): i for i, row in enumerate(self.states)}
        real = self.states[:, :n]
        self.legal = np.asarray([len(set(row)) == n for row in real])
        self.costs = np.zeros(len(real))
        coords = np.asarray(problem.sites)
        labels = {c: i for i, c in enumerate(problem.cells)}
        for u, v, weight in problem.nets:
            self.costs += weight * np.abs(coords[real[:, labels[u]]] - coords[real[:, labels[v]]]).sum(axis=1)
        self.energies = self.costs.copy()
        for u, v in itertools.combinations(range(n), 2):
            self.energies += 2 * problem.penalty * (real[:, u] == real[:, v])
        self.exact = float(self.costs[self.legal].min())
        self.optimal = self.legal & np.isclose(self.costs, self.exact, atol=1e-9, rtol=0)
        self.pairs = {}
        if method == "token":
            from token_permutation_encoding import token_graph_edges
            for u, v in token_graph_edges(problem, graph):
                moved = self.states.copy()
                moved[:, [u, v]] = moved[:, [v, u]]
                self.pairs[u, v] = np.asarray([self.index[tuple(row)] for row in moved])

    def probabilities(self, theta, initial, reps=3):
        from token_permutation_encoding import assignment_to_token_sites
        from run_token_permutation_p3_quality_uplift import scheduled_token_edges
        key = assignment_to_token_sites(self.problem, initial) if self.method == "token" else tuple(
            initial[c] for c in self.problem.cells)
        state = np.zeros(len(self.states), dtype=complex)
        state[self.index[key]] = 1
        n, m = len(self.problem.cells), len(self.problem.sites)
        for layer in range(reps):
            if layer:
                state *= np.exp(-1j * theta[2 * layer - 1] * self.energies)
            beta = theta[2 * layer]
            if self.method == "token":
                for edge in scheduled_token_edges(self.problem, self.graph, self.schedule, layer):
                    state = np.cos(beta) * state - 1j * np.sin(beta) * state[self.pairs[edge]]
            else:
                unitary = np.eye(m, dtype=complex)
                for u, v in itertools.combinations(range(m), 2):
                    old = unitary[[u, v]].copy()
                    unitary[u] = np.cos(2 * beta) * old[0] - 1j * np.sin(2 * beta) * old[1]
                    unitary[v] = np.cos(2 * beta) * old[1] - 1j * np.sin(2 * beta) * old[0]
                tensor = state.reshape((m,) * n)
                for axis in range(n):
                    moved = np.moveaxis(tensor, axis, 0)
                    tensor = np.moveaxis((unitary @ moved.reshape(m, -1)).reshape(moved.shape), 0, axis)
                state = tensor.reshape(-1)
        probs = np.abs(state) ** 2
        if abs(probs.sum() - 1) > 1e-8:
            raise ValueError("Reduced simulation lost normalization")
        return probs / probs.sum()
