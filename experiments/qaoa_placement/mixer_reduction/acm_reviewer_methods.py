"""Prospective reviewer controls; earlier frozen implementations are unchanged."""
from __future__ import annotations
import itertools
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from scipy.linalg import hadamard
from scipy.optimize import linprog
from invariant_placement import ReducedPlacement, collision_only_qubo, token_phase
from placement_core import ising_terms
from token_permutation_encoding import token_data_qubits, prepare_token_permutation_state
from run_token_permutation_p3_quality_uplift import scheduled_token_edges
from register_partial_swap import append_phase_estimation_partial_swap
from acm_gap_methods import gray_phase


def ring_edges(m):
    if m < 3:
        raise ValueError('ring requires at least three sites')
    return [(i, i+1) for i in range(m-1)] + [(m-1, 0)]


class RingPlacement(ReducedPlacement):
    def __init__(self, problem):
        super().__init__(problem, 'penalty_row_xy')

    def probabilities(self, theta, initial, reps=3):
        n, m = len(self.problem.cells), len(self.problem.sites)
        state = np.zeros(len(self.states), complex)
        state[self.index[tuple(initial[c] for c in self.problem.cells)]] = 1
        for layer in range(reps):
            if layer:
                state *= np.exp(-1j * theta[2*layer-1] * self.energies)
            beta = theta[2*layer]
            unitary = np.eye(m, dtype=complex)
            for u, v in ring_edges(m):
                old = unitary[[u, v]].copy()
                unitary[u] = np.cos(2*beta)*old[0] - 1j*np.sin(2*beta)*old[1]
                unitary[v] = np.cos(2*beta)*old[1] - 1j*np.sin(2*beta)*old[0]
            tensor = state.reshape((m,)*n)
            for axis in range(n):
                moved = np.moveaxis(tensor, axis, 0)
                tensor = np.moveaxis((unitary @ moved.reshape(m, -1)).reshape(moved.shape), 0, axis)
            state = tensor.reshape(-1)
        probs = abs(state)**2
        assert abs(probs.sum()-1) < 1e-8
        return probs/probs.sum()


def ring_circuit(problem, theta=None, initial=None):
    theta = ParameterVector('theta', 5) if theta is None else theta
    initial = dict(zip(problem.cells, range(len(problem.cells)))) if initial is None else initial
    m = len(problem.sites)
    c = QuantumCircuit(problem.num_variables)
    for u, cell in enumerate(problem.cells):
        c.x(u*m+initial[cell])
    constant, h, j = ising_terms(collision_only_qubo(problem))
    for layer in range(3):
        if layer:
            gamma = theta[2*layer-1]
            c.global_phase -= constant*gamma
            for q, coeff in sorted(h.items()): c.rz(2*coeff*gamma, q)
            for (u,v), coeff in sorted(j.items()): c.rzz(2*coeff*gamma, u, v)
        for u in range(len(problem.cells)):
            for a,b in ring_edges(m):
                c.rxx(2*theta[2*layer], u*m+a, u*m+b)
                c.ryy(2*theta[2*layer], u*m+a, u*m+b)
    return c


def dependency_stages(edges):
    """ASAP stages preserving every original shared-register dependency."""
    last = {}; stages = []
    for edge in edges:
        level = max((last.get(v, -1) for v in edge))+1
        while len(stages) <= level: stages.append([])
        stages[level].append(edge)
        for v in edge: last[v] = level
    return stages


def append_parallel_mixer(circuit, problem, beta, layer, parallel):
    data = token_data_qubits(problem); k = (len(problem.sites)-1).bit_length()
    edges = scheduled_token_edges(problem, 'line', 'empty_prioritized', layer)
    stages = dependency_stages(edges) if parallel else [[edge] for edge in edges]
    for group in stages:
        for anc, (u,v) in enumerate(group):
            append_phase_estimation_partial_swap(circuit, list(range(u*k,(u+1)*k)),
                list(range(v*k,(v+1)*k)), data+anc, beta)


def workspace_circuit(problem, parallel, synthesis='beam', policy='zero', theta=None,
                      initial=None, mixer_only=False):
    data = token_data_qubits(problem)
    c = QuantumCircuit(data+(len(problem.sites)//2 if parallel else 1))
    theta = ParameterVector('theta', 5) if theta is None else theta
    if not mixer_only:
        initial = dict(zip(problem.cells, range(len(problem.cells)))) if initial is None else initial
        prepare_token_permutation_state(c, problem, initial)
    for layer in range(1 if mixer_only else 3):
        if layer:
            phase = gray_phase if synthesis == 'gray' else token_phase
            c.compose(phase(problem, theta[2*layer-1], policy), range(data), inplace=True)
        append_parallel_mixer(c, problem, theta[2*layer], layer, parallel)
    return c


def coordinate_coefficients(sites):
    """Separate coordinate LPs on the SAME site-label domain, then coefficient sum."""
    m = len(sites); k = (m-1).bit_length(); dim = 1 << (2*k)
    w = hadamard(dim).astype(float)
    valid = [a|(b<<k) for b in range(m) for a in range(m)]
    weights = np.array([.01+2*max(j.bit_count()-1,0) for j in range(dim)])
    weights[0] = 0
    parts = []
    for axis in (0,1):
        values = np.array([abs(sites[a][axis]-sites[b][axis]) for b in range(m) for a in range(m)])
        if m == 1 << k:
            coeff = w @ values / dim
        else:
            result = linprog(np.r_[weights,weights], A_eq=np.c_[w[valid],-w[valid]],
                             b_eq=values, bounds=(0,None), method='highs')
            if not result.success: raise RuntimeError(result.message)
            coeff = result.x[:dim]-result.x[dim:]
        coeff[abs(coeff)<1e-10] = 0
        assert np.max(abs(w[valid]@coeff-values)) < 1e-8
        parts.append(coeff)
    return parts[0]+parts[1], parts
