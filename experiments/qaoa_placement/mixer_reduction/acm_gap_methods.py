"""Additional controls; historical/frozen implementations remain unchanged."""
from __future__ import annotations
import itertools
import math
import random
from dataclasses import replace
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import DiagonalGate
from scipy.linalg import hadamard
from invariant_placement import distance_coefficients, token_phase, placement_circuit
from placement_core import PlacementProblem, hpwl
from token_permutation_encoding import token_data_qubits, prepare_token_permutation_state
from run_token_permutation_p3_quality_uplift import append_scheduled_mixer
from run_acm_confirmation import greedy


def fresh_problem(n, m, family, seed):
    """Fresh integer coordinates, shuffled site labels, and connected weighted nets."""
    rng=random.Random(seed)
    if family=='line':
        coords=[(x,0) for x in rng.sample(range(3*m),m)]
    elif family=='l_shape':
        coords=rng.sample([(x,0) for x in range(m)]+[(0,y) for y in range(1,m)],m)
    elif family=='compact':
        width=math.ceil(math.sqrt(m));coords=rng.sample([(x,y) for x in range(width+1) for y in range(width+1)],m)
    elif family=='scatter':
        coords=rng.sample([(x,y) for x in range(3*m) for y in range(3*m)],m)
    else: raise ValueError(family)
    rng.shuffle(coords)
    cells=tuple(chr(65+i) for i in range(n)); nets=[]
    for i,j in itertools.combinations(range(n),2):
        if j==i+1 or rng.random()<.65:
            nets.append((cells[i],cells[j],round(rng.uniform(.5,2),2)))
    return PlacementProblem(cells,tuple(coords),tuple(nets),5.0)


def inverse_gray(gray):
    value=0
    while gray:
        value ^= gray; gray >>= 1
    return value


def gray_phase(problem, gamma, policy):
    """Exact global parity aggregation and sparse reflected-Gray traversal.

    For each highest support bit, retain parity between rotations; restore
    identity before changing target. Skipping zero terms does not truncate.
    """
    width=(len(problem.sites)-1).bit_length()
    circuit=QuantumCircuit(token_data_qubits(problem))
    labels={c:i for i,c in enumerate(problem.cells)}; terms={}
    coeff=distance_coefficients(problem.sites,policy)
    for u,v,w in problem.nets:
        qubits=[labels[u]*width+i for i in range(width)]+[labels[v]*width+i for i in range(width)]
        for mask,c in enumerate(coeff):
            if c==0:continue
            global_mask=sum(1<<q for i,q in enumerate(qubits) if mask>>i&1)
            terms[global_mask]=terms.get(global_mask,0)+w*float(c)
    circuit.global_phase=-gamma*terms.pop(0,0)
    for target in range(circuit.num_qubits):
        masks=[m for m,c in terms.items() if m.bit_length()-1==target and abs(c)>1e-12]
        masks.sort(key=lambda m:inverse_gray(m^(1<<target)))
        previous=0
        for mask in masks:
            controls=mask^(1<<target)
            changed=previous^controls
            for bit in range(target):
                if changed>>bit&1:circuit.cx(bit,target)
            circuit.rz(2*terms[mask]*gamma,target)
            previous=controls
        for bit in range(target):
            if previous>>bit&1:circuit.cx(bit,target)
    return circuit


def library_phase(problem, gamma, policy):
    """Qiskit's independent generic diagonal synthesis, numerically bound only."""
    width=(len(problem.sites)-1).bit_length()
    circuit=QuantumCircuit(token_data_qubits(problem))
    labels={c:i for i,c in enumerate(problem.cells)}
    coeff=distance_coefficients(problem.sites,policy); distances=hadamard(len(coeff))@coeff
    for u,v,w in problem.nets:
        qubits=[labels[u]*width+i for i in range(width)]+[labels[v]*width+i for i in range(width)]
        circuit.append(DiagonalGate(np.exp(-1j*float(gamma)*w*distances)),qubits)
    return circuit


def controlled_circuit(problem, method, theta=None):
    if method=='penalty_row_xy':return placement_circuit(problem,method,theta=theta)
    _,synthesis,policy=method.split('_')
    if synthesis=='beam':return placement_circuit(problem,'token',schedule='empty_prioritized',theta=theta,completion=policy)
    theta=ParameterVector('theta',5) if theta is None else theta
    data=token_data_qubits(problem); circuit=QuantumCircuit(data+1)
    prepare_token_permutation_state(circuit,problem,dict(zip(problem.cells,range(len(problem.cells)))))
    for layer in range(3):
        if layer:circuit.compose(gray_phase(problem,theta[2*layer-1],policy),range(data),inplace=True)
        append_scheduled_mixer(circuit,problem,theta[2*layer],'line','empty_prioritized',layer)
    return circuit


def initial_assignment(model, mode, seed):
    p=model.problem;n=len(p.cells);m=len(p.sites)
    if mode=='deterministic':real=tuple(range(n))
    elif mode=='random':real=random.Random(seed).sample(range(m),n)
    elif mode=='poor':
        legal=np.flatnonzero(model.legal);i=legal[np.argmax(model.costs[legal])];real=model.states[i,:n]
    else:raise ValueError(mode)
    return {c:int(s) for c,s in zip(p.cells,real)}


def cached_cvar(probabilities, sorted_indices, energies, alpha=.25):
    mass=probabilities[sorted_indices]; ordered=energies[sorted_indices]
    before=np.cumsum(mass)-mass
    used=np.minimum(mass,np.maximum(0,alpha-before))
    return float(np.dot(used,ordered)/alpha)


def sampled_cvar(probabilities, energies, rng, shots=2048, alpha=.25):
    if shots<1 or abs(shots*alpha-round(shots*alpha))>1e-12:raise ValueError('Integer CVaR tail required')
    samples=rng.choice(len(energies),size=shots,p=probabilities)
    tail=int(round(shots*alpha))
    return float(np.partition(energies[samples],tail-1)[:tail].mean())


def classical_search(problem, initial, budget, seed, method, temperature=.2):
    """Count every objective query, including repeated candidates and initialization."""
    rng=random.Random(seed);n=len(problem.cells);m=len(problem.sites)
    current=tuple(initial[c] for c in problem.cells);cost=hpwl(problem,initial)
    best=cost;calls=1
    if method=='multistart_greedy':
        while calls<budget:
            start=initial if calls==1 else dict(zip(problem.cells,rng.sample(range(m),n)))
            value,used=greedy(problem,start,budget-calls+(calls==1))
            calls+=used-(calls==1);best=min(best,value)
        return float(best),calls
    start_temp=max(1e-9,temperature*cost/n)
    while calls<budget:
        if method=='uniform':candidate=tuple(rng.sample(range(m),n))
        elif method=='annealing':
            candidate=list(current);u=rng.randrange(n);site=rng.randrange(m)
            if site in current:
                v=current.index(site);candidate[u],candidate[v]=candidate[v],candidate[u]
            else:candidate[u]=site
            candidate=tuple(candidate)
        else:raise ValueError(method)
        value=hpwl(problem,dict(zip(problem.cells,candidate)));calls+=1;best=min(best,value)
        temp=start_temp*(.01**((calls-1)/max(1,budget-1)))
        if method=='uniform' or value<=cost or rng.random()<math.exp(min(0,(cost-value)/temp)):
            current,cost=candidate,value
    return float(best),calls
