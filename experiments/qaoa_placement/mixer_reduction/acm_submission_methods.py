"""September 5 controls. Earlier scientific implementations are immutable."""
from __future__ import annotations
import itertools
from unittest.mock import patch
import numpy as np
from scipy.linalg import hadamard
from qiskit.circuit import QuantumCircuit, ParameterVector
from invariant_placement import ReducedPlacement, collision_only_qubo, distance_coefficients
from placement_core import ising_terms
from acm_reviewer_methods import ring_edges, append_parallel_mixer
from token_permutation_encoding import token_data_qubits, prepare_token_permutation_state
import acm_gap_methods as gap
import invariant_placement as inv


def extension(sites, policy):
    m=len(sites); k=(m-1).bit_length(); capacity=1<<k
    if policy in ('zero','l1'):return distance_coefficients(tuple(map(tuple,sites)),policy).copy()
    coords=list(map(tuple,sites))
    if policy=='virtual':
        xmin=min(x for x,y in coords);xmax=max(x for x,y in coords)
        y=min(y for x,y in coords)
        while len(coords)<capacity:
            for x in range(xmin,xmax+1):
                if (x,y) not in coords:coords.append((x,y))
                if len(coords)==capacity:break
            y+=1
    elif policy=='nearest':
        coords += [coords[min(range(m),key=lambda a:((z^a).bit_count(),a))] for z in range(m,capacity)]
    elif policy!='mean':raise ValueError(policy)
    values=np.zeros(capacity**2)
    mean=np.mean([sum(abs(sites[a][i]-sites[b][i]) for i in (0,1)) for a in range(m) for b in range(m)])
    for a,b in itertools.product(range(capacity),repeat=2):
        values[a|(b<<k)] = mean if policy=='mean' and max(a,b)>=m else sum(abs(coords[a][i]-coords[b][i]) for i in (0,1))
    coeff=hadamard(len(values))@values/len(values)
    coeff[abs(coeff)<1e-10]=0
    return coeff


def truncate(sites, coeff, eta):
    m=len(sites);k=(m-1).bit_length();W=hadamard(len(coeff))
    indices=[a|(b<<k) for b in range(m) for a in range(m)]
    truth=np.array([sum(abs(sites[a][i]-sites[b][i]) for i in (0,1)) for b in range(m) for a in range(m)])
    result=coeff.copy();error=W[indices]@result-truth;bound=eta*max(truth)
    for j in sorted(np.flatnonzero(result[1:])+1,key=lambda j:(abs(result[j]),j)):
        candidate=error-W[indices,j]*result[j]
        if max(abs(candidate))<=bound+1e-12:result[j]=0;error=candidate
    return result,float(max(abs(error)))


def absolute_spectrum(bits):
    """Sparse exact Walsh dictionary for |a-b|, a low bits, b high bits.

    Uses D_l=(1+A B)D_(l-1)/2 + 2^(l-2)(1-A B)
      +(B-A)(a_low-b_low)/2, A=(-1)^a_high, B=(-1)^b_high.
    No dense matrix, table, or LP. Integer dyadic coefficients are exact here.
    """
    coeff={}
    for length in range(1,bits+1):
        # Relocate b bits as the register width increases.
        old={ (mask&((1<<(length-1))-1))|((mask>>(length-1))<<length):v for mask,v in coeff.items()}
        A=1<<(length-1);B=1<<(2*length-1);out={}
        def add(mask,value):out[mask]=out.get(mask,0)+value
        for mask,value in old.items():add(mask,value/2);add(mask^A^B,value/2)
        add(0,2.**(length-2));add(A^B,-2.**(length-2))
        for i in range(length-1):
            ai=1<<i;bi=1<<(length+i);v=2.**(i-2)
            add(B^bi,v);add(B^ai,-v);add(A^bi,-v);add(A^ai,v)
        coeff={mask:v for mask,v in out.items() if v}
    return coeff


def grid_spectrum(kx,ky):
    k=kx+ky;out={}
    for width,offset in [(kx,0),(ky,kx)]:
        for mask,value in absolute_spectrum(width).items():
            remap=0
            for i in range(width):
                if mask>>i&1:remap|=1<<(offset+i)
                if mask>>(width+i)&1:remap|=1<<(k+offset+i)
            out[remap]=out.get(remap,0)+value
    return {m:v for m,v in out.items() if v}


def phase_from_coeff(p,gamma,coeff,flow='gray'):
    module=gap if flow in ('gray','library') else inv
    builder={'gray':gap.gray_phase,'library':gap.library_phase,'beam':inv.token_phase}[flow]
    with patch.object(module,'distance_coefficients',lambda *a:coeff):return builder(p,gamma,'l1')


def row_circuit(p,graph,reps=3,theta=None,component='full'):
    theta=ParameterVector('theta',2*reps-1) if theta is None else theta
    m=len(p.sites);c=QuantumCircuit(p.num_variables)
    if component in ('full','prep'):
        for u in range(len(p.cells)):c.x(u*m+u)
    constant,h,j=ising_terms(collision_only_qubo(p))
    for layer in range(reps if component=='full' else 1):
        if (component=='full' and layer) or component=='phase':
            gamma=theta[2*layer-1] if component=='full' else theta[0]
            c.global_phase-=constant*gamma
            for q,v in sorted(h.items()):c.rz(2*v*gamma,q)
            for (u,v),w in sorted(j.items()):c.rzz(2*w*gamma,u,v)
        if component in ('full','mixer'):
            edges=ring_edges(m) if graph=='ring' else list(itertools.combinations(range(m),2))
            for u in range(len(p.cells)):
                for a,b in edges:
                    c.rxx(2*theta[2*layer],u*m+a,u*m+b);c.ryy(2*theta[2*layer],u*m+a,u*m+b)
    return c


def token_circuit(p,reps=3,parallel=False,component='full'):
    theta=ParameterVector('theta',2*reps-1);data=token_data_qubits(p)
    c=QuantumCircuit(data+(len(p.sites)//2 if parallel else 1))
    if component in ('full','prep'):prepare_token_permutation_state(c,p,dict(zip(p.cells,range(len(p.cells)))))
    for layer in range(reps if component=='full' else 1):
        if (component=='full' and layer) or component=='phase':
            gamma=theta[2*layer-1] if component=='full' else theta[0]
            c.compose(gap.gray_phase(p,gamma,'l1'),range(data),inplace=True)
        if component in ('full','mixer'):append_parallel_mixer(c,p,theta[2*layer],layer,parallel)
    return c


class RandomToken(ReducedPlacement):
    def __init__(self,p,seed):
        super().__init__(p,'token','line','empty_prioritized');self.seed=seed
    def probabilities(self,theta,initial,reps=3):
        import random
        from token_permutation_encoding import assignment_to_token_sites
        state=np.zeros(len(self.states),complex);state[self.index[assignment_to_token_sites(self.problem,initial)]]=1
        for layer in range(reps):
            if layer:state*=np.exp(-1j*theta[2*layer-1]*self.energies)
            edges=list(self.pairs);random.Random(self.seed+1905+layer).shuffle(edges)
            for edge in edges:state=np.cos(theta[2*layer])*state-1j*np.sin(theta[2*layer])*state[self.pairs[edge]]
        return abs(state)**2
