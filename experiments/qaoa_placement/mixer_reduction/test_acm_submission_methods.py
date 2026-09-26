import unittest
import itertools
import numpy as np
from scipy.linalg import hadamard
from qiskit.quantum_info import Operator,Statevector
from qiskit import QuantumCircuit
from dataclasses import replace
from acm_gap_methods import fresh_problem
from acm_submission_methods import extension,absolute_spectrum,grid_spectrum,truncate,row_circuit,token_circuit,phase_from_coeff
from acm_reviewer_methods import ring_circuit,workspace_circuit,RingPlacement
from invariant_placement import placement_circuit,circuit_hash

class SubmissionTests(unittest.TestCase):
    def test_extensions_and_error_bound(self):
        p=fresh_problem(3,6,'scatter',1289001);k=3;W=hadamard(64);indices=[a|(b<<k) for b in range(6) for a in range(6)]
        truth=np.array([sum(abs(p.sites[a][i]-p.sites[b][i]) for i in (0,1)) for b in range(6) for a in range(6)])
        for policy in ('zero','l1','virtual','nearest','mean'):
            c=extension(p.sites,policy);self.assertLess(max(abs(W[indices]@c-truth)),1e-8)
            for eta in (.001,.01,.05):
                tr,error=truncate(p.sites,c,eta);self.assertLessEqual(error,eta*max(truth)+1e-10)
    def test_structured_independent_dense(self):
        for kx,ky in [(1,0),(3,0),(2,2),(3,2)]:
            k=kx+ky;dim=1<<(2*k);c=np.zeros(dim)
            for mask,v in grid_spectrum(kx,ky).items():c[mask]=v
            sites=[(a% (1<<kx),a>>kx) for a in range(1<<k)]
            truth=[sum(abs(sites[a][i]-sites[b][i]) for i in (0,1)) for b in range(len(sites)) for a in range(len(sites))]
            np.testing.assert_allclose(hadamard(dim)@c,truth,atol=1e-12)
    def test_generalized_contract(self):
        p=fresh_problem(3,4,'compact',1289002)
        self.assertEqual(circuit_hash(row_circuit(p,'ring')),circuit_hash(ring_circuit(p)))
        self.assertEqual(circuit_hash(row_circuit(p,'complete')),circuit_hash(placement_circuit(p,'penalty_row_xy')))
        self.assertEqual(circuit_hash(token_circuit(p)),circuit_hash(workspace_circuit(p,False,'gray','l1')))
        initial=dict(zip(p.cells,range(3)))
        for depth in (1,2,4):
            theta=np.arange(2*depth-1)*.17+.13
            state=Statevector.from_instruction(row_circuit(p,'ring',depth,theta)).probabilities()
            reduced=RingPlacement(p);q=reduced.probabilities(theta,initial,depth)
            indices=[sum(1<<(u*4+int(site)) for u,site in enumerate(row)) for row in reduced.states]
            np.testing.assert_allclose(state[indices],q,atol=1e-10)
    def test_zx_numeric_unitary(self):
        from acm_submission_study import pyzx_flow
        p=fresh_problem(2,3,'compact',1289003);p=replace(p,nets=((p.cells[0],p.cells[1],1.13),))
        for policy in ('zero','l1','virtual','nearest','mean'):
            c=phase_from_coeff(p,.371,extension(p.sites,policy));active=QuantumCircuit(4)
            for op in c.data:active.append(op.operation,[c.find_bit(q).index for q in op.qubits])
            active.global_phase=c.global_phase
            d=pyzx_flow(active);self.assertTrue(Operator(active).equiv(Operator(d),atol=1e-7))

if __name__=='__main__':unittest.main()
