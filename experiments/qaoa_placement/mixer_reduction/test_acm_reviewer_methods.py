"""Independent state/action checks before amendment evaluation."""
import itertools
import unittest
import numpy as np
from scipy.linalg import hadamard
from qiskit.quantum_info import Statevector
from placement_core import PlacementProblem
from acm_reviewer_methods import (RingPlacement,ring_circuit,workspace_circuit,
    dependency_stages,coordinate_coefficients,ring_edges)
from invariant_placement import placement_circuit,distance_coefficients


class ReviewerTests(unittest.TestCase):
    def test_ring_reduced_matches_full_state_and_row_invariant(self):
        p=PlacementProblem(('A','B'),((0,0),(2,0),(1,2)),(('A','B',1.3),),5.)
        model=RingPlacement(p)
        for theta in ([.17,.31,.41,.53,.67],[.9,-.23,.12,.5,-.31]):
            initial={'A':2,'B':0}
            full=Statevector.from_instruction(ring_circuit(p,theta,initial)).probabilities()
            encoded=[sum(1<<(u*3+int(s)) for u,s in enumerate(state)) for state in model.states]
            np.testing.assert_allclose(full[encoded],model.probabilities(theta,initial),atol=1e-11)
            self.assertAlmostEqual(float(full[encoded].sum()),1.)
        self.assertEqual(len(set(tuple(sorted(e)) for e in ring_edges(6))),6)

    def test_parallel_mixer_arbitrary_superposition_and_clean_workspace(self):
        p=PlacementProblem(('A','B'),((0,0),(1,0),(0,1),(1,1)),(('A','B',1.),),5.)
        rng=np.random.default_rng(1177101)
        # Arbitrary data state tests more than just the valid permutation subspace.
        data=rng.normal(size=256)+1j*rng.normal(size=256);data/=np.linalg.norm(data)
        for beta in (.17,.371,1.03):
            outputs=[]
            for parallel in (False,True):
                c=workspace_circuit(p,parallel,theta=[beta,0,0,0,0],mixer_only=True)
                state=np.zeros(1<<c.num_qubits,complex);state[:256]=data
                out=Statevector(state).evolve(c).data
                np.testing.assert_allclose(out[256:],0,atol=1e-12)
                outputs.append(out[:256])
            np.testing.assert_allclose(*outputs,atol=1e-11)

    def test_dependency_schedule_and_integrated_action(self):
        edges=[(4,5),(0,1),(1,2),(2,3),(3,4)]
        groups=dependency_stages(edges)
        self.assertEqual(len(groups),4)
        stage={edge:i for i,g in enumerate(groups) for edge in g}
        for i,left in enumerate(edges):
            for right in edges[i+1:]:
                if set(left)&set(right): self.assertLess(stage[left],stage[right])
        p=PlacementProblem(('A','B'),((0,0),(1,0),(0,1),(1,1)),(('A','B',1.),),5.)
        theta=[.19,.37,.23,.51,.11]
        ref=Statevector.from_instruction(placement_circuit(p,'token',schedule='empty_prioritized',theta=theta)).data
        for synthesis in ('beam','gray'):
            c=workspace_circuit(p,True,synthesis,'l1',theta)
            out=Statevector.from_instruction(c).data
            np.testing.assert_allclose(out[:256],ref[:256],atol=1e-11)
            np.testing.assert_allclose(out[256:],0,atol=1e-11)

    def test_coordinate_completion_and_full_capacity(self):
        for sites in (((0,0),(2,1),(1,3)),((0,0),(2,1),(1,3),(4,2))):
            c,parts=coordinate_coefficients(sites);dense=distance_coefficients(sites,'l1')
            k=(len(sites)-1).bit_length();w=hadamard(len(c))
            for a,b in itertools.product(range(len(sites)),repeat=2):
                self.assertAlmostEqual((w@c)[a|(b<<k)],sum(abs(sites[a][i]-sites[b][i]) for i in (0,1)))
            np.testing.assert_allclose(c,parts[0]+parts[1])
            if len(sites)==4:np.testing.assert_allclose(c,dense,atol=1e-9)


if __name__=='__main__':unittest.main()
