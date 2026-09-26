"""Independent correctness checks for new synthesis/statistical controls."""
import itertools
import unittest
import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from scipy.linalg import hadamard
from acm_gap_methods import (fresh_problem,gray_phase,library_phase,controlled_circuit,
    cached_cvar,sampled_cvar,classical_search)
from invariant_placement import distance_coefficients,placement_circuit
from objective_functions import cvar_cost
from placement_core import PlacementProblem,hpwl

class GapMethodsTests(unittest.TestCase):
    def test_all_basis_phases_match_independent_truth_table(self):
        p=PlacementProblem(('A','B'),((0,0),(1,2),(3,1)),(('A','B',1.37),),5)
        rng=np.random.default_rng(13);state=rng.normal(size=64)+1j*rng.normal(size=64);state/=np.linalg.norm(state)
        for policy in ('zero','l1'):
            values=hadamard(16)@distance_coefficients(p.sites,policy)
            for angle in (.17,-.391,1.03):
                expected=state*np.exp(-1j*angle*1.37*values[np.arange(64)%16])
                for builder in (gray_phase,library_phase):
                    circuit=builder(p,angle,policy)
                    np.testing.assert_allclose(Statevector(state).evolve(circuit).data,expected,atol=1e-10)

    def test_global_shared_support_aggregation(self):
        p=fresh_problem(3,3,'compact',778)
        angle=Parameter('g');theta=.291
        from invariant_placement import token_phase
        for policy in ('zero','l1'):
            reference=token_phase(p,angle,policy).assign_parameters({angle:theta})
            new=gray_phase(p,angle,policy).assign_parameters({angle:theta})
            state=Statevector(np.ones(64)/8)
            np.testing.assert_allclose(state.evolve(new).data,state.evolve(reference).data,atol=1e-10)

    def test_prepared_full_circuit_equivalence_and_clean_ancilla(self):
        p=fresh_problem(2,3,'line',44)
        for theta in ([.2,.31,-.17,.12,.4],[-.3,.25,.13,-.41,.32]):
            expected=Statevector.from_instruction(placement_circuit(p,'token',schedule='empty_prioritized',theta=theta)).data
            for method in ('token_gray_zero','token_gray_l1','token_beam_l1'):
                actual=Statevector.from_instruction(controlled_circuit(p,method,theta)).data
                np.testing.assert_allclose(actual,expected,atol=1e-9)
                self.assertLess(np.linalg.norm(actual[len(actual)//2:]),1e-10)

    def test_cached_cvar_fractional_tail_and_duplicate_energies(self):
        p=np.array([.1,.3,.2,.4]);e=np.array([4.,1.,1.,9.]);order=np.argsort(e)
        for alpha in (.15,.25,.7,1):
            self.assertAlmostEqual(cached_cvar(p,order,e,alpha),cvar_cost(p,e,alpha),places=12)

    def test_sampled_cvar_is_empirical_tail_not_analytic_loss(self):
        p=np.array([.1,.3,.2,.4]);e=np.array([4.,1.,1.,9.]);shots=32
        for seed in (1,9,81):
            actual=sampled_cvar(p,e,np.random.default_rng(seed),shots,.25)
            samples=np.random.default_rng(seed).choice(len(e),shots,p=p)
            expected=sorted(e[samples])[:8]
            self.assertAlmostEqual(actual,np.mean(expected),places=12)

    def test_classical_budgets_and_initial_fallback(self):
        p=fresh_problem(3,6,'scatter',711);initial=dict(zip(p.cells,[0,1,2]));cost=hpwl(p,initial)
        optimum=min(hpwl(p,dict(zip(p.cells,a))) for a in itertools.permutations(range(6),3))
        for method,budget in itertools.product(('uniform','annealing','multistart_greedy'),(1,2,32,128)):
            best,calls=classical_search(p,initial,budget,19,method)
            self.assertEqual(calls,budget);self.assertLessEqual(best,cost);self.assertGreaterEqual(best,optimum-1e-10)

if __name__=='__main__':unittest.main()
