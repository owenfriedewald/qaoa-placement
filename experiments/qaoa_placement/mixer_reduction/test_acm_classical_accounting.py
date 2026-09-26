import unittest
from unittest.mock import patch
import acm_gap_methods
import run_acm_confirmation
from acm_classical_accounting import counted_search
from placement_core import hpwl

class ClassicalAccountingTests(unittest.TestCase):
    def test_real_oracle_invocations_match_budget_and_preserve_search_results(self):
        problem=acm_gap_methods.fresh_problem(3,6,'scatter',521);initial=dict(zip(problem.cells,[0,1,2]))
        for method in ('uniform','annealing','multistart_greedy'):
            for budget in (1,2,32,128,512):
                calls=[]
                def counted(p,a):calls.append(1);return hpwl(p,a)
                expected=acm_gap_methods.classical_search(problem,initial,budget,619,method)
                with patch.object(acm_gap_methods,'hpwl',counted),patch.object(run_acm_confirmation,'hpwl',counted):
                    actual=counted_search(problem,initial,budget,619,method)
                self.assertEqual(len(calls),budget)
                self.assertEqual(actual,expected)

if __name__=='__main__':unittest.main()
