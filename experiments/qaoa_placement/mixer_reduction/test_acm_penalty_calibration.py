import itertools
import unittest
from dataclasses import replace
from acm_gap_methods import fresh_problem
from acm_penalty_calibration import sufficient_penalty_scale
from placement_core import hpwl

class PenaltyBoundTests(unittest.TestCase):
    def test_every_collision_has_strictly_lower_energy_empty_site_repair(self):
        for seed in (31,47,59):
            p=fresh_problem(3,4,'scatter',seed);lam=sufficient_penalty_scale(p)*(1+1e-6)
            def energy(a):
                return hpwl(p,dict(zip(p.cells,a)))+2*lam*sum(a[u]==a[v] for u,v in itertools.combinations(range(3),2))
            for a in itertools.product(range(4),repeat=3):
                if len(set(a))==3:continue
                cell=next(u for u in range(3) if a.count(a[u])>1);empty=next(s for s in range(4) if s not in a)
                b=list(a);b[cell]=empty
                self.assertLess(energy(b),energy(a))

if __name__=='__main__':unittest.main()
