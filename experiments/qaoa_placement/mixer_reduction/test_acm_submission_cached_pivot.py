import random,unittest
from fractions import Fraction
from acm_submission_cached_pivot import accelerate_pyzx


class CachedPivotTests(unittest.TestCase):
    def test_same_matches_and_no_cache_after_mutation(self):
        import pyzx as zx
        from pyzx.rewrite import RewriteSimpDoubleVertex
        import pyzx.rewrite_rules.pivot_rule as pivot
        from pyzx.utils import EdgeType
        rule=RewriteSimpDoubleVertex(pivot.check_pivot,pivot.unsafe_pivot)
        original=RewriteSimpDoubleVertex.find_all_matches;matched=0;changed=0
        def signature(g):
            return ([(v,g.type(v),g.phase(v),g.qubit(v),g.row(v)) for v in sorted(g.vertices())],sorted((g.edge_st(e),g.edge_type(e)) for e in g.edges()))
        for seed in range(12):
            rng=random.Random(seed);c=zx.Circuit(6)
            for _ in range(60):
                a,b=rng.sample(range(6),2);c.add_gate('CNOT',a,b)
                c.add_gate('ZPhase',b,phase=Fraction(rng.randrange(7),3))
                if rng.randrange(3)==0:c.add_gate('HAD',a)
            g=c.to_graph();zx.simplify.to_gh(g);zx.simplify.spider_simp(g);zx.simplify.id_simp(g)
            with accelerate_pyzx():
                for _ in range(3):
                    before=signature(g);expected=original(rule,g);actual=rule.find_all_matches(g)
                    self.assertEqual(expected,actual);self.assertEqual(before,signature(g));matched+=len(expected)
                    if expected:
                        a,b=next(iter(expected));g.set_edge_type(g.edge(a,b),EdgeType.SIMPLE)
                        after=original(rule,g);self.assertNotEqual(expected,after);changed+=1
                        self.assertEqual(after,rule.find_all_matches(g))
        self.assertGreater(matched,0);self.assertGreater(changed,0)

if __name__=='__main__':unittest.main()
