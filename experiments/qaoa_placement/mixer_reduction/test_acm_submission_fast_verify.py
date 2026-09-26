import random,unittest
from fractions import Fraction
from acm_submission_fast_verify import OrderedCandidates,fast_matcher,accelerate_verification


class FastVerifyTests(unittest.TestCase):
    def test_duplicate_operation_traces(self):
        for seed in range(50):
            rng=random.Random(seed);old=[(rng.randrange(8),rng.randrange(8)) for _ in range(100)]
            new=OrderedCandidates(old)
            while old:
                self.assertEqual(len(old),len(new))
                for value in [(rng.randrange(10),rng.randrange(10)) for _ in range(4)]:
                    self.assertEqual(value in old,value in new)
                if rng.randrange(2):self.assertEqual(old.pop(),new.pop())
                else:
                    value=rng.choice(old);old.remove(value);new.remove(value)
            self.assertEqual(len(new),0)
            with self.assertRaises(IndexError):new.pop()
            with self.assertRaises(ValueError):new.remove((100,100))
    def test_match_order_and_graph_mutations(self):
        import pyzx as zx
        import pyzx.rewrite_rules.pivot_rule as pivot
        fast=fast_matcher();matched=0
        def signature(g):
            return ([(v,g.type(v),g.phase(v),g.qubit(v),g.row(v)) for v in sorted(g.vertices())],
                sorted((g.edge_st(e),g.edge_type(e)) for e in g.edges()),str(g.scalar))
        for seed in range(12):
            rng=random.Random(seed);c=zx.Circuit(6)
            for _ in range(60):
                a,b=rng.sample(range(6),2);c.add_gate('CNOT',a,b)
                c.add_gate('ZPhase',b,phase=Fraction(rng.randrange(7),3))
                if rng.randrange(3)==0:c.add_gate('HAD',a)
            g=c.to_graph();zx.simplify.to_gh(g);zx.simplify.spider_simp(g);zx.simplify.id_simp(g)
            a,b=g.copy(),g.copy();left=pivot.match_pivot_gadget(a);right=fast(b)
            matched+=len(left);self.assertEqual(left,right);self.assertEqual(signature(a),signature(b))
        self.assertGreater(matched,0)
    def test_equal_and_unequal_verification(self):
        import pyzx as zx
        for seed in range(4):
            rng=random.Random(seed);c=zx.Circuit(4)
            for _ in range(12):
                a,b=rng.sample(range(4),2);c.add_gate('CNOT',a,b)
                c.add_gate('ZPhase',b,phase=Fraction(rng.randrange(7),5))
            g=c.to_graph();zx.full_reduce(g);other=zx.extract_circuit(g)
            wrong=other.copy();wrong.add_gate('ZPhase',0,phase=Fraction(1,3))
            expected=[c.verify_equality(other),c.verify_equality(wrong)]
            self.assertEqual(expected,[True,False])
            with accelerate_verification():actual=[c.verify_equality(other),c.verify_equality(wrong)]
            self.assertEqual(expected,actual)

if __name__=='__main__':unittest.main()
