import itertools,random,unittest
from acm_submission_fast_beam import order_parity_terms as fast
from phase_separator_optimization import order_parity_terms as archived,parity_order_metrics

class FastBeamTests(unittest.TestCase):
    def test_identical_decisions_with_duplicates_and_ties(self):
        for seed in range(20):
            rng=random.Random(seed);records=[]
            for i in range(8+seed):
                support=tuple(sorted(rng.sample(range(8),rng.randrange(1,7))))
                records.append(dict(term_id=i,net_id=i//5,mask=rng.randrange(1,256),support=support,support_size=len(support),coefficient=1.))
            for width,pool in [(1,1),(4,10),(3,5)]:
                a=archived(records,'beam_search',width,pool);b=fast(records,'beam_search',width,pool)
                self.assertEqual([r['term_id'] for r in a],[r['term_id'] for r in b])
                self.assertEqual(parity_order_metrics(a),parity_order_metrics(b))
    def test_increment_boundary_exhaustive(self):
        supports=[s for n in range(1,5) for s in itertools.combinations(range(4),n)]
        for a,b in itertools.product(supports,repeat=2):
            common=0
            if a[-1]==b[-1]:
                for x,y in zip(a[:-1],b[:-1]):
                    if x!=y:break
                    common+=1
            expected=2*(len(a)+len(b)-2-common)
            records=[dict(support=s) for s in (a,b)]
            self.assertEqual(expected,parity_order_metrics(records)['explicit_parity_cx'])

if __name__=='__main__':unittest.main()
