import unittest
from acm_reviewer_ring_retry import confirmation_ring


class AdapterTests(unittest.TestCase):
    def test_legacy_postprocessing_does_not_drop_ring_rows(self):
        row=dict(instance_id='adapter_preflight',family='compact',cells=['A','B'],
            geometry=[[0,0],[2,0],[0,1]],nets=[['A','B',1.]],
            initializations=[dict(init_mode=m,init_seed=1177101,assignment={'A':0,'B':1}) for m in ('deterministic','poor','random')])
        rows=confirmation_ring(row)
        self.assertEqual(len(rows),12)
        self.assertEqual({r['method'] for r in rows},{'ring_row_xy'})
        self.assertEqual({(r['initialization'],r['optimizer_seed']) for r in rows},
            {(m,s) for m in ('deterministic','poor','random') for s in (11,17,23,31)})
        self.assertTrue(all(0<=r['optimal_probability']<=r['feasible_probability']+1e-10 for r in rows))


if __name__=='__main__':unittest.main()
