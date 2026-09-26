"""Bounded memoization of identical, already-verified numeric phase inputs."""
from collections import OrderedDict
from contextlib import contextmanager
import json,math
from unittest.mock import patch
from qiskit.circuit.library import CXGate,RZGate
from invariant_placement import circuit_hash


class MemoizedPhaseCompiler:
    def __init__(self,compiler,max_entries=5):
        assert 1<=max_entries<=5
        self.compiler=compiler;self.max_entries=max_entries
        self.cache=OrderedDict();self.hits=0;self.misses=0

    def __call__(self,circuit):
        assert not circuit.parameters and not circuit.num_clbits
        assert math.isfinite(float(circuit.global_phase))
        for item in circuit.data:
            assert isinstance(item.operation,(CXGate,RZGate))
            assert all(math.isfinite(float(p)) for p in item.operation.params)
        key=circuit_hash(circuit)
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key)
        else:
            self.misses+=1
            self.cache[key]=self.compiler(circuit).copy()
            if len(self.cache)>self.max_entries:self.cache.popitem(last=False)
        return self.cache[key].copy()


@contextmanager
def memoized_pyzx():
    import acm_submission_study as study
    memo=MemoizedPhaseCompiler(study.pyzx_flow)
    try:
        with patch.object(study,'pyzx_flow',memo):yield memo
    finally:
        print(json.dumps(dict(pyzx_memo_hits=memo.hits,pyzx_memo_misses=memo.misses,cache_limit=memo.max_entries)),flush=True)
