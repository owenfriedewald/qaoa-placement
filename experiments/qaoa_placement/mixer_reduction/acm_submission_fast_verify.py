"""Preserve PyZX 0.10.3 verifier decisions with an indexed candidate sequence.

Only the verifier's candidate container changes. Circuit synthesis, extraction,
phases, match order and identity acceptance are unchanged. Installed packages
and the frozen study implementation are never edited.
"""
from collections import defaultdict,deque
from contextlib import contextmanager
import hashlib,importlib.metadata as md,inspect
from pathlib import Path
from unittest.mock import patch

UPSTREAM_SHA256='d7cfdf4551004549d60a99fc92b9c74b457464fabf2a6314e64b0991a905f9de'


class OrderedCandidates:
    """List-equivalent len, membership, pop-last and remove-first for hashables."""
    def __init__(self,values):
        self.values=list(values);n=len(self.values)
        self.previous=list(range(-1,n-1));self.following=list(range(1,n))+[-1] if n else []
        self.positions=defaultdict(deque)
        for i,v in enumerate(self.values):self.positions[v].append(i)
        self.tail=n-1;self.size=n
    def __len__(self):return self.size
    def __contains__(self,value):return value in self.positions
    def _unlink(self,index):
        before=self.previous[index];after=self.following[index]
        if before!=-1:self.following[before]=after
        if after!=-1:self.previous[after]=before
        else:self.tail=before
        self.size-=1
    def pop(self):
        if self.tail==-1:raise IndexError('pop from empty list')
        index=self.tail;value=self.values[index];positions=self.positions[value]
        assert positions.pop()==index
        if not positions:del self.positions[value]
        self._unlink(index);return value
    def remove(self,value):
        if value not in self.positions:raise ValueError('value absent')
        positions=self.positions[value];index=positions.popleft()
        if not positions:del self.positions[value]
        self._unlink(index)


def fast_matcher():
    import pyzx.rewrite_rules.pivot_rule as pivot
    assert md.version('pyzx')=='0.10.3'
    assert hashlib.sha256(Path(pivot.__file__).read_bytes()).hexdigest()==UPSTREAM_SHA256
    source=inspect.getsource(pivot.match_pivot_gadget)
    old='candidates = list(Counter(candidates_set).elements())'
    assert source.count(old)==1
    new=source.replace(old,'candidates = _OrderedCandidates(Counter(candidates_set).elements())')
    namespace=dict(vars(pivot),_OrderedCandidates=OrderedCandidates)
    exec(compile(new,str(Path(__file__).resolve())+'::upstream_matcher','exec'),namespace)
    return namespace['match_pivot_gadget']


@contextmanager
def accelerate_verification():
    import pyzx
    import pyzx.rewrite_rules.pivot_rule as pivot
    original=pyzx.Circuit.verify_equality;matcher=fast_matcher()
    def verify(self,*args,**kwargs):
        with patch.object(pivot,'match_pivot_gadget',matcher):
            return original(self,*args,**kwargs)
    with patch.object(pyzx.Circuit,'verify_equality',verify):yield
