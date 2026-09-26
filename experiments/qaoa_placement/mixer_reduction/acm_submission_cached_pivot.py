"""Equivalent PyZX execution: indexed candidates and read-only pivot caching."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch
from acm_submission_fast_verify import fast_matcher,UPSTREAM_SHA256

REWRITE_SHA256='423f1d5b26093472283abad9cf192b0a90f3fc7134c75c9ca170f4dbeb7d2964'


@contextmanager
def accelerate_pyzx():
    import pyzx.rewrite as rewrite
    import pyzx.rewrite_rules.pivot_rule as pivot
    assert hashlib.sha256(Path(rewrite.__file__).read_bytes()).hexdigest()==REWRITE_SHA256
    assert hashlib.sha256(Path(pivot.__file__).read_bytes()).hexdigest()==UPSTREAM_SHA256
    matcher=fast_matcher();cls=rewrite.RewriteSimpDoubleVertex
    original_find=cls.find_all_matches;original_boundary=pivot.boundary_list_for_vertex
    def find(self,graph):
        predicate=self.simp_match if self.simp_match is not None else self.is_match
        if predicate is not pivot.check_pivot:return original_find(self,graph)
        cache={}
        def boundary(current,vertex):
            assert current is graph
            if vertex not in cache:cache[vertex]=original_boundary(current,vertex)
            return cache[vertex]
        # find_all_matches only inspects the graph. Its caller performs every
        # mutation after this context has ended, using uncached rechecks.
        with patch.object(pivot,'boundary_list_for_vertex',boundary):
            return original_find(self,graph)
    with patch.object(cls,'find_all_matches',find),patch.object(pivot,'match_pivot_gadget',matcher):
        yield
