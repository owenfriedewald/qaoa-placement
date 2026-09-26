"""Corrected query accounting without modifying the frozen first implementation."""
import random
from acm_gap_methods import classical_search as legacy_search
from run_acm_confirmation import greedy


def counted_search(problem,initial,budget,seed,method,temperature=.2):
    if method!='multistart_greedy':return legacy_search(problem,initial,budget,seed,method,temperature)
    # Let the first greedy search perform the initial query exactly once.
    rng=random.Random(seed);calls=0;best=float('inf')
    while calls<budget:
        start=initial if calls==0 else dict(zip(problem.cells,rng.sample(range(len(problem.sites)),len(problem.cells))))
        value,used=greedy(problem,start,budget-calls);calls+=used;best=min(best,value)
    return float(best),calls
