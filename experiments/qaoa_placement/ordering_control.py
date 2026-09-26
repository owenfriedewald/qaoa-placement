"""Ordering-controlled legal mixer schedules for the paper suite."""

from __future__ import annotations

import random
from typing import Sequence, Tuple

from mixer_diagnostics import Edge, MixerSchedule


def make_random_once_schedule(edges: Sequence[Edge], seed: int) -> MixerSchedule:
    shuffled = list(edges)
    random.Random(seed).shuffle(shuffled)
    return MixerSchedule(
        name="random_once",
        edge_layers=(tuple(shuffled),),
        description="One seeded random legal edge ordering reused for every QAOA layer.",
        seed=seed,
    )


def make_palindromic_schedule(edges: Sequence[Edge]) -> MixerSchedule:
    """Return metadata for a forward/reverse symmetric legal schedule.

    The runner applies this as a half-angle composition:

        U_M(beta) = prod_reverse exp(-i beta/2 X_e) prod_forward exp(-i beta/2 X_e)

    This keeps the same legal transition edge set while reducing dependence on
    arbitrary forward-only ordering.
    """

    forward = tuple(edges)
    reverse = tuple(reversed(edges))
    return MixerSchedule(
        name="palindromic_fr",
        edge_layers=(forward, reverse),
        description="Forward/reverse palindromic legal edge schedule with half-angle composition.",
    )


def make_ordering_ensemble_schedules(edges: Sequence[Edge], seeds: Sequence[int]) -> Tuple[MixerSchedule, ...]:
    return tuple(make_random_once_schedule(edges, seed) for seed in seeds)
