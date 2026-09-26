# Execution correction: cache read-only PyZX pivot predicates

Verifier preflight 17011342 passed the three indexed-candidate tests, the
independent Qiskit unitary test, and all 200 retained PyZX circuit hashes.
The previously timed-out case then exposed another hotspot. A retained debugger
trace identifies `full_reduce -> pivot_simp -> find_all_matches -> check_pivot
-> boundary_list_for_vertex`, in initial synthesis. For each eligible edge,
the same vertex's boundary predicate is recomputed, although the graph does not
change during `find_all_matches`.

Freeze this note before implementing the following additional execution change:

1. Cache `boundary_list_for_vertex(graph, vertex)` only inside a single call to
   the two-vertex rule's `find_all_matches`, and only when the selected predicate
   is the pinned `check_pivot`. The original scan, pair insertion order and set
   of matches remain unchanged. Clear the cache on returning from that scan;
   subsequent match rechecks and graph mutations use uncached predicates.
   Guard the exact upstream source hashes. Never edit the installed dependency.
2. Apply the already tested ordered candidate container during initial PyZX
   reduction as well as verification. Its pop-last/remove-first/duplicate
   semantics preserve matcher decisions; circuit-output hashes must confirm
   that the scope extension does not change synthesis output.
3. Compare original and cached match sets and graph states on seeded graphs,
   then mutate graphs and repeat to ensure no stale cache. Require the earlier
   indexed-candidate tests, independent small unitary checks, all 200 retained
   circuit hashes, and all 31 rows of the previously timed-out case again.
   Freeze implementation hashes before validation. Reject any discrepancy.
4. Stop the diagnosed preflight by its exact job ID and retain its passed
   tests, 200 matches, source snapshot and diagnostic log. Do not label it a
   complete preflight because the larger case has not finished. Its temporary
   debugger pauses exclude it from timing comparisons. New preflight and resumed
   campaign directories must be distinct. Preserve the scientific specification,
   cases, objectives, circuit contracts, compiler versions, seeds and endpoints.

The change is a read-only predicate cache and a list-equivalent data structure,
not a new rewrite rule or a weaker equivalence test. At most two Slurm jobs
run concurrently; native pools remain one, preflight has two workers, and
campaign jobs have at most eight workers, 8 GB and two-hour limits. Full
aggregates still require complete coverage; incomplete attempts remain separate.
