# Identical-input PyZX memoization — prospective execution correction

The scientific specification in `spec.md` is unchanged: 160 phase cases,
five exact policies, five flows and six approximate controls, with the same
4,960-row endpoint and circuit contracts. This note precedes implementation
and validation of the following execution-only optimization.

Cached phase job 17013756 has passed the preceding equivalence gates and is
running. Its successful rows and all earlier attempts remain immutable.
Full-capacity policies specify the same distance table, so they can send
identical numeric circuits to the same deterministic PyZX pipeline repeatedly.

Memoize the already validated ordered-candidate/read-only-pivot pipeline by
the frozen `circuit_hash` of its numeric input. Accept only phase circuits
without classical bits or free parameters, consisting of CX and RZ gates.
The key includes width, global phase, ordered operations, wire indices and
numeric angles. Cache at most five compiled circuits per worker process.
Return independent circuit copies, so caller mutation cannot change a retained
result. A cache miss runs the unchanged verified pipeline and its equality
check; a hit copies a result already validated for the identical input.

Before campaign use, test cache-key distinctions, bounded retention and copy
isolation, and repeat all 200 retained PyZX output-hash comparisons from the
original 40-case ledger. Record hits/misses and the new implementation hashes.
No additional large-case equivalence gate is needed: misses run the preceding
pipeline unchanged, and hits require an identical numeric input. The preceding
larger-case gate and its evidence remain part of the validation chain.

Do not cancel current job 17013756 to adopt this optimization. If another
execution segment is needed, start in a new run directory after it ends,
reuse only its complete checksummed case records, and preserve incomplete
records in the earlier directory. Retain the eight-worker, 8 GB, two-hour
campaign limit and at most two concurrent jobs. Preflight uses two workers,
4 GB and at most one hour. No work moves to the workstation or login node.

Counts, circuit hashes, coefficients, valid-domain tolerances, compiler
versions, policy ordering and scientific selection rules do not change.
Record memoization in execution metadata and worker logs. Per-policy synthesis
times now include cache hits; do not pool these into an uncached runtime
comparison or claim a measured compiler speedup from mixed attempts.
