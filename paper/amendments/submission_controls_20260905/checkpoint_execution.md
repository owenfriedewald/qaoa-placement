# Execution correction: checkpoints and equivalent beam scoring

The original jobs retain their source snapshot, logs and every successful case.
After roughly 40 minutes, the phase job had completed 40/160 cases and the
budget job had not yet completed a whole-instance task (375 routed circuits).
Measured peak memory was about 1.1 GB and 1.5 GB respectively across four
workers. Both were using their CPUs; this was not a stalled login-node process.

Prepare the following execution correction before its validation/use. It changes
no instance, coefficient objective, circuit, seed, layer, parameter contract,
compiler, endpoint or outcome-based decision:

1. Budget tasks checkpoint one method/depth (15 routes), instead of one whole
   instance (375 routes). Completed original case rows can be split into unchanged
   groups, with their source CSV hash and original job recorded. Earlier attempts
   and any partial failures remain separate.
2. Beam scoring is computed incrementally. Every term is separated by RZ, so the
   only cancellation added at a boundary is twice the common prefix of the two
   ordered control lists, when their parity targets agree. Appending support B
   therefore adds 2(|B|-1)-2*common_prefix CX. This is the archived metric exactly.
   Candidate rankings, width four, pool ten, starting records and stable ties
   are unchanged. Static candidate rankings are cached. The old implementation
   is not modified or relabeled.
3. Before use, compare the new scorer with the original on seeded duplicate/tie
   tests and exhaustive small support boundaries; compare complete circuit hashes
   against every successful original phase case available in a frozen checkpoint
   ledger. Also compare all 15 shared Sherbrooke p=3 route rows from the split
   budget runner against the passed preflight, including counts and hashes.
4. On an actual task timeout or job-limit failure, resume only uncompleted work
   in a new run directory. Cancel obsolete dependents by exact job IDs. Do not
   overwrite the old source or outputs. Full aggregates require complete coverage.
5. Retain the specification's maximum eight workers per job, 8 GB and two-hour
   job limit. Up to two compute jobs may run concurrently (16 workers total),
   still with native pools one. The first attempt voluntarily used four per job;
   increasing to eight after observed low memory use is an execution adjustment,
   not a change to an experimental endpoint. All work remains on Hellbender.

The correction was implemented and source-frozen before its validation/use.
The initial scientific amendment preceded both the original implementation and
all experimental outcomes. This separate record does not retrospectively replace
that initial specification or its hashes. Synthesis timing from the two scorer
implementations must be distinguished; LP solve-time records are separate.
