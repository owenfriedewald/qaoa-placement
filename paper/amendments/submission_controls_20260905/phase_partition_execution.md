# Fixed-corpus parallel phase partition — prospective execution addendum

This execution addendum leaves `spec.md` and its 160-case/4,960-row endpoint
unchanged. It is specified before implementing or running the partition.
Job 17013756 is running the complete original task list, in which historical
cases occupy indices 0--99 and fresh cases 100--159. At its 29:46 inspection,
52 cases were complete, including 40 reused cases. The memoization preflight
is separately running; it must pass before the fresh partition starts.

To use the authorized second eight-worker job slot, allow a fresh-only
partition to run concurrently with the existing campaign after the
memoization gate. This supersedes only the sequential-start instruction in
`pyzx_memo_execution.md`; identical-input memoization and all scientific
settings remain unchanged. Keep current job 17013756 running to completion
or its existing Slurm limit. Do not edit its task list or source.

Construct the fresh task list by taking exactly indices 100--159 from the
immutable full task manifest, checking their `fresh` cohort tags and unique
instance IDs. Record the full-manifest hash and global index mapping. Execute
the unchanged memoization runner on this 60-task list with eight workers,
8 GB, a two-hour limit and the pinned environment. At most two jobs and
sixteen experiment workers may run concurrently.

After both attempts end, assemble a new 160-task checkpoint with fixed origin
rules: historical cases come from job 17013756; fresh cases come from the
fresh partition. Copy only complete case files whose checksums match their
source ledgers. Preserve coefficient JSON, source job/index, environment
hashes and payload hashes. If the original full job also computes a fresh
case, retain it and compare all non-timing CSV fields against the selected
fresh-partition copy. Do not choose the cheaper output. A discrepancy must
be investigated and recorded, not silently pooled or selected away.

Missing or timed-out cases remain missing in the assembled checkpoint. Run
the existing memoization resume wrapper on every missing task in the frozen
160-case list. Preserve all partial attempts and overlapping work. Only a
complete, validated 4,960-row corpus may support the final aggregate.

The assembler is standard-library bookkeeping on a Slurm compute node.
Source/manifest hashes and index checks precede use. The complete artifact
retains both source attempts and the assembly record. No new scientific
cases, endpoints, tolerances, circuit contracts or compiler versions are
introduced, and mixed cache timings are not a compiler speed benchmark.
