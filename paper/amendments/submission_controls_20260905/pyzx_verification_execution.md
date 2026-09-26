# Execution correction: preserve PyZX verification with ordered candidate indexing

The first phase attempt timed out on cases 40–42. Its retry reuses 40 complete
cases and has not completed its first larger cases. A bounded debugger trace
of retry job 17010300, task 40, identifies the active call as
`Circuit.verify_equality -> full_reduce -> match_pivot_gadget`, specifically
membership of an edge in the candidate list in PyZX 0.10.3. Circuit synthesis
has completed at that point. The original beam acceleration does not address
this independent verification bottleneck. All original and retry evidence,
including diagnostic traces, must be retained.

Before implementing or testing this correction, freeze this note and its hash.
The intended correction changes only the verifier's candidate container:

- Preserve the initial sequence produced by `Counter(candidates_set).elements()`.
  Its operations are length, pop-last, membership and remove-first. An indexed
  doubly linked sequence with a per-value deque can implement these same
  operations, including duplicate edges, without linear membership scans.
- Guard the exact source of the pinned upstream matcher and replace only that
  candidate-construction expression in an isolated function namespace. Use the
  replacement only during `Circuit.verify_equality`; the initial reduction,
  extraction, returned circuit, equality criterion, phases and rewrite order
  remain those of the frozen PyZX flow. Never modify the installed dependency.
- Test exact operation traces against Python lists, including duplicates;
  compare ordered match results and graph mutations against the unmodified
  matcher on small seeded graphs; compare the original and accelerated equality
  outcomes for equal and deliberately unequal small circuits.
- Require returned circuit-hash matches for every PyZX policy in all 40
  completed original cases (200 circuits), plus the independent small Qiskit
  unitary checks. Then run the first previously timed-out case as a diagnostic
  preflight under the unchanged acceptance criteria. No failed equality check
  can be admitted or replaced by Gray counts.
- Freeze implementation/test hashes before these validation runs. If equivalence
  checks fail, retain their evidence and do not use the correction.

Cancel the still-running, nonprogressing phase retry by its exact job ID after
recording this diagnostic correction, to release its allocation for validation;
this is a diagnosed recurrence of the original timeout, not a numerical
selection rule. Keep its completed case copies and all logs. Resume only after
validation, in a new directory with source hashes. Full coverage remains
required. No scientific instance, objective, optimizer, circuit, compiler
version, endpoint or budget-selection rule changes. Timing records must
identify the verification implementation and exclude the briefly debugger-paused
attempt from performance comparisons. Native pools remain one, at most eight
workers per job and two concurrent jobs, 8 GB and two hours per campaign job.
