# Prospective reviewer-control amendment, 4 September 2026

Status at specification: all earlier confirmation, gap-closure, calibration and
query-accounting outcomes have been seen. No experiment in this amendment has
been run. These are post-review robustness controls, not a new independent
confirmation of the selected method. This specification is committed and its
SHA-256 recorded before implementation and execution. Implementation snapshots
are separately hashed before their jobs. Amendments and failed attempts remain
available; no outcome-dependent exclusions or silent source-hash updates.

## Scope and sequencing

1. Freeze this document and exact input-file hashes before new code or outcomes.
2. Validate, then launch ring quality and order-preserving multi-ancilla routing.
3. While those run, generate the two existing-data analyses and coordinate LP
   control. Gate the manuscript's quality framing on the ring results.
4. Archive the current manuscript, rewrite, build and verify on Slurm.

All simulation, optimization, transpilation, plotting, and TeX builds run on
Hellbender compute nodes. Maximum eight single-thread workers, 8 GB and two hours
per campaign job; preflight two CPUs/4 GB/one hour. No hardware or paid API calls.
`qrl/` and historical scientific inputs remain read-only. A failed case blocks a
complete aggregate; retries need a new run ID and a disclosed reason.

## A. Matched-depth Row-XY ring

The ring uses ordered edges (0,1),(1,2),...,(m-2,m-1),(m-1,0), once per row per
layer, each with RXX(2 beta) RYY(2 beta). No ring-order search or retuning of token.
The connected ring preserves row weight one, but not collision freedom. Use p=3,
five live parameters, prepared basis states, omitted first cost phase, collision
penalty 2 lambda, CVaR alpha=.25, COBYLA at most 200 calls, rhobeg=.2, tol=.001.
Match the archived initial placements and parameter seeds exactly within each
cohort. Report unconditional optimum mass, legal mass, improvement mass, actual
calls, training shots, and returned parameters. Exact optimum is diagnostic.

Evaluation cohorts: all 36 four-cell/six-site September confirmation cases at
lambda=5, all three starts and seeds 11,17,23,31 (432 ring runs); all 36 gap-study
cases at 4c/6s,5c/8s,6c/8s, all three starts, seeds 41,53, same analytic training
and four-cell finite2048 conditions as archived (288 ring runs per penalty policy).
For gap cases evaluate fixed5 and an independently selected ring penalty. Select
ring penalty on the same 18 disjoint calibration-development cases, four original
candidates (5,B/4,1.000001B,4B), three starts and seed61, analytic CVaR (216 runs),
maximizing mean unconditional optimum mass separately by size, ties in candidate
order. Freeze selection before ring evaluation. Complete-XY uses its archived
independent selection. Selection costs remain separate. If fixed5 is selected,
retain both labeled conditions; never count them as independent observations.

Primary endpoint: token-minus-ring mean optimum mass on the 12 gap 4c/6s analytic
instances using independently calibrated Row comparators. Average starts/seeds
within instance. Report fixed5 and calibrated comparisons side by side, plus the
36-case confirmation replication, finite shots and larger sizes separately.
For each cohort define closure=(ring-complete)/(token-complete) using cohort means;
report only if the original denominator is positive, otherwise mark undefined.
"Ring competitive" for the editorial gate means it closes at least 75% of the
positive calibrated four-cell analytic gap OR matches/exceeds token's mean.
"Ring wins" means ring has larger mean AND the paired 95% interval for ring-minus-
token lies strictly above zero. Smaller gains, unresolved intervals, and differing
cohorts must be reported, not collapsed into a binary scientific conclusion.
This 75% threshold is an editorial rule, not a significance or equivalence test.

Uncertainty: 20,000 percentile bootstrap draws resampling instances within each
fixed family, seed 1144004; 95% intervals. Restarts and starts are not replicates.
Retain the previously disclosed confirmation smoke case and also report exclusion
sensitivity for that one case. No stopping based on statistical significance.

## B. Multi-ancilla token mixer

Compare one clean ancilla with floor(m/2) clean ancillas. Preserve the current
line/EMPTY-prioritized ordered product exactly: move only disjoint register swaps
past each other. Assign each swap the earliest dependency-respecting stage and
the lowest available ancilla within that stage. All ancillas must uncompute to
zero. Do not substitute an edge-colored ansatz that reorders overlapping swaps.
Report the dependency critical path as well as isolated mixer logical CX/depth,
full-circuit logical CX/depth, active width, routed two-qubit count/total depth/
two-qubit depth. Exact equivalence and ancilla cleanliness tested on small full
states and superpositions with several non-special angles before routing.

Route all 36 confirmation four-cell instances with original zero/beam token
phases on Qiskit2.5.2 FakeSherbrooke, seeds101..105. Recompute serial, parallel,
and complete-XY under that same five-seed contract (540 rows). Do not compare a
five-seed new statistic directly against the archived twenty-seed headline.
Also route all 12 gap six-site phase instances on line25 and grid25, original
seeds211,223,227,229,233, with Gray/L1 serial and parallel token and complete/ring
Row-XY at fixed5 (480 rows). All full circuits are symbolic, prepared, unmeasured,
level3 SABRE. Synthetic targets share rz/sx/x/cx basis; ECR and CX never pooled.
Primary endpoint: per-instance seed-median parallel/serial routed depth change;
secondary counts, widths, and changes relative to each Row comparator. Report
all instances, even if parallel workspace worsens routing. More ancillas alone
need not remove the serial critical path of overlapping swaps.

## C. Existing-data analyses

Free fraction F=1-m^2/2^(2 ceil(log2 m)) of two-register truth-table entries.
Plot within-instance L1/zero CX reduction versus F, separate historical 100-case
fixed-mask and fresh 60-case beam/Gray synthesis. Include every full-capacity
control. Display points and site-count summaries, not a causal fitted law:
geometry, labels, size, and synthesis also affect savings. F is invalid-code
freedom, not vacant-site fraction or EMPTY degeneracy.

Same-size shrinkage: show token-minus-complete-XY instance differences for the
archived development cohort, the 36-case confirmation, and the 12-case gap
4c/6s analytic cohort, separating optimization budgets/starts and generators.
Only token/Row observations within the same instance/protocol are paired.
Never connect different-cohort instances as if paired or attribute all changes
to selection bias: generator, geometry, seeds and protocol differ. Display raw
instance effects and cohort means. If the development rows do not support a
matched protocol, label them descriptive historical context and omit a pooled
shrinkage estimate. Confirmation smoke-case sensitivity remains disclosed.

## D. Coordinate-separable Manhattan completion

For each exact coordinate/label set in the historical 100-case and gap 60-case
audits, independently solve the existing weighted-L1 LP for |x_a-x_b| and for
|y_a-y_b| on the same binary site-label domain; sum their coefficient vectors.
Compare with a fresh solve of the dense joint-distance LP and zero extension.
Same constant-free weights, sign-split variables, HiGHS, coefficient threshold
1e-10, all valid pairs including equality. Every combined residual must be below
1e-8. Save both component vectors, combined and dense vectors, objective values,
solve times, dimensions and constraint sizes. Dense objective cannot exceed
the feasible summed-vector objective beyond numerical tolerance; full-capacity
solutions must agree. Synthesize dense and separable through identical symbolic
flows: historical fixed-mask on 100 cases, beam and Gray on 60 cases (440 phase
rows). Recompute comparison sides under Qiskit2.5.2, never mix historical2.4.1.

Primary endpoints: paired separable/dense logical CX change by corpus, synthesis
and site count, and valid-distance residual. Secondary: weighted-L1 objective,
nonzero support, logical depth and solve time. Tie means equal integer CX; report
all wins/ties/losses. Independently solving coordinates on the same site-label
domain does NOT reduce the O(m^4) dense storage order. If it matches dense CX,
claim an additive construction control, not that scalability has been solved.
Coordinate-register compression would require a different encoding or a costed
reversible lookup and is outside this amendment. No new QAOA optimization needed:
exact valid-distance equivalence preserves the ideal algorithm.

## Validation and editorial disposition

Preflight geometry seeds 1177101..1177103, not evaluation cases. Validate ring
reduced simulation against full-state circuits, all row weights, both mixer
implementations on arbitrary small valid-state superpositions, order dependencies,
coordinate sum and valid residuals, coverage and source hashes. Meaningful failed
checks stop the affected campaign. Preserve every partial execution log.

Completion becomes contribution (i) in either outcome. If ring is competitive,
lead with exact completion and use placement as the application; quality becomes
a compact baseline-dependence result. Otherwise retain the bounded quality
observation without attributing the difference solely to legality. Put all
same-size/transfer quality controls together, merge completion replications,
move the detailed OpenROAD scope check to an appendix, and remove repetitive
discussion. Aim for approximately 30% less prose where supported; retain enough
methods, adverse evidence and uncertainty to reproduce and assess all claims.
New findings must not be described as prospectively selected before prior data.
Update artifacts, PDF, Overleaf, evidence matrix and evaluation report together.
