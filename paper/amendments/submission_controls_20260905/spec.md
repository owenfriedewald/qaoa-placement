# Submission controls, frozen 5 September 2026

This prospective amendment follows inspection of the September 4 outcomes and
the PDF-only review in `paper/feedback/TQC_Review_Invariant_Aware_QAOA_Placement.md`.
It is post hoc robustness work, not independent confirmation. Freeze this file
and the input ledger in a spec-only commit before implementing or running new
experiments. Historical source and failed attempts remain immutable.

## Extension controls (C1, C7, M1)

Reuse all 100 historical and 60 fresh phase cases, preserving site labels and
nets. Recompute every new comparison under Qiskit 2.5.2, logical basis
rz/sx/x/cx, optimization level 3, seed 123. Exact policies: zero, joint weighted
L1, virtual-coordinate Manhattan, nearest-valid-code and mean fill. Virtual
coordinates retain all existing coordinates in their original code order and
append unused integer lattice points in row-major order within the bounding
rectangle, extending rows above it if necessary. Nearest fill maps each invalid
code to the valid code of least Hamming distance (ties: smallest code), then
uses the distance of the mapped pair. Mean fill uses the arithmetic mean of all
m squared valid ordered-pair distances whenever either code is invalid.
None is assumed sparse under arbitrary shuffled labels. Zero is a naive control,
not a proven worst case. All exact valid-pair residuals must be below 1e-8 after
coefficient thresholding at 1e-10.

Compare beam and Gray symbolic flows and generic Qiskit diagonal at gamma=.371.
Add a PyZX phase-gadget simplification/extraction flow, initialized from numeric
Gray, with the version pinned before execution; retain the unoptimized numeric
Gray control. Verify the numeric flow by its diagonal phase-polynomial action
or an independent small full-unitary check before admitting counts. Failures are
reported, never silently replaced by Gray. No mixing numeric/symbolic aggregates.

For L1 and virtual extensions, greedily test removal of nonconstant Walsh terms
in ascending absolute magnitude, breaking ties by mask, accepting a removal
only when the maximum residual on valid pairs remains at most eta times the
valid diameter, eta in {.001,.01,.05}. Report achieved residual, epsilon times
sum absolute net weights, and Gray CX/rotations; these are approximate controls,
not exact competitors or a claim of optimal truncation.

Derive a structured Walsh recurrence for absolute difference on unsigned binary
coordinates and its sum on row-major Cartesian grids. Verify all values for
coordinate widths 1 through 6 and construction-only growth through width 16.
Compare its exact coefficients with the dense transform on small regular
rows/prefixes and Cartesian grids (m=6,8,9,12,16), with zero and L1. This is a
structured subclass, not a scalable solution for arbitrary permuted coordinates.
Report preprocessing seconds, stored coefficient count and support by popcount.

## Homogeneous resources and matched budgets (C2, C3, C6)

On the 12 fresh m=6 phase cases, run serial and parallel Gray/L1 token,
complete-XY and ring-XY p=3 on Sherbrooke at seeds 211,223,227,229,233, matching
the existing line25/grid25 controls. Preserve the earlier zero/beam Sherbrooke
table separately. Include 1q, RZ, 2q, depth and width for full circuits and
standalone preparation, one mixer layer and one phase. Standalone routed
component depths/counts are not additive contributions to an optimized full
circuit; label the distinction explicitly. Report the clean-ancilla primitive
including both controlled-register swaps and its ancilla-free Fredkin lowering.

On all 12 fresh 4c/6s QUALITY cases, calibrate resource budgets without quality
outcomes: token serial Gray/L1 at p=3 versus each Row graph at p=1 through 12,
on each of Sherbrooke,line25,grid25, using the same five routing seeds above.
Use fixed lambda=5 for this separate matched-budget comparison for both graphs;
do not substitute evaluation-selected penalties. Select a single p per graph,
target and resource endpoint as the largest p whose median across instance seed
medians does not exceed token's median. Record undershoot/overshoot and whether
the upper cap binds. This is an at-most-budget comparison, not exact equality.
Fit each distinct selected graph/p only once on the same 12 cases, three starts,
seeds 41,53, CVaR .25, COBYLA200/rhobeg.2/tol.001, analytic training and independent
initial parameters for p (no transfer warm starts). Also run p=1,2,3,4,6 for all
three methods, and token p=3 with a seeded random permutation of the same line
edges, fixed per layer and instance (seed=instance seed+1905+layer).

Primary matched-budget endpoints: unconditional optimum probability,
conditional-on-legal optimum probability (undefined if legal mass is zero),
expected best-of-128 measured shots retaining the known initial placement,
and legal mass. Average seeds/starts within each instance before pairing.
Report mean and median paired differences, W/T/L, family-stratified 20,000-draw
percentile intervals (seed 1209505), and unadjusted/Holm-adjusted paired
Wilcoxon and exact sign tests as descriptive post hoc inference. A Row baseline
wins an endpoint if its paired mean is better; evidence of superiority requires
a corresponding interval excluding zero. No universal encoding claim if any
matched-budget comparison reverses; sampling and optimization budgets are also
reported since circuit matching does not equalize training work.

## Existing-row analysis (C4 and statistical/figure requests)

For every existing ring-table condition report each method's legal mass,
run-wise conditional optimum (never ratio of cohort means), and expected
best-of-128/512. Reconstruct saved final distributions if an endpoint is absent;
do not optimize again. Recompute ring gap closure for each metric, with explicit
direction for costs and undefined denominator when complete already beats token.
Preserve fixed/calibrated baselines. Add the stronger per-instance envelope only
as an explicitly evaluation-oracle sensitivity bound, never a deployable tuned
method. Holm corrections cover all original table conditions per endpoint and
test; report family size and zero handling. Plot per-size paired gate reductions
with dispersion and cohort mean bootstrap intervals; analyze coefficient support
without assuming the review's explanation for coincident medians is correct.
Verify zero/L1 ideal quality neutrality on saved token final parameters using
completed energies on the reduced invariant domain; no reoptimization needed.

## Scope and execution

All numerical work, plotting, scientific tests and TeX builds run under bounded
Slurm on Hellbender. Use checksummed isolated sources, maximum 8 workers, native
thread pools 1, 8 GB memory, two-hour jobs; preflight first. New failures require
a separate execution correction and new run ID. Workstation: editing, AST and
small standard-library integrity checks only. qrl remains read-only.

Noise, domain-wall encodings, industrial optimization and second-optimizer
campaigns are not prerequisites for this exact-synthesis revision: remove
hardware/encoding-superiority claims rather than infer noisy utility from ideal
counts. A full noisy state simulation must not be disguised as an inexpensive
reduced-subspace simulation: faults break its invariant. These are explicit
limitations, not unrun experiments described as negative results.

Rewrite after these controls. Completion need not beat cheap extensions for the
paper to report the outcome: narrow the method's role to the measured cases.
Supply a verified reviewer-access archive; public deposit, funding and author
identifiers require actual author metadata and must not be invented. Move internal
process history to provenance, add reproducible algorithms, and synchronize TeX,
generated claims, PDF and Overleaf before calling the revision reviewable.
