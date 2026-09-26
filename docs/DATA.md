# Data and manuscript map

Paths below are relative to `experiments/qaoa_placement/mixer_reduction/`
unless stated otherwise. CSV files have a header row. JSON files contain
protocols, task definitions, coefficient vectors, environments or execution
ledgers. Instance identifiers connect measurements to their geometry and
initialization. Optimizer seeds, starting placements and routing seeds are
repeated conditions within an instance, not independent samples.

| Manuscript evidence | Saved input/results |
|---|---|
| Initial 100-geometry completion audit | `exact_phase_completion_20260904/` |
| 60-geometry replication, topology controls and larger/sampled quality | `acm_gap_closure_20260904/` |
| Five exact extensions, approximation and numeric synthesis; Table 1 | `acm_submission_controls_20260905/phase/` (4,960 rows) |
| Coordinate-separated completion; Table 2 | `acm_reviewer_controls_20260904/phase/` (440 rows) |
| Coordinate recurrence; Table 3 | `acm_submission_controls_20260905/structured/` (46 records) |
| Full-circuit and component resources; Tables 4–5 | `acm_submission_controls_20260905/homogeneous/` (2,400 rows), with matching synthetic-target rows from `acm_reviewer_controls_20260904/workspace/` |
| Resource-budget depth selection; Table 6 | `acm_submission_controls_20260905/budget/` (4,500 rows) and `quality/` (1,224 rows) |
| Initial quality confirmation | `acm_confirmation_20260904/quality_run_level.csv` (1,152 rows) |
| Penalty tuning/evaluation | `acm_penalty_calibration_20260904/` (216/288 rows) |
| Ring tuning/evaluation | `acm_reviewer_controls_20260904/ring/` (216/1,008 rows) |
| Corrected classical query accounting | `acm_classical_accounting_20260904/` (3,888 rows) |
| Earlier homogeneous routing and integrated completion | `acm_routing_qiskit252_20260904/`, `acm_integrated_phase_completion_20260904/` |
| OpenROAD integration; Table 7 | `experiments/qaoa_placement/eda_bridge/results/` |

The 160 geometries comprise the original 100 and replication 60. Reusing them
for extension and synthesis controls does not create a third independent
cohort. The 96 cases with unused binary labels provide completion freedom;
the remaining 64 are full-capacity controls.

## Derived results

`paper/tables/` contains tabular analyses as CSV and the manuscript's TeX tables.
`paper/*_summary.json` contains the corresponding summaries and uncertainty
estimates. `paper/figures/` contains the retained publication figures.
The figure/table generators are in `paper/build_*.py`.

## Historical and corrected data

Keep compiler versions, numeric versus symbolic angles, mixer schedules,
preparation conventions and phase synthesis methods separate. In particular:

- The initial phase audit used Qiskit 2.4.1 and fixed mask order. The extension
  comparison uses 2.5.2; its counts are compared within that version.
- Historical resource results that treated symbolic parameters asymmetrically
  are not the main full-circuit comparison.
- The original classical query counts in `acm_gap_closure_20260904/` are retained
  for provenance. Use `acm_classical_accounting_20260904/` for corrected budgets.
- Confirmation and transfer cohorts are unpaired with each other. Pair methods
  only within the same instance and protocol.
- The confirmation smoke case and its exclusion analysis remain disclosed.
- The OpenROAD windows are nested within six designs and use a distinct
  objective and schedule. Do not pool them with the synthetic placement study.

Execution corrections and unsuccessful attempts are retained where they bear
on the published scientific comparisons. They are separate from complete
aggregates; validators reject missing cases and inconsistent contracts.
