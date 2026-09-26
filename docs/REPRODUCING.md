# Reproducing the study

## 1. Verify the deposited evidence

From the repository root:

```sh
python3 reproduce.py verify
```

This standard-library command checks every manifest-listed file and the
scientific collectors' expected task coverage, saved source hashes, output
constraints, compiler versions and circuit contracts. It does not require
Qiskit and does not execute experiments. Verification will report a mismatch
if you edit a deposited file; keep a pristine copy for audit.

## 2. Install dependencies

Use a fresh Python 3.10 environment and `requirements.txt`. Complete recorded
package lists are in `paper/environments/`. PyZX's hashed wheel list describes
the recorded Linux environment; it is not a portable, complete lock for every
platform. Install the core requirements for ordinary use and retain the exact
recorded environment when testing numerical/compiler reproducibility.

The earlier Qiskit 2.4.1 phase audit has a separate environment record.
Do not merge its counts with the 2.5.2 comparisons. Compiler output may change
with versions even when a circuit is logically equivalent.

## 3. Regenerate figures and analyses

On a Slurm compute node, with the environment active:

```sh
sbatch scripts/reproduce.sbatch figures ../placement-figures
sbatch scripts/reproduce.sbatch analyze ../placement-analysis
```

The wrapper requests one CPU, 4 GB, and one hour, with numerical library
threads fixed at one. Adjust the partition/account to your institution's
scheduler if required. Output paths must not already exist. Each command
copies the artifact into that output directory and writes regenerated
material there. Original evidence remains unchanged.

`figures` uses stored summaries and coefficient vectors, recomputes the saved
cohort figure's bootstrap intervals, and redraws the manuscript's
analysis figures. `analyze` runs the successive confirmation, replication,
control and submission analyses. It includes statistical resampling and
reconstruction of saved states/coefficients, and can take substantially longer
than verification. It does not optimize new QAOA parameters or launch an
experimental campaign. Regenerated files can have new PDF metadata or changed
floating-point formatting; compare numerical content rather than PDF hashes.

The retained figures and tables are sufficient for compiling the manuscript;
recomputation is optional.

## 4. Compile the manuscript

Use a complete TeX installation supporting the included ACM class, or Tectonic
0.17.0. On an allocated compute node, from `paper/`:

```sh
tectonic main.tex
```

Alternatively import the contents of `paper/` into Overleaf and select
`main.tex`. The class files and their sources/notices are included. No analysis
script is needed to compile the retained manuscript inputs.

## 5. Rerun experiments

These commands perform substantial computation and are deliberately separate
from verification and analysis. Use a compute allocation, keep native threads
at one, and always use a new output directory. Review each runner's `--help`
and the corresponding saved protocol before allocating resources.

Add these directories to `PYTHONPATH`:

```sh
export PYTHONPATH="$PWD:$PWD/paper:$PWD/experiments/qaoa_placement:$PWD/experiments/qaoa_placement/mixer_reduction"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Primary runner entry points, under `experiments/qaoa_placement/mixer_reduction/`:

| Campaign | Runner |
|---|---|
| Original confirmation | `run_acm_confirmation.py` (`freeze`, `quality`, `route`; use `--workers 1`) |
| Initial completion | `run_exact_phase_completion.py` |
| Replication/transfer | `acm_gap_study.py` |
| Penalty calibration | `acm_penalty_calibration.py` |
| Corrected classical accounting | `acm_classical_accounting_hpc.py` |
| Ring/workspace/coordinate controls | `acm_reviewer_study.py` |
| Extension, structured and resource/quality controls | `acm_submission_study.py` and its checkpoint/resume variants |

Some runners require a `payload.sha256` source snapshot ledger. Create it on
the compute node before launching a new run:

```sh
python3 reproduce.py snapshot --output ../placement-run-inputs.sha256
```

Copy that ledger to `payload.sha256` in the new working copy used for the run.
It describes that new run's actual source; do not replace a historical ledger
or label a modified source as the original experiment. Protocols and task
lists in the saved campaign directories identify the configurations to replay.
The chapter-level claim is the aggregate over the declared cases, not a smoke
run or a single selected instance.

## 6. Physical-design integration

The public package includes extracted windows and complete local cost
landscapes, selected assignments, search records, downstream measurement
summaries, and the scripts that prepare and inspect OpenROAD cases. Reading
and analyzing these records does not require OpenROAD.

Repeating physical implementation additionally requires the upstream OpenROAD
Flow Scripts and Nangate45 design inputs. The recorded ORFS commit is
`be0dca0b1fd4`; the recorded image is:

```
openroad/orfs@sha256:d995618be9f2bcdfa5538b885123463070dfbf178bea1818716d4652fe0fa380
```

The six designs are GCD, AES, Ibex, Mempool Group, SweRV and TinyRocket.
Use the scripts in `experiments/qaoa_placement/eda_bridge/` to prepare selected
assignment cases and run them against your own paths to the upstream baseline
databases. `prepare_downstream_routing.py`,
`prepare_downstream_classical_comparison.py`, and
`run_downstream_routing_case.py` expose their inputs through command-line
arguments. The recorded containers, RTL/library sources and complete design
databases are external dependencies, not bundled binaries. Local paths in
published historical metadata are placeholders; they are not portable inputs.
