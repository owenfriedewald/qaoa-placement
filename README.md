# Exact diagonal completion for QAOA placement

Reproducibility artifact for **Exact Diagonal Completion on Reachable Subspaces:
Application to QAOA Placement**, by Owen Friedewald, Ali Shiri Sichani, and
Chi-Ren Shyu, University of Missouri.

**Archival DOI:** [10.5281/zenodo.22970077](https://doi.org/10.5281/zenodo.22970077)

The study uses freedom on unreachable binary labels to reduce the cost of
Manhattan-distance phase circuits without changing their action on valid
placements. It compares weighted-L1 completion, a sparse coordinate recurrence,
alternative exact extensions, synthesis methods, full-circuit resources, and
placement quality. The artifact also contains the separate OpenROAD integration
study on six RTL designs.

## Start here

Read [the paper](paper/main.pdf), then run the integrity and coverage checks:

```sh
python3 reproduce.py verify
```

Verification uses Python's standard library. It checks published file hashes,
case coverage, saved protocol constraints, compiler versions, and circuit
contracts. It does not simulate a quantum circuit or rerun an experiment.
Use Python 3.10 or newer; the recorded experiment environment used Python 3.10.12.

For the experimental software environment, create a fresh virtual environment:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The full recorded dependency lists are in `paper/environments/`. Install them
in isolated environments when reproducing a particular historical campaign;
Qiskit 2.4.1 and 2.5.2 results must remain separate. No IBM account, API token,
or connection to a quantum processor is needed for this artifact.

## What is included

| Material | Location |
|---|---|
| Current paper, bibliography, tables and figures | `paper/` |
| Scientific implementations and scoped tests | `experiments/qaoa_placement/` |
| Synthetic study instances, results, coefficients and protocols | `experiments/qaoa_placement/mixer_reduction/` |
| Physical-design windows, search results and downstream summaries | `experiments/qaoa_placement/eda_bridge/results/` |
| Scientific specifications and execution amendments | `paper/amendments/` |
| Retained scientific execution evidence | `paper/hpc/evidence/` |
| Published-file checksums | `MANIFEST.json` |
| Original versus published file digests | `SOURCE_MAP.json` |

[Data and results](docs/DATA.md) maps the manuscript's claims to the relevant
files and explains which historical comparisons must not be combined.
[Reproduction](docs/REPRODUCING.md) separates checking saved evidence, regenerating
analyses, rebuilding the paper, and rerunning experiments.
[Provenance](docs/PROVENANCE.md) explains the source freezes and public packaging.

## Reproduce analyses or figures

Run compute-intensive work on an allocated compute node. The supplied Slurm
wrapper bounds resources and native thread counts:

```sh
sbatch scripts/reproduce.sbatch figures ../qaoa-placement-figures
sbatch scripts/reproduce.sbatch analyze ../qaoa-placement-analysis
```

Both commands create a new output directory; they leave this release intact.
`figures` plots stored summaries and recomputes the cohort figure's intervals. `analyze` recomputes derived quantities from
saved results, including confidence intervals and some state reconstruction;
it does not launch a new training campaign. Neither command is needed to read
or verify the published results.

## Scope

The completion construction is exact on the specified valid pairs, up to the
reported numerical tolerances. Its gate savings depend on synthesis. Full-circuit
resource rankings depend on the mixer and topology. The placement experiments
do not establish a quantum solver advantage. OpenROAD results demonstrate
integration of selected local assignments; they do not measure a downstream
benefit caused by diagonal completion.

## Citation and licenses

Please cite the artifact DOI when using the code or data. Machine-readable
citation metadata is in [CITATION.cff](CITATION.cff).

Original code is licensed under [MIT](LICENSE). Original data and documentation,
including the authors' manuscript source, are licensed under
[CC BY 4.0](LICENSE-DATA). Third-party material retains its own license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgments

The computation for this work was performed on the high performance computing
infrastructure provided by Research Support Services at the University of
Missouri, Columbia MO. DOI: [10.32469/10355/97710](https://doi.org/10.32469/10355/97710).

The computation for this work was performed on the University of Missouri’s
Quantum Innovation Center, in partnership with IBM Quantum and facilitated by
Research Support Solutions at the University of Missouri, Columbia MO.
DOI: [10.32469/10355/107781](https://doi.org/10.32469/10355/107781).

This work received no funding. The authors declare no conflicts of interest.
