# Provenance of this public artifact

The study records scientific specifications before their own execution,
source digests, case-level results, and subsequent execution corrections.
Some robustness analyses were specified after earlier outcomes were known.
They are not external preregistrations or independent application samples.
The manuscript describes that sequence and its statistical implications.

This release is a curated scientific snapshot, not the original development
repository or its Git history. Internal correspondence, manuscript drafts,
private review material, machine-specific logs and credentials are excluded.
No experimental result was selected for inclusion according to whether it
favored the proposed method.

`MANIFEST.json` hashes the files in this public distribution. `SOURCE_MAP.json`
records the original and published digests of copied scientific material,
including a description of each publication transformation. These are distinct
from the source hashes frozen before experiments.

The scientific specification documents retain their original bytes. Their
input ledgers are explicitly **public subsets** containing unchanged,
redistributed scientific inputs. Internal manuscript/review inputs and
metadata that required sanitization are omitted from those subsets. The
adjacent `SHA256SUMS` describes the public specification/ledger pair; it is not
represented as the original pre-execution checksum ledger.

Public metadata removes local filesystem paths and machine/account execution
fields. Numerical observations, circuit counts, coefficients, seeds and
protocol parameters are retained. Scientific code is copied without changing
its algorithms. The public validation entry point uses the public manifest
and checks a standalone manuscript directory instead of a private Overleaf
working copy. The campaign collectors retain their coverage, source,
compiler-version and circuit-contract checks.

The separate manuscript's citations to related work do not imply that this
artifact contains the code or data for those papers.
