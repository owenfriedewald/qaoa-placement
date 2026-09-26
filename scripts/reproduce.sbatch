#!/usr/bin/env bash
#SBATCH --job-name=qaoa-placement-reproduce
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python3 -B reproduce.py "${1:?Choose figures or analyze}" --output-dir "${2:?Supply a new output directory}"
