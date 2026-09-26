"""Bounded Slurm-only execution wrapper around the frozen ACM routing study.

Fresh subprocesses isolate native crashes; failure stops the run. Scientific
source is unchanged. Preflight results are never merged into full results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "batch", "case"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", choices=("frozen", "fresh"), default="frozen")
    parser.add_argument("--instance")
    parser.add_argument("--seed", type=int, nargs="+")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("This wrapper runs only in a Slurm allocation, never on the workstation/login node")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "QISKIT_NUM_PROCS"):
        os.environ[name] = "1"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    import run_acm_confirmation as campaign
    protocol = json.loads((campaign.DEFAULT / "protocol.json").read_text())
    if protocol["code_sha256"] != campaign.source_hashes():
        raise RuntimeError("Frozen scientific source hash mismatch")
    cohorts = {"frozen": campaign.load_manifest(campaign.FROZEN),
               "fresh": campaign.load_manifest(campaign.DEFAULT / "benchmark_manifest.json")}
    cohorts = {k: [r for r in v if int(r["num_cells"]) == 4 and int(r["num_sites"]) == 6]
               for k, v in cohorts.items()}
    if args.command == "case":
        if args.out.exists():
            raise FileExistsError(args.out)
        instance = next(r for r in cohorts[args.cohort] if r["instance_id"] == args.instance)
        print(f"case {args.cohort} {args.instance} seeds={args.seed}", flush=True)
        rows = campaign.route_case(instance, protocol, args.cohort, args.seed or protocol["routing_seeds"])
        campaign.write_csv(args.out, rows)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    # The scientific environment helper expects the protocol beside its output.
    import shutil
    for name in ("protocol.json", "benchmark_manifest.json"):
        shutil.copy2(campaign.DEFAULT / name, args.out / name)
    campaign.environment(args.out, "routing", protocol)
    record = dict(job_id=os.environ["SLURM_JOB_ID"], host=os.uname().nodename,
                  command=sys.argv, wrapper_sha256=sha(Path(__file__)),
                  scientific_changes="none", per_case_timeout_seconds=1200,
                  native_threads=1, concurrency=1, cases=[])
    tasks = [("frozen", r) for r in cohorts["frozen"][:2]] if args.command == "preflight" else [
        (cohort, r) for cohort in ("frozen", "fresh") for r in cohorts[cohort]]
    for cohort, instance in tasks:
        name = f"{cohort}_{instance['instance_id']}"
        path = args.out / f"{name}.csv"
        cmd = [sys.executable, "-X", "faulthandler", "-B", "-u", str(Path(__file__).resolve()), "case",
               "--out", str(path.resolve()), "--cohort", cohort, "--instance", instance["instance_id"]]
        if args.command == "preflight":
            cmd += ["--seed", "101"]
        started = time.monotonic()
        with (args.out / f"{name}.log").open("w") as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        record["cases"].append(dict(cohort=cohort, instance_id=instance["instance_id"],
                                    returncode=code, seconds=time.monotonic()-started,
                                    output_sha256=sha(path) if path.exists() else None))
        (args.out / "execution.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: returncode={code}", flush=True)
        if code != 0:
            raise SystemExit("Routing case failed; evidence saved. Diagnose before submitting the corpus.")


if __name__ == "__main__":
    main()
