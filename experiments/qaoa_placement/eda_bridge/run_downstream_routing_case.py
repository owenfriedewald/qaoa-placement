"""Run one preregistered CTS/global-route case inside the pinned ORFS image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence


OPENROAD = "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
CONGESTION_TOTAL = re.compile(
    r"^Total\s+(?P<resource>\d+)\s+(?P<demand>\d+)\s+"
    r"(?P<usage>[0-9.]+)%\s+(?P<h>\d+)\s*/\s*(?P<v>\d+)\s*/\s*(?P<total>\d+)\s*$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: Sequence[str], env: dict[str, str], log: Path, cwd: Path | None = None) -> None:
    with log.open("a") as handle:
        handle.write("QEDA_COMMAND " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def parse_congestion(log: Path) -> dict[str, float | int]:
    matches = []
    for line in log.read_text(errors="replace").splitlines():
        match = CONGESTION_TOTAL.match(line.strip())
        if match:
            matches.append(match.groupdict())
    if not matches:
        raise ValueError(f"no final congestion total in {log}")
    values = matches[-1]
    return {
        "resource": int(values["resource"]),
        "demand": int(values["demand"]),
        "usage_percent": float(values["usage"]),
        "horizontal_overflow": int(values["h"]),
        "vertical_overflow": int(values["v"]),
        "total_overflow": int(values["total"]),
    }


def cell_retention(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    retained = [
        row
        for row in rows
        if int(row["expected_x"]) == int(row["final_x"])
        and int(row["expected_y"]) == int(row["final_y"])
        and row["expected_orient"] == row["final_orient"]
    ]
    return {"cell_count": len(rows), "retained_count": len(retained), "all_retained": len(rows) == len(retained)}


def slurm_metadata() -> dict[str, str]:
    return {
        "job_id": os.environ.get("QEDA_SLURM_JOB_ID", ""),
        "array_job_id": os.environ.get("QEDA_SLURM_ARRAY_JOB_ID", ""),
        "array_task_id": os.environ.get("QEDA_SLURM_ARRAY_TASK_ID", ""),
        "partition": os.environ.get("QEDA_SLURM_JOB_PARTITION", ""),
        "cpus_per_task": os.environ.get("QEDA_SLURM_CPUS_PER_TASK", ""),
        "memory_per_node_mb": os.environ.get("QEDA_SLURM_MEM_PER_NODE", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--flow-home", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    case = plan["cases"][args.case_index]
    case_id = case["case_id"]
    design = case["design"]
    work_home = (args.work_root / case_id).resolve()
    artifact_dir = (args.artifact_root / case_id).resolve()
    result_dir = work_home / "results" / "nangate45" / design / case["variant"]
    log_dir = work_home / "logs" / "nangate45" / design / case["variant"]
    report_dir = work_home / "reports" / "nangate45" / design / case["variant"]
    object_dir = work_home / "objects" / "nangate45" / design / case["variant"]
    for path in (result_dir, log_dir, report_dir, object_dir, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)
    execution_log = artifact_dir / "execution.log"
    started = time.time()
    status = "failed"
    error = ""
    try:
        source_odb = args.input_root / design / "qeda_3_4_legalize_only.odb"
        source_sdc = args.input_root / design / "2_floorplan.sdc"
        env = os.environ.copy()
        env.update(
            {
                "QEDA_INPUT_ODB": str(source_odb),
                "QEDA_OUTPUT_ODB": str(result_dir / "3_place.odb"),
            }
        )
        run([OPENROAD, "-exit", str(Path(case["apply_tcl"]).resolve())], env, execution_log)
        shutil.copy2(source_sdc, result_dir / "3_place.sdc")

        common = [
            "make",
            f"DESIGN_CONFIG=./designs/nangate45/{design}/config.mk",
            f"FLOW_VARIANT={case['variant']}",
            f"WORK_HOME={work_home}",
            f"NUM_CORES={plan['num_cores']}",
            f"GRT_SEED={plan['grt_seed']}",
            "SKIP_CTS_REPAIR_TIMING=1",
            "SKIP_INCREMENTAL_REPAIR=1",
            "OPT_POST_GRT_WNS=1",
            "SKIP_ANTENNA_REPAIR=1",
        ]
        run([*common, "do-4_1_cts"], env, execution_log, args.flow_home)
        run([*common, "do-4_cts"], env, execution_log, args.flow_home)
        run([*common, "do-5_1_grt"], env, execution_log, args.flow_home)

        positions = artifact_dir / "final_cell_positions.csv"
        inspect_env = env | {
            "QEDA_INPUT_ODB": str(result_dir / "5_1_grt.odb"),
            "QEDA_OUTPUT_CSV": str(positions),
        }
        run([OPENROAD, "-exit", str(Path(case["inspect_tcl"]).resolve())], inspect_env, execution_log)

        cts_json = log_dir / "4_1_cts.json"
        grt_json = log_dir / "5_1_grt.json"
        grt_log = log_dir / "5_1_grt.log"
        cts = json.loads(cts_json.read_text())
        grt = json.loads(grt_json.read_text())
        result = {
            "schema_version": "qeda-downstream-routing-case-v2",
            "case_index": args.case_index,
            "case": case,
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "execution_commit": os.environ.get("QEDA_EXECUTION_COMMIT", "unknown"),
            "container_sha256": os.environ.get("QEDA_CONTAINER_SHA256", "unknown"),
            "slurm": slurm_metadata(),
            "source_odb_sha256": sha256(source_odb),
            "source_sdc_sha256": sha256(source_sdc),
            "cts_metrics": cts,
            "global_route_metrics": grt,
            "congestion_total": parse_congestion(grt_log),
            "cell_retention": cell_retention(positions),
            "native_sha256": {
                "4_1_cts.json": sha256(cts_json),
                "5_1_grt.json": sha256(grt_json),
                "5_1_grt.log": sha256(grt_log),
            },
        }
        for source in (cts_json, grt_json, grt_log):
            shutil.copy2(source, artifact_dir / source.name)
        (artifact_dir / "case_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (artifact_dir / "case_result.json").write_text(
            json.dumps(
                {
                    "schema_version": "qeda-downstream-routing-case-v2",
                    "case_index": args.case_index,
                    "case": case,
                    "status": status,
                    "error": error,
                    "elapsed_seconds": time.time() - started,
                    "execution_commit": os.environ.get("QEDA_EXECUTION_COMMIT", "unknown"),
                    "container_sha256": os.environ.get("QEDA_CONTAINER_SHA256", "unknown"),
                    "slurm": slurm_metadata(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise
    finally:
        print(json.dumps({"case_id": case_id, "status": status, "error": error}))


if __name__ == "__main__":
    main()
