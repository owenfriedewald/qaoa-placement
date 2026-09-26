"""Slurm-only bounded parallel execution of unchanged scientific routing cases."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--workers',type=int,choices=range(1,9),default=8)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.workers>int(os.environ['SLURM_CPUS_PER_TASK']):
        raise SystemExit('Requires a Slurm allocation with enough requested CPUs')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        os.environ[name]='1'
    import run_acm_confirmation as c
    protocol=json.loads((c.DEFAULT/'protocol.json').read_text())
    assert protocol['code_sha256']==c.source_hashes()
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.json','benchmark_manifest.json'):
        shutil.copy2(c.DEFAULT/name,args.out/name)
    c.environment(args.out,'routing',protocol)
    environment=json.loads((args.out/'routing_environment.json').read_text())
    assert environment['versions']['qiskit']=='2.5.2'
    record=dict(job_id=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,command=sys.argv,
                wrapper_sha256=sha(Path(__file__)),scientific_changes='none',concurrency=args.workers,
                native_threads=1,per_case_timeout_seconds=1200,cases=[])
    retained=set()
    if args.resume:
        old_env=json.loads((args.resume/'routing_environment.json').read_text())
        for name in ('versions','code_sha256','protocol_sha256','manifest_sha256','coupling_edges_sha256','backend_snapshot_sha256'):
            assert old_env[name]==environment[name],f'Resume environment differs: {name}'
        old=json.loads((args.resume/'execution.json').read_text())
        for case in old['cases']:
            if case['returncode']!=0: continue
            name=f"{case['cohort']}_{case['instance_id']}"
            path=args.resume/f'{name}.csv'
            assert sha(path)==case['output_sha256']
            rows=c.read_csv(path)
            assert len(rows)==40 and {(r['method'],int(r['routing_seed'])) for r in rows}=={
                (m,s) for m in c.METHODS for s in protocol['routing_seeds']}
            shutil.copy2(path,args.out/path.name)
            if (args.resume/f'{name}.log').exists(): shutil.copy2(args.resume/f'{name}.log',args.out/f'{name}.log')
            record['cases'].append(dict(case,retained_from_job=old['job_id']))
            retained.add((case['cohort'],case['instance_id']))
    tasks=[]
    for cohort, manifest in [('frozen',c.FROZEN),('fresh',c.DEFAULT/'benchmark_manifest.json')]:
        for row in c.load_manifest(manifest):
            if int(row['num_cells'])==4 and int(row['num_sites'])==6 and (cohort,row['instance_id']) not in retained:
                tasks.append((cohort,row['instance_id']))
    def save():
        record['cases'].sort(key=lambda r:(r['cohort'],r['instance_id']))
        temporary=args.out/'execution.tmp'
        temporary.write_text(json.dumps(record,indent=2)+'\n'); temporary.replace(args.out/'execution.json')
    def task(item):
        cohort, instance=item; path=args.out/f'{cohort}_{instance}.csv'; started=time.monotonic()
        cmd=[sys.executable,'-X','faulthandler','-B','-u',str(Path(__file__).with_name('acm_routing_hpc.py')),
             'case','--out',str(path.resolve()),'--cohort',cohort,'--instance',instance]
        with path.with_suffix('.log').open('w') as log:
            try: code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired: code='timeout'
        return dict(cohort=cohort,instance_id=instance,returncode=code,seconds=time.monotonic()-started,
                    output_sha256=sha(path) if path.exists() else None)
    save(); failed=False; remaining=iter(tasks)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active={pool.submit(task,item) for item in [next(remaining,None) for _ in range(args.workers)] if item is not None}
        while active:
            completed, active=wait(active,return_when=FIRST_COMPLETED)
            for future in completed:
                row=future.result(); record['cases'].append(row); failed |= row['returncode']!=0; save()
                print(f"{row['cohort']} {row['instance_id']}: {row['returncode']} ({len(record['cases'])}/68)",flush=True)
            if not failed:
                for _ in range(len(completed)):
                    item=next(remaining,None)
                    if item is not None: active.add(pool.submit(task,item))
    if failed: raise SystemExit('Case failure: remaining cases were not launched; logs preserved')
    assert len(record['cases'])==68


if __name__=='__main__':
    main()
