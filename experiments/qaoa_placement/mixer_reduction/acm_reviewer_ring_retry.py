"""Narrow retry of failed historical-cohort adapter; frozen v1 is preserved."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
from acm_reviewer_study import CONF,ROOT,sha,dump,write_csv,read_csv
from acm_reviewer_methods import RingPlacement


def confirmation_ring(row):
    import run_acm_confirmation as old
    config=json.loads((CONF/'protocol.json').read_text())
    # Legacy postprocessing requires a dictionary entry named "token". Its
    # value is still RingPlacement; only the adapter's internal key changes.
    with patch.object(old,'METHODS',['token']),patch.object(old,'ReducedPlacement',lambda p,*a:RingPlacement(p)):
        rows,_=old.quality_case(row,config)
    for r in rows:
        mode=r.pop('init_mode');_,initial=old.init_assignment(row,mode)
        r.update(cells=r.pop('num_cells'),sites=r.pop('num_sites'),initialization=mode,
            training='analytic',training_shots=0,penalty_lambda=5.,initial_assignment=json.dumps(initial,sort_keys=True),
            method='ring_row_xy',cohort='confirmation',candidate='fixed5')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--task',type=int);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    if a.task is not None:
        task=json.loads((a.out/'evaluation/tasks.json').read_text())[a.task]
        assert task['cohort']=='confirmation'
        write_csv(a.out/'evaluation'/f'{a.task:04d}.csv',confirmation_ring(task['row']));return
    assert not a.out.exists();shutil.copytree(a.original,a.out)
    original=json.loads((a.original/'evaluation_execution.json').read_text())
    failed=[r for r in original if r['returncode']!=0]
    assert {r['task'] for r in failed}==set(range(36))
    for r in failed:
        log=a.out/'evaluation'/f"{r['task']:04d}.log"
        assert "KeyError: 'token'" in log.read_text()
        log.rename(log.with_suffix('.failed-v1.log'))
    env=json.loads((a.out/'environment.json').read_text())
    env['retry_job']=os.environ['SLURM_JOB_ID'];env['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    dump(a.out/'environment.json',env)
    dump(a.out/'retry_amendment.json',dict(reason='Historical adapter omitted dictionary key required by discarded legacy selector diagnostics',
        original_execution_sha256=sha(a.original/'evaluation_execution.json'),source_sha256=sha(Path(__file__)),
        rerun_tasks=list(range(36)),retained_successful_tasks=list(range(36,108)),
        unchanged='ring kernel, parameters, objective, cohorts, seeds, calibration and endpoints',job=os.environ['SLURM_JOB_ID']))
    def work(i):
        with (a.out/'evaluation'/f'{i:04d}.log').open('w') as log:
            code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--out',str(a.out),'--original',str(a.original),'--task',str(i)],stdout=log,stderr=subprocess.STDOUT,timeout=3600).returncode
        path=a.out/'evaluation'/f'{i:04d}.csv'
        return dict(task=i,returncode=code,sha256=sha(path) if path.exists() else None)
    results=[r for r in original if r['returncode']==0]
    with ThreadPoolExecutor(max_workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
        for f in as_completed([pool.submit(work,i) for i in range(36)]):
            results.append(f.result());dump(a.out/'evaluation_execution.json',results);print('retry',len(results),results[-1],flush=True)
    assert len(results)==108 and all(r['returncode']==0 for r in results)
    rows=[r for f in sorted((a.out/'evaluation').glob('*.csv')) for r in read_csv(f)]
    fields=list(dict.fromkeys(k for r in rows for k in r));write_csv(a.out/'evaluation_run_level.csv',[{k:r.get(k,'') for k in fields} for r in rows])


if __name__=='__main__':main()
