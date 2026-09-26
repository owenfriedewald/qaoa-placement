"""Slurm-only corrected classical rerun, with observed objective-call counts."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):os.environ[v]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
for p in (HERE,HERE.parent):sys.path.insert(0,str(p))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--base',type=Path,required=True);parser.add_argument('--case');args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Slurm required')
    from acm_gap_study import dump,sha,problem
    from run_acm_confirmation import write_csv
    base=json.loads((args.base/'protocol.json').read_text());selection=json.loads((args.base/'classical_selection.json').read_text())
    if not args.case:
        args.out.mkdir(exist_ok=False,parents=True);(args.out/'cases').mkdir()
        dump(args.out/'protocol.json',dict(base_protocol_sha256=sha(args.base/'protocol.json'),selection_sha256=sha(args.base/'classical_selection.json'),
            amendment='Remove one uncounted repeated initial query in multistart greedy; preserve intended search trajectory and caps; rerun all classical controls with instrumented oracle calls',
            source_sha256={**base['source_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/'acm_classical_accounting.py',HERE/'test_acm_classical_accounting.py')}},job=os.environ['SLURM_JOB_ID']))
    protocol=json.loads((args.out/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    if args.case:
        from unittest.mock import patch
        import acm_gap_methods,run_acm_confirmation
        from acm_classical_accounting import counted_search
        from invariant_placement import ReducedPlacement
        from placement_core import hpwl
        row=next(r for r in base['quality'] if r['id']==args.case);p=problem(row);model=ReducedPlacement(p,'token','line','empty_prioritized');rows=[]
        for mode in base['initializations']:
            initial=acm_gap_methods.initial_assignment(model,mode,row['seed']+77);initial_cost=hpwl(p,initial)
            for method,budget,seed in itertools.product(base['classical_methods'],base['classical_budgets'],base['classical_seeds']):
                counter=[0]
                def counted(pr,a):counter[0]+=1;return hpwl(pr,a)
                with patch.object(acm_gap_methods,'hpwl',counted),patch.object(run_acm_confirmation,'hpwl',counted):
                    best,calls=counted_search(p,initial,budget,seed,method,selection['temperature'])
                assert calls==counter[0]==budget
                rows.append(dict(instance_id=row['id'],family=row['family'],cells=len(p.cells),sites=len(p.sites),method=method,
                    initialization=mode,seed=seed,budget=budget,evaluations=calls,observed_oracle_calls=counter[0],best_cost=best,
                    exact_cost=model.exact,initial_cost=initial_cost,optimal_hit=abs(best-model.exact)<1e-9,ratio=best/model.exact))
        write_csv(args.out/'cases'/(args.case+'.csv'),rows);return
    def task(row):
        path=args.out/'cases'/(row['id']+'.csv')
        with path.with_suffix('.log').open('w') as log:
            result=subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--out',str(args.out.resolve()),'--base',str(args.base.resolve()),'--case',row['id']],stdout=log,stderr=subprocess.STDOUT,timeout=600)
        return dict(case=row['id'],returncode=result.returncode,sha256=sha(path) if path.exists() else None)
    complete=[]
    with ThreadPoolExecutor(max_workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
        for future in as_completed([pool.submit(task,r) for r in base['quality']]):
            r=future.result();complete.append(r);dump(args.out/'execution.json',dict(job=os.environ['SLURM_JOB_ID'],cases=complete));print(r['case'],r['returncode'],len(complete),flush=True)
    if len(complete)!=36 or any(r['returncode']!=0 for r in complete):raise SystemExit('Incomplete corrected control')

if __name__=='__main__':main()
