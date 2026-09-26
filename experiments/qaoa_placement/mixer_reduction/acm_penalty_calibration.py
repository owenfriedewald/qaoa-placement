"""Slurm-only independent development calibration of the Row-XY comparator."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):os.environ[v]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
for p in (HERE,HERE.parent):sys.path.insert(0,str(p))


def sufficient_penalty_scale(problem):
    """2 lambda > weighted-degree_max * diameter permits strict collision repair."""
    degree={c:0. for c in problem.cells}
    for u,v,w in problem.nets:
        assert w>=0;degree[u]+=w;degree[v]+=w
    diameter=max(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in itertools.product(problem.sites,repeat=2))
    return .5*diameter*max(degree.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--case');parser.add_argument('--candidate');parser.add_argument('--stage',choices=['tuning','evaluation'])
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Slurm required')
    from acm_gap_study import sha,dump,problem,quality_case
    from acm_gap_methods import fresh_problem
    from run_acm_confirmation import write_csv,read_csv
    base=json.loads((args.base/'protocol.json').read_text())
    if not args.case:
        args.out.mkdir(exist_ok=False,parents=True)
        development=[]
        for n,m in ((4,6),(5,8),(6,8)):
            for f,r in itertools.product(('compact','line','scatter'),range(2)):
                seed=1040000+n*10000+100*('compact','line','scatter').index(f)+r
                development.append(dict(id=f'calibration_{n}_{f}_{r}',family=f,seed=seed,**asdict(fresh_problem(n,m,f,seed))))
        protocol=dict(base_protocol=base,base_protocol_sha256=sha(args.base/'protocol.json'),development=development,
            candidates=['fixed5','quarter_bound','strict_bound','four_bound'],
            strengths='5, 0.25*B, 1.000001*B, 4*B; B=diameter*max_weighted_degree/2',
            select='per size: maximize mean development unconditional optimum probability; ties candidate list order',
            calibration_optimizer_seeds=[61],calibration_initializations=base['initializations'],
            evaluation='all 36 base instances, same starts and training modes, only selected Row-XY baseline; no token reoptimization',
            development_filters='none',source_sha256={**base['source_sha256'],str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
            frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),job=os.environ['SLURM_JOB_ID'])
        dump(args.out/'protocol.json',protocol)
    else:protocol=json.loads((args.out/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    if args.case:
        row=dict(next(r for r in (protocol['development'] if args.stage=='tuning' else base['quality']) if r['id']==args.case))
        candidate=args.candidate if args.stage=='tuning' else json.loads((args.out/'selection.json').read_text())[str(len(row['cells']))]['candidate']
        bound=sufficient_penalty_scale(problem(row));row['penalty']={'fixed5':5.,'quarter_bound':.25*bound,'strict_bound':1.000001*bound,'four_bound':4*bound}[candidate]
        config=dict(base)
        if args.stage=='tuning':config.update(optimizer_seeds=[61],training_modes={str(n):['analytic'] for n in (4,5,6)})
        rows=quality_case(row,config,'penalty_row_xy')
        for r in rows:r.update(candidate=candidate,penalty_lambda=row['penalty'],sufficient_penalty_threshold=bound)
        name=args.case+('_'+candidate if args.stage=='tuning' else '')
        write_csv(args.out/args.stage/(name+'.csv'),rows);return
    for stage in ('tuning','evaluation'):
        folder=args.out/stage;folder.mkdir()
        cases=protocol['development'] if stage=='tuning' else base['quality']
        tasks=[(r,c) for r in cases for c in (protocol['candidates'] if stage=='tuning' else [None])]
        def work(task):
            r,c=task;name=r['id']+('_'+c if c else '')
            command=[sys.executable,'-X','faulthandler','-B','-u',str(Path(__file__).resolve()),'--out',str(args.out.resolve()),'--base',str(args.base.resolve()),'--stage',stage,'--case',r['id']]
            if c:command+=['--candidate',c]
            with (folder/(name+'.log')).open('w') as log:
                try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
                except subprocess.TimeoutExpired:code='timeout'
            path=folder/(name+'.csv');return dict(case=name,returncode=code,sha256=sha(path) if path.exists() else None)
        complete=[]
        with ThreadPoolExecutor(max_workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
            for future in as_completed([pool.submit(work,t) for t in tasks]):
                result=future.result();complete.append(result);dump(args.out/(stage+'_execution.json'),dict(job=os.environ['SLURM_JOB_ID'],expected=len(tasks),cases=complete))
                print(stage,result['case'],result['returncode'],len(complete),'/',len(tasks),flush=True)
        if any(r['returncode']!=0 for r in complete):raise SystemExit('Incomplete calibration: no selection/evaluation claim')
        if stage=='tuning':
            rows=[r for f in folder.glob('*.csv') for r in read_csv(f)];selection={}
            for n in (4,5,6):
                scores={c:sum(float(r['optimal_probability']) for r in rows if r['cells']==str(n) and r['candidate']==c)/sum(r['cells']==str(n) and r['candidate']==c for r in rows) for c in protocol['candidates']}
                chosen=max(protocol['candidates'],key=lambda c:scores[c]);selection[str(n)]=dict(candidate=chosen,scores=scores)
            dump(args.out/'selection.json',selection)

if __name__=='__main__':main()
