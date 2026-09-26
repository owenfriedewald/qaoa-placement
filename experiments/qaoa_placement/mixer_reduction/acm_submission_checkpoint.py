"""Execution-only checkpointing; frozen circuit builders and settings unchanged.

Budget tasks contain one method/depth instead of an entire instance. Complete
prior cases are split into unchanged row groups and their original hashes are
recorded. Other campaign kinds resume complete case files byte for byte.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata as md,json,os,shutil,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import acm_submission_study as frozen
from acm_submission_study import ROOT,HERE,SEEDS,counts,problem,write_csv,read_csv,dump,sha


def budget_fragment(row,method,depth):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from invariant_placement import circuit_hash
    from acm_submission_methods import token_circuit,row_circuit
    assert (method=='serial' and depth==3) or (method in ('complete','ring') and 1<=depth<=12)
    p=replace(problem(row),penalty=5.);rows=[];backend=FakeSherbrooke()
    c=token_circuit(p,depth,False,'full') if method=='serial' else row_circuit(p,method,depth,component='full')
    logical=transpile(c,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
    for target in ('sherbrooke','line25','grid25'):
        kw=dict(backend=backend) if target=='sherbrooke' else dict(basis_gates=['rz','sx','x','cx'],coupling_map=CouplingMap.from_line(25,bidirectional=True) if target=='line25' else CouplingMap.from_grid(5,5,bidirectional=True))
        for seed in SEEDS:
            start=time.monotonic();routed=transpile(c,**kw,optimization_level=3,seed_transpiler=seed,layout_method='sabre',routing_method='sabre')
            rows.append(dict(instance_id=row['id'],family=row['family'],method=method,p=depth,component='full',
                topology=target,seed=seed,circuit_sha256=circuit_hash(c),active_qubits=c.num_qubits,
                parameters=c.num_parameters,seconds=time.monotonic()-start,**counts(routed),
                **{'logical_'+key:value for key,value in counts(logical).items()}))
    return rows,None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',required=True,choices=['budget','phase'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--prior',type=Path);parser.add_argument('--task',type=int);args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID');assert md.version('qiskit')=='2.5.2';out=args.out.resolve()
    if args.kind=='phase':
        import invariant_placement
        from acm_submission_fast_beam import order_parity_terms
        invariant_placement.order_parity_terms=order_parity_terms
    if args.task is not None:
        task=json.loads((out/'tasks.json').read_text())[args.task]
        rows,extra=(budget_fragment if args.kind=='budget' else frozen.phase_case)(**task)
        path=out/'cases'/f'{args.task:04d}.csv';assert not path.exists();write_csv(path,rows)
        if extra:dump(path.with_suffix('.json'),extra)
        return
    assert args.prior is not None;prior=args.prior.resolve();out.mkdir(parents=True,exist_ok=False);(out/'cases').mkdir()
    old_tasks=json.loads((prior/'tasks.json').read_text());old_env=json.loads((prior/'environment.json').read_text())
    old_ledger=json.loads((prior/'execution.json').read_text()) if (prior/'execution.json').exists() else []
    successful={r['task']:r for r in old_ledger if r['returncode']==0}
    if args.kind=='budget':
        # Support both original whole-instance and already-fragmented prior runs.
        if 'depth' in old_tasks[0]:tasks=old_tasks;origins={i:i for i in range(len(tasks))}
        else:
            variants=[('serial',3)]+[(m,p) for m in ('complete','ring') for p in range(1,13)]
            tasks=[dict(row=r['row'],method=m,depth=p) for r in old_tasks for m,p in variants]
            origins={i:i//25 for i in range(len(tasks))}
    else:tasks=old_tasks;origins={i:i for i in range(len(tasks))}
    dump(out/'tasks.json',tasks);ledger=[]
    for i,task in enumerate(tasks):
        origin=origins[i]
        if origin not in successful:continue
        source=prior/'cases'/f'{origin:04d}.csv';assert sha(source)==successful[origin]['sha256']
        target=out/'cases'/f'{i:04d}.csv'
        if args.kind=='budget' and 'depth' not in old_tasks[0]:
            rows=[r for r in read_csv(source) if r['method']==task['method'] and int(r['p'])==task['depth']];assert len(rows)==15;write_csv(target,rows)
        else:
            shutil.copy2(source,target)
            if source.with_suffix('.json').exists():shutil.copy2(source.with_suffix('.json'),target.with_suffix('.json'))
        ledger.append(dict(task=i,returncode=0,sha256=sha(target),reused_from_job=old_env['job'],source_case=origin,source_sha256=sha(source)))
    dump(out/'execution.json',ledger)
    dump(out/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],kind=args.kind,
        spec_sha256=sha(ROOT/'paper/amendments/submission_controls_20260905/spec.md'),payload_sha256=sha(ROOT/'payload.sha256'),
        versions={n:md.version(n) for n in ('qiskit','numpy','scipy','qiskit-ibm-runtime','pyzx')},
        prior_job=old_env['job'],prior_environment_sha256=sha(prior/'environment.json'),frozen_runner_sha256=sha(Path(frozen.__file__)),checkpoint_runner_sha256=sha(Path(__file__))))
    done={r['task'] for r in ledger};remaining=[i for i in range(len(tasks)) if i not in done]
    def work(i):
        with (out/'cases'/f'{i:04d}.log').open('w') as log:
            try:code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--kind',args.kind,'--out',str(out),'--task',str(i)],stdout=log,stderr=subprocess.STDOUT,timeout=3600).returncode
            except subprocess.TimeoutExpired:code='timeout'
        p=out/'cases'/f'{i:04d}.csv';return dict(task=i,returncode=code,sha256=sha(p) if p.exists() else None)
    with ThreadPoolExecutor(max_workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
        for future in as_completed([pool.submit(work,i) for i in remaining]):
            ledger.append(future.result());dump(out/'execution.json',ledger);print(args.kind,len(ledger),len(tasks),ledger[-1],flush=True)
    assert len(ledger)==len(tasks) and all(r['returncode']==0 for r in ledger),'Partial attempts retained'
    write_csv(out/'run_level.csv',[r for path in sorted((out/'cases').glob('*.csv')) for r in read_csv(path)])

if __name__=='__main__':main()
