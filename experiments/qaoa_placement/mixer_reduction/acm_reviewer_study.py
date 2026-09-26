"""Bounded Slurm runner for the committed reviewer-control amendment."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import importlib.metadata as md
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
for p in (HERE,HERE.parent): sys.path.insert(0,str(p))
from acm_gap_study import problem, sha, dump
from run_acm_confirmation import write_csv, read_csv
BASE = HERE/'acm_gap_closure_20260904'
CONF = HERE/'acm_confirmation_20260904'
CAL = HERE/'acm_penalty_calibration_20260904'


def ring_run(row, cohort, candidate, tuning=False):
    from acm_reviewer_methods import RingPlacement
    from acm_penalty_calibration import sufficient_penalty_scale
    from acm_gap_study import quality_case
    if cohort == 'confirmation':
        import run_acm_confirmation as old
        config = json.loads((CONF/'protocol.json').read_text())
        with patch.object(old,'METHODS',['penalty_row_xy']), patch.object(old,'ReducedPlacement',lambda p,*a:RingPlacement(p)):
            rows,_ = old.quality_case(row,config)
        for r in rows:
            r.update(cells=r.pop('num_cells'), sites=r.pop('num_sites'), initialization=r.pop('init_mode'),
                     training='analytic',training_shots=0,penalty_lambda=5.)
            _, initial = old.init_assignment(row,r['initialization'])
            r['initial_assignment'] = json.dumps(initial,sort_keys=True)
    else:
        config = json.loads((BASE/'protocol.json').read_text())
        if tuning: config.update(optimizer_seeds=[61],training_modes={str(n):['analytic'] for n in (4,5,6)})
        row = dict(row); bound = sufficient_penalty_scale(problem(row))
        row['penalty'] = dict(fixed5=5.,quarter_bound=.25*bound,strict_bound=1.000001*bound,four_bound=4*bound)[candidate]
        with patch('invariant_placement.ReducedPlacement',lambda p,*a:RingPlacement(p)):
            rows = quality_case(row,config,'penalty_row_xy')
        for r in rows: r.update(penalty_lambda=row['penalty'],sufficient_penalty_threshold=bound)
    for r in rows: r.update(method='ring_row_xy',cohort=cohort,candidate=candidate)
    return rows


def route_run(row,cohort):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from invariant_placement import placement_circuit,circuit_hash
    from acm_reviewer_methods import workspace_circuit,ring_circuit,dependency_stages
    from run_token_permutation_p3_quality_uplift import scheduled_token_edges
    from benchmark_suite import problem_from_manifest
    p = replace(problem_from_manifest(row),penalty=5.) if cohort=='confirmation' else problem(row)
    identifier = row.get('id',row.get('instance_id')); rows=[]
    synthesis,policy = ('beam','zero') if cohort=='confirmation' else ('gray','l1')
    targets = ['sherbrooke'] if cohort=='confirmation' else ['line25','grid25']
    seeds = list(range(101,106)) if cohort=='confirmation' else [211,223,227,229,233]
    methods = ['serial','parallel','complete']+([] if cohort=='confirmation' else ['ring'])
    backend = FakeSherbrooke() if cohort=='confirmation' else None
    for method in methods:
        c = (placement_circuit(p,'penalty_row_xy') if method=='complete' else ring_circuit(p) if method=='ring'
             else workspace_circuit(p,method=='parallel',synthesis,policy))
        logical = transpile(c,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
        extra={}
        if method in ('serial','parallel'):
            mixer=workspace_circuit(p,method=='parallel',mixer_only=True)
            lm=transpile(mixer,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
            extra=dict(mixer_cx=lm.count_ops().get('cx',0),mixer_depth=lm.depth(),
                dependency_stages=len(dependency_stages(scheduled_token_edges(p,'line','empty_prioritized',0))))
        for target,seed in itertools.product(targets,seeds):
            kw=dict(backend=backend) if backend else dict(basis_gates=['rz','sx','x','cx'],coupling_map=(CouplingMap.from_line(25,bidirectional=True) if target=='line25' else CouplingMap.from_grid(5,5,bidirectional=True)))
            t=time.monotonic()
            routed=transpile(c,**kw,optimization_level=3,seed_transpiler=seed,layout_method='sabre',routing_method='sabre')
            rows.append(dict(instance_id=identifier,family=row['family'],cohort=cohort,method=method,topology=target,
                seed=seed,active_qubits=c.num_qubits,parameters=c.num_parameters,circuit_sha256=circuit_hash(c),
                logical_cx=logical.count_ops().get('cx',0),logical_depth=logical.depth(),
                native_gate='ecr' if backend else 'cx',routed_two_qubit=routed.count_ops().get('ecr' if backend else 'cx',0),
                routed_depth=routed.depth(),routed_two_qubit_depth=routed.depth(lambda op:len(op.qubits)==2),
                routed_sha256=circuit_hash(routed),runtime_seconds=time.monotonic()-t,**extra))
    return rows


def phase_run(row,cohort):
    import numpy as np
    from scipy.linalg import hadamard
    from qiskit import transpile
    from qiskit.circuit import Parameter
    from acm_reviewer_methods import coordinate_coefficients
    from invariant_placement import distance_coefficients,circuit_hash
    import invariant_placement as inv
    import acm_gap_methods as gap
    import run_exact_phase_completion as hist
    p=problem(row); t=time.monotonic(); dense=distance_coefficients(p.sites,'l1');dense_time=time.monotonic()-t
    t=time.monotonic(); sep,parts=coordinate_coefficients(p.sites);sep_time=time.monotonic()-t
    k=(len(p.sites)-1).bit_length();dim=len(dense);w=hadamard(dim)
    valid=[a|(b<<k) for b in range(len(p.sites)) for a in range(len(p.sites))]
    truth=np.array([abs(p.sites[a][0]-p.sites[b][0])+abs(p.sites[a][1]-p.sites[b][1]) for b in range(len(p.sites)) for a in range(len(p.sites))])
    weights=np.array([.01+2*max(j.bit_count()-1,0) for j in range(dim)]);weights[0]=0
    objectives={name:float(weights@abs(c)) for name,c in [('dense',dense),('separable',sep)]}
    assert objectives['dense'] <= objectives['separable']+1e-7*max(1,objectives['separable'])
    if len(p.sites)==1<<k: assert np.max(abs(dense-sep))<1e-8
    rows=[]
    builders=[('mask',hist.repository_phase,hist)] if cohort=='historical' else [('beam',inv.token_phase,inv),('gray',gap.gray_phase,gap)]
    for name,c in [('dense',dense),('separable',sep)]:
        residual=float(np.max(abs((w@c)[valid]-truth)));assert residual<1e-8
        for synthesis,builder,module in builders:
            with patch.object(module,'distance_coefficients',lambda *a:c): circuit=builder(p,Parameter('gamma'),'l1')
            lowered=transpile(circuit,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
            rows.append(dict(instance_id=row['id'],family=row['family'],sites=len(p.sites),cohort=cohort,policy=name,
                synthesis=synthesis,logical_cx=lowered.count_ops().get('cx',0),logical_depth=lowered.depth(),
                raw_cx=circuit.count_ops().get('cx',0),terms=int(np.count_nonzero(c[1:])),objective=objectives[name],
                valid_residual=residual,solve_seconds=dense_time if name=='dense' else sep_time,
                coefficients=dim,equalities=len(valid),constraint_entries=dim*len(valid),circuit_sha256=circuit_hash(circuit)))
    return rows,dict(instance=row,dense=dense.tolist(),separable=sep.tolist(),x=parts[0].tolist(),y=parts[1].tolist(),
                     zero=distance_coefficients(p.sites,'zero').tolist())


def run_tasks(tasks,out,stage,kind):
    folder=out/stage;folder.mkdir()
    dump(folder/'tasks.json',tasks)
    def work(i):
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--out',str(out),'--kind',kind,'--stage',stage,'--task',str(i)]
        with (folder/f'{i:04d}.log').open('w') as log:
            try: code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3600).returncode
            except subprocess.TimeoutExpired: code='timeout'
        path=folder/f'{i:04d}.csv'
        return dict(task=i,returncode=code,sha256=sha(path) if path.exists() else None)
    results=[]
    with ThreadPoolExecutor(max_workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
        for future in as_completed([pool.submit(work,i) for i in range(len(tasks))]):
            results.append(future.result());dump(out/(stage+'_execution.json'),results)
            print(stage,len(results),'/',len(tasks),results[-1],flush=True)
    assert len(results)==len(tasks) and all(r['returncode']==0 for r in results),'Incomplete campaign'
    rows=[r for f in sorted(folder.glob('*.csv')) for r in read_csv(f)]
    fields=list(dict.fromkeys(k for r in rows for k in r))
    rows=[{k:r.get(k,'') for k in fields} for r in rows]
    write_csv(out/(stage+'_run_level.csv'),rows)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--kind',choices=['ring','workspace','phase'],required=True)
    parser.add_argument('--stage');parser.add_argument('--task',type=int);args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    assert md.version('qiskit')=='2.5.2'
    out=args.out.resolve()
    if args.task is not None:
        task=json.loads((out/args.stage/'tasks.json').read_text())[args.task];extra=None
        if args.kind=='ring': rows=ring_run(**task)
        elif args.kind=='workspace':rows=route_run(**task)
        else:rows,extra=phase_run(**task)
        path=out/args.stage/f'{args.task:04d}.csv';assert not path.exists()
        write_csv(path,rows)
        if extra:dump(path.with_suffix('.json'),extra)
        return
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],kind=args.kind,
        versions={n:md.version(n) for n in ('qiskit','numpy','scipy','qiskit-ibm-runtime')},
        spec_sha256=sha(ROOT/'paper/amendments/reviewer_controls_20260904/spec.md'),
        payload_sha256=sha(ROOT/'payload.sha256'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('acm_reviewer*.py')}))
    base=json.loads((BASE/'protocol.json').read_text());cal=json.loads((CAL/'protocol.json').read_text())
    confirmation=[r for r in json.loads((CONF/'benchmark_manifest.json').read_text())['instances'] if r['num_cells']==4]
    if args.kind=='ring':
        candidates=cal['candidates']
        rows=run_tasks([dict(row=r,cohort='development',candidate=c,tuning=True) for r in cal['development'] for c in candidates],out,'tuning','ring')
        selection={}
        for n in (4,5,6):
            scores={c:float(np_mean([float(r['optimal_probability']) for r in rows if int(r['cells'])==n and r['candidate']==c])) for c in candidates}
            selection[str(n)]=dict(candidate=max(candidates,key=lambda c:scores[c]),scores=scores)
        dump(out/'selection.json',selection)
        tasks=[dict(row=r,cohort='confirmation',candidate='fixed5') for r in confirmation]
        tasks += [dict(row=r,cohort='gap_fixed',candidate='fixed5') for r in base['quality']]
        tasks += [dict(row=r,cohort='gap_calibrated',candidate=selection[str(len(r['cells']))]['candidate']) for r in base['quality']]
        run_tasks(tasks,out,'evaluation','ring')
    elif args.kind=='workspace':
        tasks=[dict(row=r,cohort='confirmation') for r in confirmation]+[dict(row=r,cohort='gap') for r in base['phase'] if len(r['sites'])==6]
        run_tasks(tasks,out,'routing','workspace')
    else:
        from run_token_geometry_padding_audit import geometry,fixed_six_net_problem
        from dataclasses import asdict
        tasks=[dict(row=dict(id=f'historical_{m}_{f}_{r}',family=f,**asdict(fixed_six_net_problem(geometry(f,m,r)))),cohort='historical') for m,f,r in json.loads((HERE/'exact_phase_completion_20260904/protocol.json').read_text())['cases']]
        tasks += [dict(row=r,cohort='gap') for r in base['phase']]
        run_tasks(tasks,out,'phase','phase')


def np_mean(values):
    assert values
    return sum(values)/len(values)


if __name__=='__main__':main()
