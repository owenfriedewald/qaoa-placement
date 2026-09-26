"""Slurm-only gap-closure studies with disjoint preflight and frozen evaluation."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):os.environ[var]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
for p in (HERE,HERE.parent):sys.path.insert(0,str(p))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def problem(row):
    from placement_core import PlacementProblem
    return PlacementProblem(tuple(row['cells']),tuple(map(tuple,row['sites'])),tuple(map(tuple,row['nets'])),row['penalty'])


def freeze(out):
    from acm_gap_methods import fresh_problem
    from run_acm_confirmation import source_hashes
    out.mkdir(exist_ok=False,parents=True)
    phase=[];quality=[]
    for size,family,repeat in itertools.product((6,8,9,12,16),('compact','line','l_shape','scatter'),range(3)):
        seed=910000+1000*size+100*('compact','line','l_shape','scatter').index(family)+repeat
        phase.append(dict(id=f'phase_{size}_{family}_{repeat}',family=family,seed=seed,**asdict(fresh_problem(4,size,family,seed))))
    for n,m in ((4,6),(5,8),(6,8)):
        for family,repeat in itertools.product(('compact','line','scatter'),range(4)):
            seed=940000+n*10000+100*('compact','line','scatter').index(family)+repeat
            quality.append(dict(id=f'quality_{n}_{m}_{family}_{repeat}',family=family,seed=seed,**asdict(fresh_problem(n,m,family,seed))))
    hashes={**source_hashes(),**{str(p.relative_to(ROOT)):sha(p) for p in (HERE/'acm_gap_methods.py',HERE/'acm_gap_study.py',HERE/'test_acm_gap_methods.py')}}
    protocol=dict(schema='acm-gap-v1',source_sha256=hashes,phase=phase,quality=quality,
        synthesis=['beam','gray'],policies=['zero','l1'],numeric_angles=[.17,.371,1.03],
        phase_cases=60,phase_primary='paired within-case L1/zero logical CX, separately by synthesis and site count',
        route_cases=[r['id'] for r in phase if len(r['sites'])==6],
        route_methods=['penalty_row_xy','token_beam_zero','token_beam_l1','token_gray_zero','token_gray_l1'],
        route_topologies=['line25','grid25'],route_seeds=[211,223,227,229,233],
        compiler=dict(qiskit='2.5.2',basis=['rz','sx','x','cx'],optimization_level=3,logical_seed=123),
        quality_methods=['token','penalty_row_xy'],p=3,alpha=.25,budget=200,optimizer='COBYLA',rhobeg=.2,tol=.001,
        initializations=['deterministic','poor','random'],optimizer_seeds=[41,53],shots=2048,
        training_modes={'4':['analytic','finite2048'],'5':['analytic'],'6':['analytic']},
        primary_quality='unconditional optimum probability: average init/restarts then pair instances; sizes/training separate',
        finite_shot_scope='shot noise in ideal objective queries; final exact diagnostic, no device noise',
        classical_methods=['uniform','annealing','multistart_greedy'],classical_budgets=[32,128,512],classical_seeds=[401,409,419,421],
        annealing_tuning=dict(scales=[.05,.2,1.0],budget=128,instances='first six archived frozen manifest entries',metric='mean best/exact ratio',tie_break='lower temperature'),
        bootstrap=dict(unit='instance',resample='within fixed family',draws=20000,seed=944004,interval='percentile 95%; descriptive conditional on generator'),
        resources=dict(max_model_states=300000,workers=8,case_timeout_seconds=3600),
        outcome_filters='none; failed cases block complete aggregate',preflight='seeds 77101, 77102, 77103; excluded',
        frozen_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    dump(out/'protocol.json',protocol)
    import importlib.metadata as md
    dump(out/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],versions={n:md.version(n) for n in ('qiskit','qiskit-ibm-runtime','numpy','scipy')},protocol_sha256=sha(out/'protocol.json')))
    assert md.version('qiskit')=='2.5.2'
    return protocol


def phase_case(row):
    import numpy as np
    from qiskit import transpile
    from qiskit.circuit import Parameter
    from scipy.linalg import hadamard
    from invariant_placement import token_phase,distance_coefficients,circuit_hash
    from acm_gap_methods import gray_phase,library_phase
    p=problem(row);gamma=Parameter('gamma');rows=[];coefficients={}
    for policy in ('zero','l1'):
        c=distance_coefficients(p.sites,policy);coefficients[policy]=c.tolist()
        width=(len(p.sites)-1).bit_length();truth=hadamard(len(c))@c
        residual=max(abs(truth[a|(b<<width)]-(abs(p.sites[a][0]-p.sites[b][0])+abs(p.sites[a][1]-p.sites[b][1]))) for a,b in itertools.product(range(len(p.sites)),repeat=2))
        assert residual<1e-8
        for synthesis,builder in [('beam',token_phase),('gray',gray_phase),('library',library_phase)]:
            angles=[None,.17,.371,1.03] if synthesis!='library' else [.17,.371,1.03]
            template=builder(p,gamma,policy) if synthesis!='library' else None
            for angle in angles:
                t=time.monotonic()
                circuit=(template if angle is None else template.assign_parameters({gamma:angle})) if template is not None else builder(p,angle,policy)
                lowered=transpile(circuit,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
                rows.append(dict(instance_id=row['id'],family=row['family'],sites=len(p.sites),policy=policy,synthesis=synthesis,
                    parameter_policy='symbolic' if angle is None else 'numeric',gamma='' if angle is None else angle,
                    logical_cx=lowered.count_ops().get('cx',0),logical_depth=lowered.depth(),
                    raw_cx=circuit.count_ops().get('cx',0),symbolic_parameters=circuit.num_parameters,
                    valid_distance_max_error=residual,circuit_sha256=circuit_hash(circuit),runtime_seconds=time.monotonic()-t))
    return rows,dict(instance=row,coefficients=coefficients)


def route_case(row,protocol):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    from acm_gap_methods import controlled_circuit
    from invariant_placement import circuit_hash
    p=problem(row);rows=[]
    for method in protocol['route_methods']:
        circuit=controlled_circuit(p,method);digest=circuit_hash(circuit)
        for topology in protocol['route_topologies']:
            coupling=CouplingMap.from_line(25,bidirectional=True) if topology=='line25' else CouplingMap.from_grid(5,5,bidirectional=True)
            coupling_hash=hashlib.sha256(json.dumps(sorted(map(list,coupling.get_edges()))).encode()).hexdigest()
            for seed in protocol['route_seeds']:
                t=time.monotonic();c=transpile(circuit,coupling_map=coupling,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=seed,layout_method='sabre',routing_method='sabre')
                rows.append(dict(instance_id=row['id'],family=row['family'],method=method,topology=topology,routing_seed=seed,
                    active_qubits=circuit.num_qubits,symbolic_parameters=circuit.num_parameters,circuit_sha256=digest,
                    coupling_sha256=coupling_hash,routed_sha256=circuit_hash(c),routed_cx=c.count_ops().get('cx',0),routed_depth=c.depth(),
                    routed_two_qubit_depth=c.depth(lambda op:len(op.qubits)==2),runtime_seconds=time.monotonic()-t))
    return rows


def quality_case(row,protocol,only_method=None,preflight=False):
    import numpy as np
    from scipy.optimize import minimize
    from invariant_placement import ReducedPlacement
    from acm_gap_methods import initial_assignment,cached_cvar,sampled_cvar
    from run_token_permutation_p3_quality_uplift import initial_parameters
    from run_acm_confirmation import expected_best
    from placement_core import hpwl
    p=problem(row);rows=[]
    for method in ([only_method] if only_method else protocol['quality_methods']):
        state_count=__import__('math').factorial(len(p.sites)) if method=='token' else len(p.sites)**len(p.cells)
        assert state_count<=protocol['resources']['max_model_states']
        model=ReducedPlacement(p,method,'line','empty_prioritized');order=np.argsort(model.energies)
        for mode in (['deterministic'] if preflight else protocol['initializations']):
            initial=initial_assignment(model,mode,row['seed']+77);initial_cost=hpwl(p,initial)
            for seed in ([41] if preflight else protocol['optimizer_seeds']):
                start=initial_parameters(3,seed)[1:]
                for training in (['analytic'] if preflight else protocol['training_modes'][str(len(p.cells))]):
                    rng=np.random.default_rng(row['seed']+seed*100+protocol['initializations'].index(mode));calls=0
                    def loss(theta):
                        nonlocal calls
                        calls+=1;probs=model.probabilities(theta,initial)
                        if training=='analytic':return cached_cvar(probs,order,model.energies)
                        return sampled_cvar(probs,model.energies,rng,protocol['shots'])
                    t=time.monotonic();result=minimize(loss,start,method='COBYLA',options=dict(maxiter=8 if preflight else protocol['budget'],rhobeg=.2,tol=.001))
                    probs=model.probabilities(result.x,initial)
                    r=dict(instance_id=row['id'],family=row['family'],cells=len(p.cells),sites=len(p.sites),method=method,
                        initialization=mode,optimizer_seed=seed,training=training,initial_assignment=json.dumps(initial,sort_keys=True),
                        initial_cost=initial_cost,exact_cost=model.exact,optimal_probability=float(probs[model.optimal].sum()),
                        feasible_probability=float(probs[model.legal].sum()),improvement_probability=float(probs[model.legal&(model.costs<initial_cost-1e-9)].sum()),
                        evaluations=calls,training_shots=calls*protocol['shots'] if training!='analytic' else 0,
                        theta=json.dumps(result.x.tolist()),reported_loss=float(result.fun),optimizer_success=bool(result.success),
                        runtime_seconds=time.monotonic()-t)
                    for budget in protocol['classical_budgets']:
                        r[f'expected_best_{budget}']=expected_best(probs,model.costs,model.legal,initial_cost,budget)
                    rows.append(r)
    return rows


def tune(out,protocol):
    from benchmark_suite import load_manifest,problem_from_manifest
    from run_acm_confirmation import FROZEN
    from invariant_placement import ReducedPlacement
    from acm_gap_methods import initial_assignment,classical_search
    from run_acm_confirmation import write_csv
    instances=load_manifest(FROZEN)[:6];rows=[]
    for instance in instances:
        p=problem_from_manifest(instance);model=ReducedPlacement(p,'token')
        for mode in protocol['initializations']:
            initial=initial_assignment(model,mode,77111)
            for scale,seed in itertools.product(protocol['annealing_tuning']['scales'],protocol['classical_seeds']):
                best,calls=classical_search(p,initial,128,seed,'annealing',scale)
                rows.append(dict(instance_id=instance['instance_id'],initialization=mode,temperature=scale,seed=seed,ratio=best/model.exact,evaluations=calls))
    scores={scale:sum(r['ratio'] for r in rows if r['temperature']==scale)/sum(r['temperature']==scale for r in rows) for scale in protocol['annealing_tuning']['scales']}
    chosen=min(scores,key=lambda x:(scores[x],x));write_csv(out/'tuning_rows.csv',rows)
    dump(out/'classical_selection.json',dict(temperature=chosen,scores=scores,development_manifest_sha256=sha(FROZEN),instances=[i['instance_id'] for i in instances]))


def classical_case(row,protocol,selection):
    from invariant_placement import ReducedPlacement
    from acm_gap_methods import initial_assignment,classical_search
    from placement_core import hpwl
    p=problem(row);model=ReducedPlacement(p,'token','line','empty_prioritized');rows=[]
    for mode in protocol['initializations']:
        initial=initial_assignment(model,mode,row['seed']+77)
        for method,budget,seed in itertools.product(protocol['classical_methods'],protocol['classical_budgets'],protocol['classical_seeds']):
            best,calls=classical_search(p,initial,budget,seed,method,selection['temperature'])
            rows.append(dict(instance_id=row['id'],family=row['family'],cells=len(p.cells),sites=len(p.sites),method=method,
                initialization=mode,seed=seed,budget=budget,evaluations=calls,best_cost=best,exact_cost=model.exact,
                initial_cost=hpwl(p,initial),optimal_hit=abs(best-model.exact)<1e-9,ratio=best/model.exact))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['preflight','freeze','phase','route','quality','classical'],required=True)
    parser.add_argument('--case');parser.add_argument('--method');args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Run on Slurm compute node')
    from run_acm_confirmation import write_csv
    if args.mode=='freeze':freeze(args.out);return
    if args.mode=='preflight':
        from acm_gap_methods import fresh_problem
        args.out.mkdir(exist_ok=False,parents=True)
        p=freeze(args.out/'specification')
        row=dict(id='preflight6',family='compact',seed=77101,**asdict(fresh_problem(6,8,'compact',77101)))
        write_csv(args.out/'timing.csv',quality_case(row,p,preflight=True))
        row=dict(id='preflight_phase',family='scatter',seed=77102,**asdict(fresh_problem(4,6,'scatter',77102)))
        values,coeff=phase_case(row);write_csv(args.out/'phase.csv',values)
        row=dict(id='preflight_route',family='compact',seed=77103,**asdict(fresh_problem(4,6,'compact',77103)))
        p['route_seeds']=[211];write_csv(args.out/'routing.csv',route_case(row,p))
        return
    protocol=json.loads((args.out/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    if args.case:
        row=next(r for r in protocol['quality' if args.mode in ('quality','classical') else 'phase'] if r['id']==args.case)
        extra=None
        if args.mode=='phase':rows,extra=phase_case(row)
        elif args.mode=='route':rows=route_case(row,protocol)
        elif args.mode=='quality':rows=quality_case(row,protocol,args.method)
        else:rows=classical_case(row,protocol,json.loads((args.out/'classical_selection.json').read_text()))
        name=args.case+('_'+args.method if args.method else '')
        path=args.out/args.mode/(name+'.csv')
        if path.exists():raise FileExistsError(path)
        write_csv(path,rows)
        if extra:dump(path.with_suffix('.json'),extra)
        return
    folder=args.out/args.mode;folder.mkdir(exist_ok=False)
    if args.mode=='classical':tune(args.out,protocol)
    cases=protocol['quality' if args.mode in ('quality','classical') else 'phase']
    if args.mode=='route':cases=[r for r in cases if r['id'] in protocol['route_cases']]
    tasks=[(r,m) for r in cases for m in (protocol['quality_methods'] if args.mode=='quality' else [None])]
    def work(pair):
        row,method=pair;name=row['id']+('_'+method if method else '');path=folder/(name+'.csv')
        command=[sys.executable,'-X','faulthandler','-B','-u',str(Path(__file__).resolve()),'--out',str(args.out.resolve()),'--mode',args.mode,'--case',row['id']]
        if method:command+=['--method',method]
        with (folder/(name+'.log')).open('w') as log:
            try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3600).returncode
            except subprocess.TimeoutExpired:code='timeout'
        return dict(case=name,returncode=code,sha256=sha(path) if path.exists() else None)
    complete=[]
    with ThreadPoolExecutor(max_workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
        for future in as_completed([pool.submit(work,t) for t in tasks]):
            r=future.result();complete.append(r);dump(args.out/(args.mode+'_execution.json'),dict(job=os.environ['SLURM_JOB_ID'],expected=len(tasks),cases=complete))
            print(f"{args.mode}: {r['case']} {r['returncode']} ({len(complete)}/{len(tasks)})",flush=True)
    if any(r['returncode']!=0 for r in complete):raise SystemExit('Failed cases retained; complete aggregate prohibited')

if __name__=='__main__':main()
