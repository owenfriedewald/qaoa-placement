"""Frozen September 5 campaign; run only inside a bounded Slurm allocation."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata as md,itertools,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict,replace
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
for path in (HERE,HERE.parent):sys.path.insert(0,str(path))
from acm_gap_study import problem,dump,sha
from run_acm_confirmation import write_csv,read_csv
BASE=HERE/'acm_gap_closure_20260904'
SEEDS=[211,223,227,229,233]
POLICIES=['zero','l1','virtual','nearest','mean']


def counts(c):
    ops=c.count_ops()
    return dict(qubits=c.num_qubits,one_qubit=sum(len(op.qubits)==1 for op in c.data),rz=ops.get('rz',0),
        two_qubit=sum(len(op.qubits)==2 for op in c.data),depth=c.depth(),two_qubit_depth=c.depth(lambda op:len(op.qubits)==2))


def pyzx_flow(c):
    import pyzx as zx
    from qiskit import qasm2
    source=zx.Circuit.from_qasm(qasm2.dumps(c))
    graph=source.to_graph();zx.full_reduce(graph);opt=zx.extract_circuit(graph)
    assert source.verify_equality(opt),'ZX identity check failed'
    # Parser precision is separately tested against Qiskit in the preflight.
    return qasm2.loads(opt.to_qasm(),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def phase_case(row,cohort):
    import numpy as np
    from scipy.linalg import hadamard
    from qiskit import transpile
    from qiskit.circuit import Parameter
    from acm_submission_methods import extension,truncate,phase_from_coeff
    from invariant_placement import circuit_hash
    p=problem(row);k=(len(p.sites)-1).bit_length();W=hadamard(1<<(2*k))
    indices=[a|(b<<k) for b in range(len(p.sites)) for a in range(len(p.sites))]
    truth=np.array([sum(abs(p.sites[a][i]-p.sites[b][i]) for i in (0,1)) for b in range(len(p.sites)) for a in range(len(p.sites))])
    rows=[];vectors={};gamma=Parameter('gamma')
    for policy in POLICIES:
        start=time.monotonic();coeff=extension(p.sites,policy);seconds=time.monotonic()-start;vectors[policy]=coeff.tolist()
        residual=float(max(abs(W[indices]@coeff-truth)));assert residual<1e-8
        variants=[(policy,coeff,0.,residual)]
        if policy in ('l1','virtual'):
            for eta in (.001,.01,.05):
                truncated,error=truncate(p.sites,coeff,eta);vectors[f'{policy}_{eta}']=truncated.tolist()
                variants.append((f'{policy}_{eta}',truncated,eta,error))
        for name,c,eta,error in variants:
            for flow in (['beam','gray','gray_numeric','library','pyzx'] if eta==0 else ['gray']):
                start=time.monotonic()
                built=phase_from_coeff(p,gamma if flow in ('beam','gray') else .371,c,'gray' if flow in ('gray_numeric','pyzx') else flow)
                if flow=='pyzx':built=pyzx_flow(built)
                lowered=transpile(built,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
                rows.append(dict(instance_id=row['id'],family=row['family'],sites=len(p.sites),cohort=cohort,policy=name,
                    flow=flow,parameter_policy='symbolic' if flow in ('beam','gray') else 'numeric',gamma='' if flow in ('beam','gray') else .371,
                    valid_residual=error,cost_error_bound=error*sum(abs(w) for u,v,w in p.nets),eta=eta,
                    solve_seconds=seconds,synthesis_seconds=time.monotonic()-start,terms=int(np.count_nonzero(c[1:])),
                    weighted_norm=sum((.01+2*max(j.bit_count()-1,0))*abs(c[j]) for j in range(1,len(c))),
                    circuit_sha256=circuit_hash(built),**counts(lowered)))
    return rows,dict(instance=row,coefficients=vectors)


def route_case(row,mode):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from invariant_placement import circuit_hash
    from acm_submission_methods import token_circuit,row_circuit
    p=replace(problem(row),penalty=5.);rows=[];backend=FakeSherbrooke()
    variants=[('serial',3),('parallel',3),('complete',3),('ring',3)] if mode=='homogeneous' else [('serial',3)]+[(m,d) for m in ('complete','ring') for d in range(1,13)]
    for method,depth in variants:
        for component in (['full','prep','mixer','phase'] if mode=='homogeneous' else ['full']):
            c=token_circuit(p,depth,method=='parallel',component) if method in ('serial','parallel') else row_circuit(p,method,depth,component=component)
            logical=transpile(c,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
            for target in (['sherbrooke'] if mode=='homogeneous' and component=='full' else ['sherbrooke','line25','grid25']):
                kw=dict(backend=backend) if target=='sherbrooke' else dict(basis_gates=['rz','sx','x','cx'],coupling_map=CouplingMap.from_line(25,bidirectional=True) if target=='line25' else CouplingMap.from_grid(5,5,bidirectional=True))
                for seed in SEEDS:
                    start=time.monotonic();routed=transpile(c,**kw,optimization_level=3,seed_transpiler=seed,layout_method='sabre',routing_method='sabre')
                    rows.append(dict(instance_id=row['id'],family=row['family'],method=method,p=depth,component=component,
                        topology=target,seed=seed,circuit_sha256=circuit_hash(c),active_qubits=c.num_qubits,
                        parameters=c.num_parameters,seconds=time.monotonic()-start,**counts(routed),
                        **{'logical_'+key:value for key,value in counts(logical).items()}))
    return rows,None


def quality_case(row,method,depth):
    import numpy as np
    from scipy.optimize import minimize
    from acm_submission_methods import RandomToken
    from invariant_placement import ReducedPlacement
    from acm_reviewer_methods import RingPlacement
    from acm_gap_methods import initial_assignment,cached_cvar
    from run_token_permutation_p3_quality_uplift import initial_parameters
    from run_acm_confirmation import expected_best
    from placement_core import hpwl
    p=replace(problem(row),penalty=5.)
    model=RingPlacement(p) if method=='ring' else RandomToken(p,row['seed']) if method=='random_token' else ReducedPlacement(p,'token' if method=='token' else 'penalty_row_xy','line','empty_prioritized')
    order=np.argsort(model.energies);rows=[]
    for mode in ('deterministic','poor','random'):
        initial=initial_assignment(model,mode,row['seed']+77);initial_cost=hpwl(p,initial)
        for seed in (41,53):
            calls=0
            def loss(theta):
                nonlocal calls
                calls+=1;return cached_cvar(model.probabilities(theta,initial,depth),order,model.energies)
            start=time.monotonic();res=minimize(loss,initial_parameters(depth,seed)[1:],method='COBYLA',options=dict(maxiter=200,rhobeg=.2,tol=.001))
            probs=model.probabilities(res.x,initial,depth);legal=float(probs[model.legal].sum());optimal=float(probs[model.optimal].sum())
            rows.append(dict(instance_id=row['id'],family=row['family'],method=method,p=depth,initialization=mode,optimizer_seed=seed,
                initial_assignment=json.dumps(initial,sort_keys=True),theta=json.dumps(res.x.tolist()),optimal_probability=optimal,
                feasible_probability=legal,conditional_optimal=optimal/legal if legal else '',expected_best_128=expected_best(probs,model.costs,model.legal,initial_cost,128),
                expected_best_512=expected_best(probs,model.costs,model.legal,initial_cost,512),exact_cost=model.exact,initial_cost=initial_cost,
                evaluations=calls,seconds=time.monotonic()-start))
    return rows,None


def structured():
    import numpy as np
    from scipy.linalg import hadamard
    from acm_submission_methods import absolute_spectrum,grid_spectrum,extension
    rows=[]
    for width in range(1,17):
        start=time.monotonic();spectrum=absolute_spectrum(width);seconds=time.monotonic()-start
        if width<=6:
            for a,b in itertools.product(range(1<<width),repeat=2):
                z=a|(b<<width);value=sum(v*(-1)**((z&mask).bit_count()) for mask,v in spectrum.items())
                assert value==abs(a-b),(width,a,b,value)
        rows.append(dict(kind='growth',width=width,terms=len(spectrum),seconds=seconds))
    for m in (6,8,9,12,16):
        k=(m-1).bit_length()
        for kind,kx in [('row',k),('grid',(k+1)//2)]:
            sites=tuple((a% (1<<kx),a>>(kx)) for a in range(m));spectrum=grid_spectrum(kx,k-kx)
            coeff=np.zeros(1<<(2*k))
            for mask,value in spectrum.items():coeff[mask]=value
            for policy in ('zero','l1','structured'):
                c=coeff if policy=='structured' else extension(sites,policy)
                valid=[a|(b<<k) for b in range(m) for a in range(m)];truth=[sum(abs(sites[a][i]-sites[b][i]) for i in (0,1)) for b in range(m) for a in range(m)]
                error=max(abs((hadamard(len(c))@c)[valid]-truth));assert error<1e-8
                from run_token_geometry_padding_audit import fixed_six_net_problem
                from acm_submission_methods import phase_from_coeff
                from qiskit import transpile
                from qiskit.circuit import Parameter
                circuit=phase_from_coeff(fixed_six_net_problem(sites),Parameter('gamma'),c)
                lowered=transpile(circuit,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
                rows.append(dict(kind=kind,sites=m,width=k,policy=policy,gray_cx=lowered.count_ops().get('cx',0),gray_rz=lowered.count_ops().get('rz',0),terms=int(np.count_nonzero(c[1:])),naive_cx=sum(2*max(mask.bit_count()-1,0) for mask in range(1,len(c)) if c[mask]),valid_residual=error))
    return rows,None


def tasks_for(kind,out):
    config=json.loads((BASE/'protocol.json').read_text())
    if kind=='phase':
        from run_token_geometry_padding_audit import geometry,fixed_six_net_problem
        old=json.loads((HERE/'exact_phase_completion_20260904/protocol.json').read_text())
        return [dict(row=dict(id=f'historical_{m}_{f}_{r}',family=f,**asdict(fixed_six_net_problem(geometry(f,m,r)))),cohort='historical') for m,f,r in old['cases']]+[dict(row=r,cohort='fresh') for r in config['phase']]
    if kind in ('homogeneous','budget'):
        return [dict(row=r,mode=kind) for r in config['phase' if kind=='homogeneous' else 'quality'] if len(r['sites'])==6]
    if kind=='structured':return [{}]
    if kind=='quality':
        routes=read_csv(out.parents[1]/'budget/data/run_level.csv');selection=[]
        import numpy as np
        def median(method,p,target,field):
            vals={}
            for r in routes:
                if r['method']==method and int(r['p'])==p and r['topology']==target:vals.setdefault(r['instance_id'],[]).append(float(r[field]))
            assert len(vals)==12 and all(len(v)==5 for v in vals.values())
            return float(np.median([np.median(v) for v in vals.values()]))
        for method,target,field in itertools.product(('complete','ring'),('sherbrooke','line25','grid25'),('two_qubit','depth')):
            limit=median('serial',3,target,field);levels={p:median(method,p,target,field) for p in range(1,13)}
            feasible=[p for p,v in levels.items() if v<=limit];assert feasible
            p=max(feasible);selection.append(dict(method=method,topology=target,endpoint=field,p=p,token=limit,row=levels[p],cap_binds=p==12,levels=levels))
        dump(out/'selection.json',selection)
        settings={(m,p) for m in ('token','complete','ring') for p in (1,2,3,4,6)}|{(r['method'],r['p']) for r in selection}|{('random_token',3)}
        return [dict(row=r,method=m,depth=p) for r in config['quality'] if len(r['sites'])==6 for m,p in sorted(settings)]
    raise ValueError(kind)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',required=True);parser.add_argument('--out',required=True,type=Path);parser.add_argument('--task',type=int);args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID');assert md.version('qiskit')=='2.5.2'
    out=args.out.resolve()
    if args.task is not None:
        task=json.loads((out/'tasks.json').read_text())[args.task]
        function={'phase':phase_case,'homogeneous':route_case,'budget':route_case,'quality':quality_case,'structured':structured}[args.kind]
        rows,extra=function(**task);path=out/'cases'/f'{args.task:04d}.csv';assert not path.exists()
        # Some diagnostic rows have different fields.
        fields=list(dict.fromkeys(k for r in rows for k in r));write_csv(path,[{k:r.get(k,'') for k in fields} for r in rows])
        if extra:dump(path.with_suffix('.json'),extra)
        return
    out.mkdir(parents=True,exist_ok=False);(out/'cases').mkdir()
    tasks=tasks_for(args.kind,out);dump(out/'tasks.json',tasks)
    dump(out/'environment.json',dict(job=os.environ['SLURM_JOB_ID'],kind=args.kind,spec_sha256=sha(ROOT/'paper/amendments/submission_controls_20260905/spec.md'),payload_sha256=sha(ROOT/'payload.sha256'),versions={n:md.version(n) for n in ('qiskit','numpy','scipy','qiskit-ibm-runtime','pyzx')}))
    def work(i):
        with (out/'cases'/f'{i:04d}.log').open('w') as log:
            try:code=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--kind',args.kind,'--out',str(out),'--task',str(i)],stdout=log,stderr=subprocess.STDOUT,timeout=3600).returncode
            except subprocess.TimeoutExpired:code='timeout'
        path=out/'cases'/f'{i:04d}.csv';return dict(task=i,returncode=code,sha256=sha(path) if path.exists() else None)
    ledger=[]
    with ThreadPoolExecutor(max_workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))) as pool:
        for future in as_completed([pool.submit(work,i) for i in range(len(tasks))]):
            ledger.append(future.result());dump(out/'execution.json',ledger);print(args.kind,len(ledger),len(tasks),ledger[-1],flush=True)
    assert len(ledger)==len(tasks) and all(r['returncode']==0 for r in ledger),'Failed cases retained'
    rows=[r for p in sorted((out/'cases').glob('*.csv')) for r in read_csv(p)];write_csv(out/'run_level.csv',rows)

if __name__=='__main__':main()
