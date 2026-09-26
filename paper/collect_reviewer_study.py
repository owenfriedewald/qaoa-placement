"""Strict stdlib amendment coverage/provenance verification, no simulation."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MIXER=ROOT/'experiments/qaoa_placement/mixer_reduction'
OUT=MIXER/'acm_reviewer_controls_20260904'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def validate(root=OUT):
    spec=ROOT/'paper/amendments/reviewer_controls_20260904'
    for line in (spec/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ');assert sha(spec/name)==digest,name
    for path,digest in json.loads((spec/'inputs.json').read_text()).items():
        # Manuscript/artifact files deliberately evolve; their frozen snapshot is retained.
        if not path.startswith('paper/'):assert sha(ROOT/path)==digest,path
    result={}
    confirmation=json.loads((MIXER/'acm_confirmation_20260904/benchmark_manifest.json').read_text())['instances']
    confirmation={r['instance_id']:r for r in confirmation if r['num_cells']==4}
    gap=json.loads((MIXER/'acm_gap_closure_20260904/protocol.json').read_text())
    quality={r['id']:r for r in gap['quality']};phases={r['id']:r for r in gap['phase']}
    development=json.loads((MIXER/'acm_penalty_calibration_20260904/protocol.json').read_text())['development']
    development={r['id']:r for r in development}
    historical=json.loads((MIXER/'exact_phase_completion_20260904/protocol.json').read_text())['cases']
    counts={'ring':{'tuning':216,'evaluation':1008},'workspace':{'routing':1020},'phase':{'phase':440}}
    for kind,stages in counts.items():
        folder=root/kind
        env=json.loads((folder/'environment.json').read_text())
        assert env['spec_sha256']==sha(spec/'spec.md')
        assert env['versions']['qiskit']=='2.5.2'
        for path,digest in env['source_sha256'].items():assert sha(ROOT/path)==digest,path
        for stage,count in stages.items():
            tasks=json.loads((folder/stage/'tasks.json').read_text())
            task_keys=[]
            for t in tasks:
                row=t['row'];i=row.get('id',row.get('instance_id'));cohort=t['cohort']
                if cohort=='confirmation':assert row==confirmation[i]
                elif kind=='ring':assert row==(development if stage=='tuning' else quality)[i]
                elif cohort=='gap':assert row==phases[i]
                else:
                    m,f,r=next((m,f,r) for m,f,r in historical if f'historical_{m}_{f}_{r}'==i)
                    coefficient_file=MIXER/'exact_phase_completion_20260904'/f'coefficients_{m}_{f}_{r}.json'
                    assert row['sites']==json.loads(coefficient_file.read_text())['sites']
                task_keys.append((cohort,i,t.get('candidate','')))
            assert len(task_keys)==len(set(task_keys)),'Duplicated instance task'
            if kind=='ring':
                if stage=='tuning':expected_tasks={('development',i,c) for i in development for c in ('fixed5','quarter_bound','strict_bound','four_bound')}
                else:
                    chosen=json.loads((root/'ring/selection.json').read_text())
                    expected_tasks={('confirmation',i,'fixed5') for i in confirmation}|{('gap_fixed',i,'fixed5') for i in quality}|{('gap_calibrated',i,chosen[str(len(r['cells']))]['candidate']) for i,r in quality.items()}
            elif kind=='workspace':expected_tasks={('confirmation',i,'') for i in confirmation}|{('gap',i,'') for i,r in phases.items() if len(r['sites'])==6}
            else:expected_tasks={('historical',f'historical_{m}_{f}_{r}','') for m,f,r in historical}|{('gap',i,'') for i in phases}
            assert set(task_keys)==expected_tasks
            execution=json.loads((folder/(stage+'_execution.json')).read_text())
            assert len(execution)==len(tasks) and {r['task'] for r in execution}==set(range(len(tasks)))
            all_rows=[]
            for record in execution:
                assert record['returncode']==0
                path=folder/stage/f"{record['task']:04d}.csv"
                assert sha(path)==record['sha256']
                rows=read(path);task=tasks[record['task']]
                identifier=task['row'].get('id',task['row'].get('instance_id'))
                assert {r['instance_id'] for r in rows}=={identifier}
                assert {r['cohort'] for r in rows}=={task['cohort']}
                if kind=='ring':
                    n=len(task['row']['cells'])
                    seeds=[61] if stage=='tuning' else [11,17,23,31] if task['cohort']=='confirmation' else [41,53]
                    training=['analytic','finite2048'] if stage!='tuning' and task['cohort']!='confirmation' and n==4 else ['analytic']
                    expected={(a,str(s),t) for a in ('deterministic','poor','random') for s in seeds for t in training}
                    keys=[(r['initialization'],r['optimizer_seed'],r['training']) for r in rows]
                    assert set(keys)==expected and len(keys)==len(expected)
                    for r in rows:
                        assert 0<=float(r['optimal_probability'])<=float(r['feasible_probability'])+1e-9<=1+1e-8
                        assert 0<int(r['evaluations'])<=200
                        assert int(r['training_shots'])==(int(r['evaluations'])*2048 if r['training']=='finite2048' else 0)
                        assert len(json.loads(r['theta']))==5
                elif kind=='workspace':
                    conf=task['cohort']=='confirmation'
                    expected={(m,t,str(s)) for m in (['serial','parallel','complete'] if conf else ['serial','parallel','complete','ring']) for t in (['sherbrooke'] if conf else ['line25','grid25']) for s in (range(101,106) if conf else [211,223,227,229,233])}
                    keys=[(r['method'],r['topology'],r['seed']) for r in rows]
                    assert set(keys)==expected and len(keys)==len(expected)
                    for r in rows:
                        assert int(r['parameters'])==5
                        assert int(r['active_qubits'])=={'serial':19,'parallel':21,'complete':24,'ring':24}[r['method']]
                        assert r['native_gate']==('ecr' if conf else 'cx')
                else:
                    expected={(p,s) for p in ('dense','separable') for s in (['mask'] if task['cohort']=='historical' else ['beam','gray'])}
                    keys=[(r['policy'],r['synthesis']) for r in rows]
                    assert set(keys)==expected and len(keys)==len(expected)
                    assert all(float(r['valid_residual'])<1e-8 for r in rows)
                    coeff=json.loads(path.with_suffix('.json').read_text())
                    assert all(abs(a+b-c)<1e-8 for a,b,c in zip(coeff['x'],coeff['y'],coeff['separable']))
                all_rows.extend(rows)
            aggregate=read(folder/(stage+'_run_level.csv'))
            assert len(aggregate)==len(all_rows)==count,(kind,stage,len(aggregate),count)
            fields=set().union(*(r.keys() for r in all_rows))
            canonical=lambda rows:sorted(json.dumps({k:r.get(k,'') for k in fields},sort_keys=True) for r in rows)
            assert canonical(aggregate)==canonical(all_rows)
            result[kind+'_'+stage]=aggregate
    tuning=result['ring_tuning'];selection=json.loads((root/'ring/selection.json').read_text())
    candidates=['fixed5','quarter_bound','strict_bound','four_bound']
    for n in (4,5,6):
        scores={c:sum(float(r['optimal_probability']) for r in tuning if int(r['cells'])==n and r['candidate']==c)/18 for c in candidates}
        assert max(candidates,key=lambda c:scores[c])==selection[str(n)]['candidate']
        for c,v in scores.items():assert abs(v-selection[str(n)]['scores'][c])<1e-12
    # Serial controls must reconstruct the earlier frozen circuit families.
    old=read(MIXER/'acm_routing_qiskit252_20260904/routing_fresh_run_level.csv')
    fresh=read(MIXER/'acm_gap_closure_20260904/route_run_level.csv')
    keyed_old={(r['instance_id'],r['method'],r['routing_seed']):r for r in old}
    keyed_fresh={(r['instance_id'],r['method'],r['topology'],r['routing_seed']):r for r in fresh}
    for r in result['workspace_routing']:
        if r['method'] not in ('serial','complete'):continue
        method='penalty_row_xy' if r['method']=='complete' else 'token'
        if r['cohort']=='confirmation': ref=keyed_old[r['instance_id'],method,r['seed']]
        else:
            if method=='token':method='token_gray_l1'
            ref=keyed_fresh[r['instance_id'],method,r['topology'],r['seed']]
        assert r['circuit_sha256']==ref['circuit_sha256']
        assert r['routed_two_qubit']==ref.get('routed_ecr',ref.get('routed_cx'))
        assert r['routed_depth']==ref['routed_depth']
    return result


if __name__=='__main__':print(json.dumps({k:len(v) for k,v in validate().items()},indent=2))
