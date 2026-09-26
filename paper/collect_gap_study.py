"""Standard-library coverage/source validation and non-overwriting collection."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
MIXER=ROOT/'experiments/qaoa_placement/mixer_reduction'
OUT=MIXER/'acm_gap_closure_20260904'


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory):
    p=json.loads((directory/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    env=json.loads((directory/'environment.json').read_text())
    assert env['versions']['qiskit']=='2.5.2' and env['protocol_sha256']==sha(directory/'protocol.json')
    rows={}
    for kind in ('phase','route','quality','classical'):
        e=json.loads((directory/(kind+'_execution.json')).read_text())
        cases=e['cases'];assert len(cases)==e['expected'] and all(c['returncode']==0 for c in cases)
        assert len({c['case'] for c in cases})==len(cases)
        group=[]
        for case in sorted(cases,key=lambda c:c['case']):
            path=directory/kind/(case['case']+'.csv');assert sha(path)==case['sha256'],path
            group.extend(read(path))
        rows[kind]=group
    expected={(r['id'],policy,synthesis,mode,gamma) for r in p['phase'] for policy in p['policies']
              for synthesis in ('beam','gray','library') for mode in ('symbolic','numeric')
              for gamma in ([''] if mode=='symbolic' and synthesis!='library' else
                            [str(v) for v in p['numeric_angles']] if mode=='numeric' else [])}
    actual=[(r['instance_id'],r['policy'],r['synthesis'],r['parameter_policy'],r['gamma']) for r in rows['phase']]
    assert len(actual)==len(expected)==1320 and set(actual)==expected
    assert max(float(r['valid_distance_max_error']) for r in rows['phase'])<1e-8
    for r in rows['phase']:assert r['symbolic_parameters']==('1' if r['parameter_policy']=='symbolic' else '0')
    expected=set(__import__('itertools').product(p['route_cases'],p['route_methods'],p['route_topologies'],map(str,p['route_seeds'])))
    actual=[(r['instance_id'],r['method'],r['topology'],r['routing_seed']) for r in rows['route']]
    assert len(actual)==len(expected)==600 and set(actual)==expected
    for instance,method in __import__('itertools').product(p['route_cases'],p['route_methods']):
        group=[r for r in rows['route'] if r['instance_id']==instance and r['method']==method]
        assert len({r['circuit_sha256'] for r in group})==1
        assert all(r['symbolic_parameters']=='5' for r in group)
    expected={(r['id'],m,i,str(seed),training) for r in p['quality'] for m in p['quality_methods']
        for i in p['initializations'] for seed in p['optimizer_seeds'] for training in p['training_modes'][str(len(r['cells']))]}
    actual=[(r['instance_id'],r['method'],r['initialization'],r['optimizer_seed'],r['training']) for r in rows['quality']]
    assert len(actual)==len(expected)==576 and set(actual)==expected
    for r in rows['quality']:
        assert 0<int(r['evaluations'])<=p['budget'] and len(json.loads(r['theta']))==5
        assert int(r['training_shots'])==(0 if r['training']=='analytic' else p['shots']*int(r['evaluations']))
        for key in ('optimal_probability','feasible_probability','improvement_probability'):
            assert math.isfinite(float(r[key])) and -1e-10<=float(r[key])<=1+1e-10
        for budget in p['classical_budgets']:assert float(r['exact_cost'])-1e-8<=float(r[f'expected_best_{budget}'])<=float(r['initial_cost'])+1e-8
    expected={(r['id'],m,i,str(seed),str(budget)) for r in p['quality'] for m in p['classical_methods']
        for i in p['initializations'] for seed in p['classical_seeds'] for budget in p['classical_budgets']}
    actual=[(r['instance_id'],r['method'],r['initialization'],r['seed'],r['budget']) for r in rows['classical']]
    assert len(actual)==len(expected)==3888 and set(actual)==expected
    for r in rows['classical']:
        assert r['evaluations']==r['budget'] and float(r['exact_cost'])-1e-9<=float(r['best_cost'])<=float(r['initial_cost'])+1e-9
    selected=json.loads((directory/'classical_selection.json').read_text())
    assert len(selected['instances'])==6 and not set(selected['instances']) & {r['id'] for r in p['quality']}
    tuning=read(directory/'tuning_rows.csv')
    expected_tuning=set(__import__('itertools').product(selected['instances'],p['initializations'],map(str,p['annealing_tuning']['scales']),map(str,p['classical_seeds'])))
    actual_tuning=[(r['instance_id'],r['initialization'],r['temperature'],r['seed']) for r in tuning]
    assert len(actual_tuning)==len(expected_tuning)==216 and set(actual_tuning)==expected_tuning
    assert all(int(r['evaluations'])==p['annealing_tuning']['budget'] for r in tuning)
    scores={scale:sum(float(r['ratio']) for r in tuning if float(r['temperature'])==scale)/sum(float(r['temperature'])==scale for r in tuning) for scale in p['annealing_tuning']['scales']}
    assert selected['temperature']==min(scores,key=lambda x:(scores[x],x))

    for row in p['phase']:
        coeff=json.loads((directory/'phase'/(row['id']+'.json')).read_text())
        assert coeff['instance']==row
        if len(row['sites']) in (8,16):assert coeff['coefficients']['zero']==coeff['coefficients']['l1']
    return rows


def collect(source):
    rows=validate(source);OUT.mkdir(exist_ok=False)
    for f in source.rglob('*'):
        if f.is_file():
            dest=OUT/f.relative_to(source);dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(f,dest)
    for kind,values in rows.items():
        with (OUT/(kind+'_run_level.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    print(json.dumps({k:len(v) for k,v in rows.items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);collect(p.parse_args().input)
