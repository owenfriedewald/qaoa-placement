"""Validate independently selected penalty controls using standard library."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import shutil
from collect_gap_study import ROOT,MIXER,OUT as GAP,read,sha
OUT=MIXER/'acm_penalty_calibration_20260904'


def validate(directory):
    p=json.loads((directory/'protocol.json').read_text());base=p['base_protocol']
    assert sha(GAP/'protocol.json')==p['base_protocol_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    assert not {r['seed'] for r in p['development']} & {r['seed'] for r in base['quality']}
    rows={}
    for stage,count in [('tuning',216),('evaluation',288)]:
        e=json.loads((directory/(stage+'_execution.json')).read_text())
        assert len(e['cases'])==e['expected'] and all(r['returncode']==0 for r in e['cases'])
        group=[]
        for case in sorted(e['cases'],key=lambda c:c['case']):
            path=directory/stage/(case['case']+'.csv');assert sha(path)==case['sha256'];group.extend(read(path))
        assert len(group)==count;rows[stage]=group
    expected={(r['id'],c,i,'61','analytic') for r in p['development'] for c in p['candidates'] for i in base['initializations']}
    actual=[(r['instance_id'],r['candidate'],r['initialization'],r['optimizer_seed'],r['training']) for r in rows['tuning']]
    assert len(actual)==len(expected) and set(actual)==expected
    selected=json.loads((directory/'selection.json').read_text())
    for n in (4,5,6):
        scores={c:sum(float(r['optimal_probability']) for r in rows['tuning'] if r['cells']==str(n) and r['candidate']==c)/sum(r['cells']==str(n) and r['candidate']==c for r in rows['tuning']) for c in p['candidates']}
        chosen=max(p['candidates'],key=lambda c:scores[c]);assert selected[str(n)]['candidate']==chosen
        assert all(abs(scores[c]-selected[str(n)]['scores'][c])<1e-12 for c in scores)
    expected={(r['id'],i,str(seed),training) for r in base['quality'] for i in base['initializations'] for seed in base['optimizer_seeds'] for training in base['training_modes'][str(len(r['cells']))]}
    actual=[(r['instance_id'],r['initialization'],r['optimizer_seed'],r['training']) for r in rows['evaluation']]
    assert len(actual)==len(expected) and set(actual)==expected
    references={(r['instance_id'],r['initialization'],r['optimizer_seed'],r['training']):r for r in read(GAP/'quality_run_level.csv') if r['method']=='penalty_row_xy'}
    for r in rows['evaluation']:
        reference=references[r['instance_id'],r['initialization'],r['optimizer_seed'],r['training']]
        assert r['initial_assignment']==reference['initial_assignment'] and r['exact_cost']==reference['exact_cost']
        assert r['candidate']==selected[r['cells']]['candidate']
        assert 0<int(r['evaluations'])<=base['budget'] and 0<=float(r['optimal_probability'])<=1+1e-10
        assert int(r['training_shots'])==(0 if r['training']=='analytic' else int(r['evaluations'])*base['shots'])
    return rows


def collect(source):
    rows=validate(source);OUT.mkdir(exist_ok=False)
    for f in source.rglob('*'):
        if f.is_file():
            dest=OUT/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for kind,group in rows.items():
        with (OUT/(kind+'_run_level.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(group[0]),lineterminator='\n');w.writeheader();w.writerows(group)
    print({k:len(v) for k,v in rows.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);collect(p.parse_args().input)
