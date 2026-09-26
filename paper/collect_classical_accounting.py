"""Admit corrected controls only after observing every solver query."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from collect_gap_study import ROOT,MIXER,OUT as GAP,read,sha
OUT=MIXER/'acm_classical_accounting_20260904'


def validate(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    assert sha(GAP/'protocol.json')==protocol['base_protocol_sha256']
    assert sha(GAP/'classical_selection.json')==protocol['selection_sha256']
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    execution=json.loads((directory/'execution.json').read_text())
    assert len(execution['cases'])==36 and all(r['returncode']==0 for r in execution['cases'])
    rows=[]
    for case in sorted(execution['cases'],key=lambda r:r['case']):
        path=directory/'cases'/(case['case']+'.csv');assert sha(path)==case['sha256'];rows.extend(read(path))
    key=lambda r:tuple(r[k] for k in ('instance_id','method','initialization','seed','budget'))
    legacy={key(r):r for r in read(GAP/'classical_run_level.csv')}
    assert len(rows)==len(legacy)==3888 and {key(r) for r in rows}==set(legacy)
    for r in rows:
        assert r['evaluations']==r['observed_oracle_calls']==r['budget']
        # Removing the redundant query must preserve this frozen search path.
        assert all(r[field]==legacy[key(r)][field] for field in legacy[key(r)]),key(r)
    return rows


def collect(source):
    rows=validate(source);OUT.mkdir(exist_ok=False)
    for f in source.rglob('*'):
        if f.is_file():
            dest=OUT/f.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    with (OUT/'run_level.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Verified 3888 corrected controls; every observed call count matches and all search outcomes are preserved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);collect(p.parse_args().input)
