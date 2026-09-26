"""Validate and collect the complete paired L1 routing audit (stdlib only)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MIXER = ROOT/'experiments/qaoa_placement/mixer_reduction'
OUT = MIXER/'acm_integrated_phase_completion_20260904'


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def validate(rows, directory):
    reference = read(MIXER/'acm_routing_qiskit252_20260904/routing_fresh_run_level.csv')
    zero = {(r['instance_id'], r['routing_seed']): r for r in reference if r['method']=='token'}
    keys = [(r['instance_id'], r['routing_seed']) for r in rows]
    assert len(keys)==720 and len(set(keys))==720 and set(keys)==set(zero)
    assert json.loads((directory/'routing_environment.json').read_text())['versions']['qiskit']=='2.5.2'
    protocol = json.loads((directory/'completion_protocol.json').read_text())
    assert hashlib.sha256((MIXER/'acm_completed_phase_hpc.py').read_bytes()).hexdigest()==protocol['source_sha256']
    for name, expected in protocol['scientific_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    for r in rows:
        z = zero[r['instance_id'],r['routing_seed']]
        assert r['zero_circuit_sha256']==z['circuit_sha256']
        assert r['completion']=='l1' and r['method']=='token_l1'
        for field in ('graph','schedule','symbolic_parameters','active_qubits','preparation','first_phase','row_penalties','measurements'):
            assert r[field]==z[field], field
    for instance in {r['instance_id'] for r in rows}:
        assert len({r['circuit_sha256'] for r in rows if r['instance_id']==instance})==1


def collect(source):
    execution = json.loads((source/'execution.json').read_text())
    assert len(execution['cases'])==36 and all(c['returncode']==0 for c in execution['cases'])
    rows=[]
    for case in execution['cases']:
        path=source/(case['instance_id']+'.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
        group=read(path)
        assert len(group)==20 and all(r['instance_id']==case['instance_id'] for r in group)
        rows.extend(group)
    validate(rows,source)
    OUT.mkdir(exist_ok=False)
    for name in ('execution.json','completion_protocol.json','routing_environment.json','protocol.json','benchmark_manifest.json'):
        shutil.copy2(source/name,OUT/name)
    rows.sort(key=lambda r:(r['instance_id'],int(r['routing_seed'])))
    with (OUT/'run_level.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print('Collected 720 paired full-circuit completion rows')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    collect(parser.parse_args().input)
