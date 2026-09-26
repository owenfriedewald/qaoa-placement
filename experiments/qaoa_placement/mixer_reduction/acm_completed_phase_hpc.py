"""Slurm-only paired full-circuit audit of exact completion on the fresh corpus.

Zero-extension references come from the homogeneous 2.5.2 routing campaign.
This job evaluates only L1 completion; zero-circuit hashes verify pairing.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
    os.environ[variable]='1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--instance')
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Requires Slurm; never run on workstation or login node')
    import run_acm_confirmation as c
    from invariant_placement import distance_coefficients
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    protocol=json.loads((c.DEFAULT/'protocol.json').read_text())
    assert protocol['code_sha256']==c.source_hashes()
    instances=[r for r in c.load_manifest(c.DEFAULT/'benchmark_manifest.json') if int(r['num_cells'])==4]
    if args.instance:
        instance=next(r for r in instances if r['instance_id']==args.instance)
        problem=c.replace(c.problem_from_manifest(instance),penalty=protocol['penalty_lambda'])
        keywords=dict(graph=protocol['token_graph'],schedule=protocol['token_schedule'])
        zero=c.placement_circuit(problem,'token',**keywords)
        zero_hash=c.circuit_hash(zero)
        circuit=c.placement_circuit(problem,'token',completion='l1',**keywords)
        logical=transpile(circuit,basis_gates=['rz','sx','x','cx'],optimization_level=3,seed_transpiler=123)
        backend=FakeSherbrooke(); rows=[]
        for seed in protocol['routing_seeds']:
            started=time.monotonic()
            routed=transpile(circuit,backend=backend,optimization_level=3,seed_transpiler=seed,layout_method='sabre',routing_method='sabre')
            rows.append(dict(instance_id=args.instance,family=instance['family'],method='token_l1',
                             graph=protocol['token_graph'],schedule=protocol['token_schedule'],
                             completion='l1',symbolic_parameters=circuit.num_parameters,active_qubits=circuit.num_qubits,
                             preparation=True,first_phase=False,row_penalties=False,measurements=False,
                             zero_circuit_sha256=zero_hash,circuit_sha256=c.circuit_hash(circuit),
                             logical_cx=logical.count_ops().get('cx',0),logical_depth=logical.depth(),routing_seed=seed,
                             routed_ecr=routed.count_ops().get('ecr',0),routed_depth=routed.depth(),
                             routed_two_qubit_depth=routed.depth(lambda op:len(op.qubits)==2),
                             routed_sha256=c.circuit_hash(routed),runtime_seconds=time.monotonic()-started))
        if args.out.exists(): raise FileExistsError(args.out)
        c.write_csv(args.out,rows)
        return
    workers=min(8,int(os.environ['SLURM_CPUS_PER_TASK']))
    args.out.mkdir(parents=True,exist_ok=False)
    import shutil
    for name in ('protocol.json','benchmark_manifest.json'):
        shutil.copy2(c.DEFAULT/name,args.out/name)
    c.environment(args.out,'routing',protocol)
    assert json.loads((args.out/'routing_environment.json').read_text())['versions']['qiskit']=='2.5.2'
    audit=dict(analysis='paired full-circuit L1 versus zero extension, all 36 fresh cases',
               specification='same symbolic prepared p=3 circuit, graph, schedule, and routing seeds',
               hypothesis='measure native ECR/depth change; no direction assumed',
               quality='same ideal valid-domain algorithm; no new optimization outcomes',
               comparison='zero references from acm-routing-corpus-20260904-v2; pair by circuit hash',
               source_sha256=c.sha(Path(__file__)),scientific_source_sha256=c.source_hashes(),
               job_id=os.environ['SLURM_JOB_ID'],workers=workers,threads_per_worker=1,case_timeout_seconds=1200)
    (args.out/'completion_protocol.json').write_text(json.dumps(audit,indent=2)+'\n')
    def task(instance):
        path=args.out/(instance['instance_id']+'.csv')
        with path.with_suffix('.log').open('w') as log:
            try:
                code=subprocess.run([sys.executable,'-X','faulthandler','-B','-u',str(Path(__file__).resolve()),
                    '--out',str(path.resolve()),'--instance',instance['instance_id']],stdout=log,stderr=subprocess.STDOUT,timeout=1200).returncode
            except subprocess.TimeoutExpired: code='timeout'
        return dict(instance_id=instance['instance_id'],returncode=code,sha256=c.sha(path) if path.exists() else None)
    completed=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # The bounded corpus is retained in full, including failed cases.
        for future in as_completed([pool.submit(task,i) for i in instances]):
            row=future.result(); completed.append(row)
            (args.out/'execution.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],cases=completed),indent=2)+'\n')
            print(f"{row['instance_id']}: {row['returncode']} ({len(completed)}/36)",flush=True)
    if any(r['returncode']!=0 for r in completed): raise SystemExit('Completion audit has failed cases; no complete claim')


if __name__=='__main__':
    main()
