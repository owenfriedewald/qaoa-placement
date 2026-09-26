"""Identical-input memoization equivalence gate; Slurm only."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from acm_submission_memo import memoized_pyzx
from acm_submission_study import read_csv,write_csv,dump,sha,problem,phase_case,pyzx_flow


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int);args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    run=Path(os.environ['QEDA_ACM_RUN_DIR']);prior=Path(os.environ['QEDA_ACM_PRIOR_PHASE'])
    ledger=json.loads((run/'phase-ledger.json').read_text());successful=[r for r in ledger if r['returncode']==0]
    assert len(successful)==40
    if args.case is not None:
        import numpy as np
        from acm_submission_cached_pivot import accelerate_pyzx as accelerate_verification
        from acm_submission_methods import phase_from_coeff
        from invariant_placement import circuit_hash
        record=next(r for r in successful if r['task']==args.case)
        path=prior/'cases'/f'{args.case:04d}.csv';assert sha(path)==record['sha256']
        obj=json.loads(path.with_suffix('.json').read_text());p=problem(obj['instance']);rows=read_csv(path);matches=[]
        with accelerate_verification(),memoized_pyzx() as memo:
            for policy in ('zero','l1','virtual','nearest','mean'):
                c=memo(phase_from_coeff(p,.371,np.array(obj['coefficients'][policy])))
                expected=next(r['circuit_sha256'] for r in rows if r['policy']==policy and r['flow']=='pyzx')
                assert circuit_hash(c)==expected,(args.case,policy)
                matches.append(dict(task=args.case,policy=policy,circuit_sha256=expected))
            assert memo.hits+memo.misses==5
            if len(p.sites)&(len(p.sites)-1)==0:assert memo.hits==4
        dump(run/'data'/f'matches-{args.case:04d}.json',matches);return
    (run/'data').mkdir(exist_ok=False)
    with (run/'logs/tests.txt').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','unittest','test_acm_submission_memo','test_acm_submission_cached_pivot','test_acm_submission_fast_verify','test_acm_submission_methods.SubmissionTests.test_zx_numeric_unitary'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    def case(i):
        with (run/'logs'/f'case-{i:04d}.txt').open('w') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--case',str(i)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        return i
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(case,r['task']) for r in successful]):print('matched case',future.result(),flush=True)
    matches=[r for path in sorted((run/'data').glob('matches-*.json')) for r in json.loads(path.read_text())]
    assert len(matches)==200;dump(run/'data/all_hash_matches.json',matches)
    print('200 PyZX circuit hashes match; identical-input memoization gate passed',flush=True)

if __name__=='__main__':main()
