"""Strict standard-library validation of the new campaign's immutable rows."""
import csv,hashlib,json,math,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/qaoa_placement/mixer_reduction/acm_submission_controls_20260905'
SPEC=ROOT/'paper/amendments/submission_controls_20260905/spec.md'
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(kinds=None):
    kinds=set(kinds or ('phase','budget','homogeneous','structured','quality'))
    assert kinds <= {'phase','budget','homogeneous','structured','quality'}
    assert 'quality' not in kinds or 'budget' in kinds
    mutable={'paper/main.tex','paper/main.pdf','paper/gap_study.tex','paper/reviewer_summary.json'}
    for name,expected in json.loads(SPEC.with_name('inputs.json').read_text()).items():
        path=ROOT/'paper/archive/pre_submission_controls_20260905'/Path(name).name if name in mutable else ROOT/name
        assert sha(path)==expected,('pre-amendment input changed',name)
    results={}
    for kind,expected in [('phase',4960),('budget',4500),('homogeneous',2400),('structured',46),('quality',None)]:
        if kind not in kinds:continue
        folder=OUT/kind;env=json.loads((folder/'environment.json').read_text())
        assert env['spec_sha256']==sha(SPEC)
        assert env['versions']['qiskit']=='2.5.2' and env['versions']['pyzx']=='0.10.3'
        tasks=json.loads((folder/'tasks.json').read_text());ledger=json.loads((folder/'execution.json').read_text())
        assert len(tasks)==len(ledger) and {r['task'] for r in ledger}==set(range(len(tasks)))
        cases=[]
        for r in sorted(ledger,key=lambda x:x['task']):
            path=folder/'cases'/f"{r['task']:04d}.csv"
            assert r['returncode']==0 and sha(path)==r['sha256'];cases+=read(path)
        rows=read(folder/'run_level.csv');assert rows==cases
        assert len(rows)==(expected if expected is not None else len(tasks)*6)
        if kind=='phase':
            variants=[(p,f) for p in ('zero','l1','virtual','nearest','mean') for f in ('beam','gray','gray_numeric','library','pyzx')]
            variants += [(f'{p}_{eta}','gray') for p in ('l1','virtual') for eta in (.001,.01,.05)]
            keys=[(r['instance_id'],r['policy'],r['flow']) for r in rows]
            assert len(keys)==len(set(keys)) and set(keys)=={(t['row']['id'],p,f) for t in tasks for p,f in variants}
            for r in rows:
                if float(r['eta'])==0:assert float(r['valid_residual'])<1e-8
                assert r['parameter_policy']==('symbolic' if r['flow'] in ('beam','gray') else 'numeric')
        if kind in ('budget','homogeneous'):
            ids={t['row']['id'] for t in tasks};assert len(ids)==12
            keys=[(r['instance_id'],r['method'],int(r['p']),r['component'],r['topology'],int(r['seed'])) for r in rows]
            if kind=='budget':
                variants=[('serial',3)]+[(m,p) for m in ('complete','ring') for p in range(1,13)]
                expected_keys={(i,m,p,'full',target,seed) for i in ids for m,p in variants for target in ('sherbrooke','line25','grid25') for seed in (211,223,227,229,233)}
            else:
                expected_keys={(i,m,3,c,target,seed) for i in ids for m in ('serial','parallel','complete','ring') for c in ('full','prep','mixer','phase') for target in (('sherbrooke',) if c=='full' else ('sherbrooke','line25','grid25')) for seed in (211,223,227,229,233)}
            assert len(keys)==len(set(keys)) and set(keys)==expected_keys
            for r in rows:
                assert int(r['parameters'])==({'prep':0,'mixer':1,'phase':1}.get(r['component'],2*int(r['p'])-1))
        if kind=='quality':
            keys=[(r['instance_id'],r['method'],r['p'],r['initialization'],r['optimizer_seed']) for r in rows]
            assert len(keys)==len(set(keys))
            for r in rows:
                assert len(json.loads(r['theta']))==2*int(r['p'])-1
                assert 0<int(r['evaluations'])<=200
                assert 0<=float(r['optimal_probability'])<=float(r['feasible_probability'])+1e-10<=1+2e-10
                assert float(r['expected_best_128'])>=float(r['exact_cost'])-1e-8
                assert float(r['expected_best_128'])<=float(r['initial_cost'])+1e-8
        results[kind]=rows
    if 'quality' not in kinds:return results
    selections=json.loads((OUT/'quality/selection.json').read_text())
    assert len(selections)==12
    settings={(m,p) for m in ('token','complete','ring') for p in (1,2,3,4,6)}|{(r['method'],r['p']) for r in selections}|{('random_token',3)}
    config=json.loads((OUT.parent/'acm_gap_closure_20260904/protocol.json').read_text())
    ids={r['id'] for r in config['quality'] if len(r['sites'])==6}
    expected_quality={(i,m,str(p),initial,str(seed)) for i in ids for m,p in settings for initial in ('deterministic','poor','random') for seed in (41,53)}
    assert {(r['instance_id'],r['method'],r['p'],r['initialization'],r['optimizer_seed']) for r in results['quality']}==expected_quality
    def metric(method,p,target,field):
        groups={}
        for r in results['budget']:
            if r['method']==method and int(r['p'])==p and r['topology']==target:groups.setdefault(r['instance_id'],[]).append(float(r[field]))
        assert len(groups)==12 and all(len(g)==5 for g in groups.values())
        return st.median(st.median(g) for g in groups.values())
    for r in selections:
        target=r['topology'];field=r['endpoint'];limit=metric('serial',3,target,field)
        levels={p:metric(r['method'],p,target,field) for p in range(1,13)}
        assert r['p']==max(p for p,v in levels.items() if v<=limit)
        assert r['token']==limit and r['row']==levels[r['p']]
        assert bool(r['cap_binds'])==(r['p']==12)
        assert {int(p):v for p,v in r['levels'].items()}==levels
    # The p=3 sweep repeats the archived fixed-penalty experiment. Treat an
    # unexpected change as a reproducibility failure, not a new quality gain.
    gap=read(OUT.parent/'acm_gap_closure_20260904/quality_run_level.csv')
    ring=read(OUT.parent/'acm_reviewer_controls_20260904/ring/evaluation_run_level.csv')
    original={}
    for r in gap:
        if r['training']=='analytic' and int(r['cells'])==4:
            method={'token':'token','penalty_row_xy':'complete'}[r['method']]
            original[r['instance_id'],method,r['initialization'],r['optimizer_seed']]=r
    for r in ring:
        if r['cohort']=='gap_fixed' and r['training']=='analytic' and int(r['cells'])==4:
            original[r['instance_id'],'ring',r['initialization'],r['optimizer_seed']]=r
    repeated=[r for r in results['quality'] if int(r['p'])==3 and r['method']!='random_token']
    assert len(repeated)==216
    for r in repeated:
        old=original[r['instance_id'],r['method'],r['initialization'],r['optimizer_seed']]
        assert json.loads(r['initial_assignment'])==json.loads(old['initial_assignment'])
        for field in ('optimal_probability','feasible_probability','expected_best_128','expected_best_512'):
            assert math.isclose(float(r[field]),float(old[field]),rel_tol=1e-8,abs_tol=1e-9),(r['instance_id'],r['method'],field,r[field],old[field])
    return results
if __name__=='__main__':print(json.dumps({k:len(v) for k,v in validate().items()},indent=2))
