"""Derived endpoints from saved states; no training or outcome-based tuning."""
import csv,itertools,json,os,statistics as st,sys
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.stats import wilcoxon,binomtest
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'paper';M=ROOT/'experiments/qaoa_placement/mixer_reduction'
for p in (M,M.parent):sys.path.insert(0,str(p))
from run_acm_confirmation import read_csv,write_csv,expected_best,init_assignment
from invariant_placement import ReducedPlacement,distance_coefficients
from acm_reviewer_methods import RingPlacement
from acm_gap_study import problem
from benchmark_suite import problem_from_manifest


def interval(values,families):
    rng=np.random.default_rng(1209505);draws=np.zeros(20000)
    for family in sorted(set(families)):
        v=np.array([x for x,f in zip(values,families) if f==family]);draws+=v[rng.integers(len(v),size=(20000,len(v)))].sum(axis=1)
    return np.percentile(draws/len(values),[2.5,97.5]).tolist()


def tests(values,lower_is_better=False):
    v=np.round(values,12);nonzero=v[v!=0]
    direction=-1 if lower_is_better else 1
    return dict(wilcoxon=float(wilcoxon(v,zero_method='wilcox',method='auto').pvalue) if len(nonzero) else 1.,sign=float(binomtest(sum(nonzero>0),len(nonzero),.5).pvalue) if len(nonzero) else 1.,wins=int(sum(direction*v>0)),ties=int(sum(v==0)),losses=int(sum(direction*v<0)))


def holm(records,key):
    ordered=sorted(records,key=lambda r:r[key]);previous=0
    for rank,r in enumerate(ordered):previous=max(previous,min(1,(len(records)-rank)*r[key]));r[key+'_holm']=previous;r['test_family_size']=len(records)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    confirmation=read_csv(M/'acm_confirmation_20260904/quality_run_level.csv');gap=read_csv(M/'acm_gap_closure_20260904/quality_run_level.csv')
    cal=read_csv(M/'acm_penalty_calibration_20260904/evaluation_run_level.csv');ring=read_csv(M/'acm_reviewer_controls_20260904/ring/evaluation_run_level.csv')
    conf={r['instance_id']:r for r in json.loads((M/'acm_confirmation_20260904/benchmark_manifest.json').read_text())['instances']}
    # Manifest keys in historical bundles can use id instead of instance_id.
    models={};reconstructed=[]
    def endpoint(r,method,cohort):
        opt=float(r['optimal_probability']);legal=float(r['feasible_probability']);out=dict(optimum=opt,legal=legal,conditional=opt/legal if legal else None)
        assert legal>0
        if not r.get('expected_best_128'):
            identifier=r['instance_id'];key=(identifier,method)
            if key not in models:
                p=replace(problem_from_manifest(conf[identifier]),penalty=5.)
                models[key]=RingPlacement(p) if method=='ring' else ReducedPlacement(p,'token' if method=='token' else 'penalty_row_xy','line','empty_prioritized')
            model=models[key];_,initial=init_assignment(conf[identifier],r.get('init_mode',r.get('initialization')))
            probs=model.probabilities(json.loads(r['theta']),initial)
            assert abs(probs[model.optimal].sum()-opt)<1e-9
            assert abs(probs[model.legal].sum()-legal)<1e-9
            initial_cost=float(r.get('initial_cost') or r['initial_hpwl'])
            for K in (128,512):out['best_'+str(K)]=expected_best(probs,model.costs,model.legal,initial_cost,K)
        else:
            for K in (128,512):out['best_'+str(K)]=float(r['expected_best_'+str(K)])
        return out
    instances=[];summaries=[]
    for cohort in ('confirmation','gap_fixed','gap_calibrated'):
        conditions=[(4,'analytic')] if cohort=='confirmation' else [(4,'analytic'),(4,'finite2048'),(5,'analytic'),(6,'analytic')]
        for n,training in conditions:
            selected=[r for r in ring if r['cohort']==cohort and int(r['cells'])==n and r['training']==training]
            source=confirmation if cohort=='confirmation' else gap
            complete_source=cal if cohort=='gap_calibrated' else source
            for identifier in sorted({r['instance_id'] for r in selected}):
                groups={'ring':[r for r in selected if r['instance_id']==identifier],
                    'token':[r for r in source if r['instance_id']==identifier and r['method']=='token' and r.get('training','analytic')==training],
                    'complete':[r for r in complete_source if r['instance_id']==identifier and r['method']=='penalty_row_xy' and r.get('training','analytic')==training]}
                record=dict(cohort=cohort,cells=n,training=training,instance_id=identifier,family=groups['ring'][0]['family'])
                for method,group in groups.items():
                    es=[endpoint(r,method,cohort) for r in group]
                    for metric in es[0]:record[method+'_'+metric]=st.mean(e[metric] for e in es)
                instances.append(record)
            paired=[r for r in instances if r['cohort']==cohort and r['cells']==n and r['training']==training]
            for metric in ('optimum','conditional','best_128','best_512','legal'):
                values=np.array([r['token_'+metric]-r['ring_'+metric] for r in paired]);lo,hi=interval(values,[r['family'] for r in paired])
                a,b,c=[st.mean(r[m+'_'+metric] for r in paired) for m in ('token','complete','ring')]
                # Closure uses the signed gap, but is defined only if token is better.
                better=(a<b) if metric.startswith('best') else (a>b)
                summaries.append(dict(cohort=cohort,cells=n,training=training,metric=metric,n=len(paired),token=a,complete=b,ring=c,
                    delta=a-c,median_delta=float(np.median(values)),ci_low=lo,ci_high=hi,closure=(c-b)/(a-b) if better else '',**tests(values,metric.startswith('best'))))
    for metric in ('optimum','conditional','best_128','best_512','legal'):
        group=[r for r in summaries if r['metric']==metric]
        for test in ('wilcoxon','sign'):holm(group,test)
    write_csv(PAPER/'tables/submission_existing_instances.csv',instances);write_csv(PAPER/'tables/submission_existing_summary.csv',summaries)
    envelope=[]
    for metric in ('optimum','conditional','best_128','best_512'):
        for n,training in [(4,'analytic'),(4,'finite2048'),(5,'analytic'),(6,'analytic')]:
            fixed=[r for r in instances if r['cohort']=='gap_fixed' and r['cells']==n and r['training']==training]
            calibrated={r['instance_id']:r for r in instances if r['cohort']=='gap_calibrated' and r['cells']==n and r['training']==training}
            values=[]
            for r in fixed:
                candidates=[g[m+'_'+metric] for g in (r,calibrated[r['instance_id']]) for m in ('complete','ring')]
                values.append(r['token_'+metric]-(min(candidates) if metric.startswith('best') else max(candidates)))
            lo,hi=interval(values,[r['family'] for r in fixed]);envelope.append(dict(cells=n,training=training,metric=metric,delta=st.mean(values),ci_low=lo,ci_high=hi,description='per-instance evaluation-oracle envelope; not selected for deployment'))
    write_csv(PAPER/'tables/submission_oracle_envelope.csv',envelope)
    # Save compact machine-readable results and a fully labeled table.
    lines=[r'\begin{tabular}{lllrrrrrr}',r'\toprule',r'Cohort & Size & Train & $L_C$ & $L_R$ & $Q_T$ & $Q_C$ & $Q_R$ & Closure\\',r'\midrule']
    for r in summaries:
        if r['metric']!='conditional':continue
        legal=next(x for x in summaries if x['cohort']==r['cohort'] and x['cells']==r['cells'] and x['training']==r['training'] and x['metric']=='legal')
        closure='--' if r['closure']=='' else f"{100*r['closure']:.1f}\\%"
        lines.append(f"{'Confirm.' if r['cohort']=='confirmation' else 'Fixed' if r['cohort']=='gap_fixed' else 'Calibrated'} & {r['cells']} & {'shots' if r['training']=='finite2048' else 'exact'} & {legal['complete']:.3f} & {legal['ring']:.3f} & {r['token']:.3g} & {r['complete']:.3g} & {r['ring']:.3g} & {closure}\\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}']);(PAPER/'tables/submission_conditional.tex').write_text('\n'.join(lines)+'\n')
    # Independent energy reconstruction from retained completion coefficients.
    from scipy.linalg import hadamard
    protocol=json.loads((M/'acm_gap_closure_20260904/protocol.json').read_text())
    manifest={r['id']:r for r in protocol['quality']}
    neutral=[]
    token_rows=[r for r in confirmation+gap if r['method']=='token']
    for identifier in sorted({r['instance_id'] for r in token_rows}):
        is_conf=identifier in conf
        p=replace(problem_from_manifest(conf[identifier]),penalty=5.) if is_conf else problem(manifest[identifier])
        model=ReducedPlacement(p,'token','line','empty_prioritized');original=model.energies.copy()
        k=(len(p.sites)-1).bit_length();coeff=distance_coefficients(p.sites,'l1');distances=hadamard(len(coeff))@coeff
        completed=np.zeros(len(model.states));labels={c:i for i,c in enumerate(p.cells)}
        for u,v,w in p.nets:
            indices=model.states[:,labels[u]] | (model.states[:,labels[v]]<<k)
            completed+=w*distances[indices]
        energy_error=float(max(abs(original-completed)))
        assert energy_error<1e-8
        for r in [r for r in token_rows if r['instance_id']==identifier]:
            if is_conf:_,initial=init_assignment(conf[identifier],r['init_mode'])
            else:initial=json.loads(r['initial_assignment'])
            model.energies=original;before=model.probabilities(json.loads(r['theta']),initial)
            model.energies=completed;after=model.probabilities(json.loads(r['theta']),initial)
            error=float(max(abs(before-after)));assert error<1e-8
            neutral.append(dict(instance_id=identifier,initialization=r.get('init_mode',r.get('initialization')),optimizer_seed=r['optimizer_seed'],training=r.get('training','analytic'),max_energy_error=energy_error,max_probability_error=error,optimal_mass_error=float(abs(before[model.optimal].sum()-after[model.optimal].sum()))))
    write_csv(PAPER/'tables/submission_neutrality.csv',neutral)
    (PAPER/'submission_existing_summary.json').write_text(json.dumps(dict(summary=summaries,oracle=envelope,neutrality=dict(rows=len(neutral),max_energy_error=max(r["max_energy_error"] for r in neutral),max_probability_error=max(r["max_probability_error"] for r in neutral))),indent=2)+'\n')

if __name__=='__main__':main()
