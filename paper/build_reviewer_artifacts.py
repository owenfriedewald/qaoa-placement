"""Derive amendment endpoints without outcome-dependent cohort selection."""
import json
import os
from pathlib import Path
import statistics as st
import numpy as np
from collect_reviewer_study import validate,OUT,MIXER,read
from build_gap_artifacts import write,table
PAPER=Path(__file__).resolve().parent


def interval(records):
    rng=np.random.default_rng(1144004);draws=np.zeros(20000)
    for f in sorted({r['family'] for r in records}):
        v=np.array([r['delta'] for r in records if r['family']==f])
        draws += v[rng.integers(len(v),size=(20000,len(v)))].sum(axis=1)
    return [float(x) for x in np.percentile(draws/len(records),[2.5,97.5])]


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    rows=validate();data={};instances=[];summaries=[]
    confirmation=read(MIXER/'acm_confirmation_20260904/quality_run_level.csv')
    gap=read(MIXER/'acm_gap_closure_20260904/quality_run_level.csv')
    calibrated=read(MIXER/'acm_penalty_calibration_20260904/evaluation_run_level.csv')
    ring=rows['ring_evaluation']
    for cohort in ('confirmation','gap_fixed','gap_calibrated'):
        for cells,training in ([(4,'analytic')] if cohort=='confirmation' else [(4,'analytic'),(4,'finite2048'),(5,'analytic'),(6,'analytic')]):
            selected=[r for r in ring if r['cohort']==cohort and int(r['cells'])==cells and r['training']==training]
            records=[]
            for i in sorted({r['instance_id'] for r in selected}):
                group=[r for r in selected if r['instance_id']==i]
                source=confirmation if cohort=='confirmation' else gap
                token=[r for r in source if r['instance_id']==i and r['method']=='token' and r.get('training','analytic')==training]
                source=calibrated if cohort=='gap_calibrated' else source
                complete=[r for r in source if r['instance_id']==i and r['method']=='penalty_row_xy' and r.get('training','analytic')==training]
                key=lambda r:(r.get('initialization',r.get('init_mode')),r['optimizer_seed'])
                assert {key(r) for r in group}=={key(r) for r in token}=={key(r) for r in complete}
                for r in group:
                    ref=next(t for t in token if key(t)==key(r))
                    assert abs(float(r.get('initial_cost') or r['initial_hpwl'])-float(ref.get('initial_cost',ref.get('initial_hpwl'))))<1e-9
                a=st.mean(float(r['optimal_probability']) for r in token)
                b=st.mean(float(r['optimal_probability']) for r in complete)
                c=st.mean(float(r['optimal_probability']) for r in group)
                record=dict(cohort=cohort,cells=cells,training=training,instance_id=i,family=group[0]['family'],
                    token=a,complete=b,ring=c,ring_legal=st.mean(float(r['feasible_probability']) for r in group),delta=a-c)
                records.append(record);instances.append(record)
            a=st.mean(r['token'] for r in records);b=st.mean(r['complete'] for r in records);c=st.mean(r['ring'] for r in records)
            lo,hi=interval(records);closure=(c-b)/(a-b) if a>b else None
            summaries.append(dict(cohort=cohort,cells=cells,training=training,instances=len(records),token=a,complete=b,ring=c,
                ring_legal=st.mean(r['ring_legal'] for r in records),delta=a-c,ci_low=lo,ci_high=hi,
                token_wins=sum(r['delta']>1e-12 for r in records),closure=closure,
                ring_competitive=c>=a or (closure is not None and closure>=.75),ring_wins=c>a and hi<0))
    data['quality']=summaries;write('reviewer_ring_instances.csv',instances);write('reviewer_ring_summary.csv',summaries)
    primary=next(r for r in summaries if r['cohort']=='gap_calibrated' and r['cells']==4 and r['training']=='analytic')
    data['editorial_gate']=dict(primary=primary,rule='75% gap closure or ring mean at least token; editorial criterion, not equivalence test',
                                completion_primary=primary['ring_competitive'])
    excluded=[r for r in instances if r['cohort']=='confirmation' and r['instance_id']!='fresh4_sparse_grid_904000']
    data['confirmation_without_smoke']=dict(instances=len(excluded),delta=st.mean(r['delta'] for r in excluded),interval=interval(excluded))
    table('reviewer_ring.tex','lllrrrrl',r'Cohort & Size & Train & Token & Complete & Ring & Ring legal & Token$-$ring interval',[
        f"{'Confirm.' if r['cohort']=='confirmation' else 'Fixed' if r['cohort']=='gap_fixed' else 'Calibrated'} & {r['cells']}c/{6 if r['cells']==4 else 8}s & {'shots' if r['training']=='finite2048' else 'exact'} & {r['token']:.3g} & {r['complete']:.3g} & {r['ring']:.3g} & {r['ring_legal']:.3f} & [{r['ci_low']:.3g}, {r['ci_high']:.3g}]\\\\" for r in summaries])
    routed=rows['workspace_routing'];resource=[];route_instances=[]
    for cohort,topology in [('confirmation','sherbrooke'),('gap','line25'),('gap','grid25')]:
        selected=[r for r in routed if r['cohort']==cohort and r['topology']==topology]
        ids=sorted({r['instance_id'] for r in selected});methods=sorted({r['method'] for r in selected})
        medians={}
        for i in ids:
            for method in methods:
                g=[r for r in selected if r['instance_id']==i and r['method']==method]
                record=dict(cohort=cohort,topology=topology,instance_id=i,method=method,active_qubits=int(g[0]['active_qubits']))
                for metric in ('routed_two_qubit','routed_depth','routed_two_qubit_depth','logical_cx','logical_depth'):
                    record[metric]=st.median(float(r[metric]) for r in g)
                medians[i,method]=record;route_instances.append(record)
        for method,reference in [('parallel','serial'),('parallel','complete'),('serial','complete')]+([('parallel','ring'),('serial','ring')] if cohort=='gap' else []):
            record=dict(cohort=cohort,topology=topology,method=method,reference=reference,instances=len(ids))
            for metric in ('routed_two_qubit','routed_depth','routed_two_qubit_depth','logical_cx','logical_depth'):
                pairs=[(medians[i,method][metric],medians[i,reference][metric]) for i in ids]
                record[metric+'_relative']=st.median(a/b-1 for a,b in pairs)
                record[metric+'_wins']=sum(a<b for a,b in pairs)
            resource.append(record)
    data['workspace']=resource;write('reviewer_workspace_summary.csv',resource);write('reviewer_workspace_instances.csv',route_instances)
    table('reviewer_workspace.tex','lllrr',r'Target & Token & Reference & 2q change & Depth change',[
        f"{r['topology']} & {r['method']} & {r['reference']} & {100*r['routed_two_qubit_relative']:+.1f}\\% & {100*r['routed_depth_relative']:+.1f}\\%\\\\" for r in resource])
    phase=rows['phase_phase'];phase_summaries=[]
    for corpus,synthesis in [('historical','mask'),('gap','beam'),('gap','gray')]:
        for m in (6,8,9,12,16):
            g=[r for r in phase if r['cohort']==corpus and r['synthesis']==synthesis and int(r['sites'])==m]
            ids=sorted({r['instance_id'] for r in g});keyed={(r['instance_id'],r['policy']):r for r in g}
            pairs=[(int(keyed[i,'separable']['logical_cx']),int(keyed[i,'dense']['logical_cx'])) for i in ids]
            phase_summaries.append(dict(corpus=corpus,synthesis=synthesis,sites=m,instances=len(ids),
                relative=st.median(a/b-1 for a,b in pairs),wins=sum(a<b for a,b in pairs),ties=sum(a==b for a,b in pairs),losses=sum(a>b for a,b in pairs),
                max_residual=max(float(r['valid_residual']) for r in g),
                objective_relative=st.median(float(keyed[i,'separable']['objective'])/float(keyed[i,'dense']['objective'])-1 for i in ids),
                dense_seconds=st.median(float(keyed[i,'dense']['solve_seconds']) for i in ids),
                separable_seconds=st.median(float(keyed[i,'separable']['solve_seconds']) for i in ids)))
    data['coordinate']=phase_summaries;write('reviewer_coordinate_summary.csv',phase_summaries)
    table('reviewer_coordinate.tex','llrrrr',r'Corpus & Flow & Sites & CX change & W/T/L & Norm change',[
        f"{'100-case' if r['corpus']=='historical' else '60-case'} & {r['synthesis']} & {r['sites']} & {100*r['relative']:+.1f}\\% & {r['wins']}/{r['ties']}/{r['losses']} & {100*r['objective_relative']:+.1f}\\%\\\\" for r in phase_summaries])
    data['coverage']={k:len(v) for k,v in rows.items()}
    (PAPER/'reviewer_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
