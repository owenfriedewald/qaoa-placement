"""Derive the prospectively specified follow-up results on a Slurm compute node."""
import json
import os
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from collect_gap_study import OUT,validate
PAPER=Path(__file__).resolve().parent


def write(name,rows):
    import csv
    with (PAPER/'tables'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def table(name,columns,header,rows):
    (PAPER/'tables'/name).write_text('\n'.join([r'\begin{tabular}{'+columns+'}',r'\toprule',header+r'\\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}'])+'\n')


def paired_summary(pairs):
    a,b=np.asarray(pairs).T
    return dict(first_median=float(np.median(a)),reference_median=float(np.median(b)),
        relative_median=float(np.median(a/b-1)),wins=int(np.sum(a<b)),ties=int(np.sum(a==b)),losses=int(np.sum(a>b)))


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Generate plots/statistical resampling on Slurm')
    rows=validate(OUT)
    from collect_classical_accounting import OUT as ACCOUNTING, validate as validate_accounting
    rows['classical']=validate_accounting(ACCOUNTING)
    p=json.loads((OUT/'protocol.json').read_text());data={}
    phase=rows['phase'];phase_summaries=[];numeric=[]
    for m in (6,8,9,12,16):
        ids=[r['id'] for r in p['phase'] if len(r['sites'])==m]
        for synthesis in ('beam','gray'):
            keyed={(r['instance_id'],r['policy']):float(r['logical_cx']) for r in phase if r['synthesis']==synthesis and r['parameter_policy']=='symbolic'}
            pairs=[(keyed[i,'l1'],keyed[i,'zero']) for i in ids]
            phase_summaries.append(dict(sites=m,synthesis=synthesis,instances=len(ids),**paired_summary(pairs)))
        for numerator,denominator in [('gray_l1','gray_zero'),('gray_zero','beam_zero'),('library_zero','gray_zero'),('library_l1','gray_l1')]:
            def values(method):
                synthesis,policy=method.split('_')
                return {i:st.median(float(r['logical_cx']) for r in phase if r['instance_id']==i and r['synthesis']==synthesis and r['policy']==policy and r['parameter_policy']=='numeric') for i in ids}
            a,b=values(numerator),values(denominator)
            numeric.append(dict(sites=m,numerator=numerator,denominator=denominator,instances=len(ids),**paired_summary([(a[i],b[i]) for i in ids])))
    data['phase']=phase_summaries;data['numeric_phase']=numeric
    write('gap_phase_summary.csv',phase_summaries);write('gap_numeric_phase_summary.csv',numeric)
    table('gap_phase.tex','rlrrrr',r'Sites & Synthesis & Zero CX & L1 CX & Change & W/T/L',[
        f"{r['sites']} & {r['synthesis'].capitalize()} & {r['reference_median']:,.0f} & {r['first_median']:,.0f} & {100*r['relative_median']:+.1f}\\% & {r['wins']}/{r['ties']}/{r['losses']}\\\\" for r in phase_summaries])
    numeric_lines=[]
    for m in (6,8,9,12,16):
        group=[r for r in numeric if r['sites']==m]
        numeric_lines.append(str(m)+' & '+' & '.join(f"{100*r['relative_median']:+.1f}\\%" for r in group)+r'\\')
    table('gap_numeric_phase.tex','rrrrr',r'Sites & Gray L1/zero & Gray/beam zero & Library/Gray zero & Library/Gray L1',numeric_lines)
    routing=[];routing_instances=[]
    for topology in p['route_topologies']:
        for method in p['route_methods']:
            for instance in p['route_cases']:
                record=dict(topology=topology,method=method,instance_id=instance)
                for metric in ('routed_cx','routed_depth','routed_two_qubit_depth'):
                    record[metric]=st.median(float(r[metric]) for r in rows['route'] if r['instance_id']==instance and r['topology']==topology and r['method']==method)
                routing_instances.append(record)
        for method in p['route_methods']:
            selected=[r for r in routing_instances if r['topology']==topology and r['method']==method]
            reference=[r for r in routing_instances if r['topology']==topology and r['method']=='penalty_row_xy']
            record=dict(topology=topology,method=method,instances=12)
            for metric in ('routed_cx','routed_depth','routed_two_qubit_depth'):
                summary=paired_summary([(a[metric],b[metric]) for a,b in zip(selected,reference)])
                record.update({metric+'_'+k:v for k,v in summary.items()})
            routing.append(record)
    data['routing']=routing;write('gap_routing_summary.csv',routing);write('gap_routing_instances.csv',routing_instances)
    labels={'penalty_row_xy':'Row-XY','token_beam_zero':'Beam zero','token_beam_l1':'Beam L1','token_gray_zero':'Gray zero','token_gray_l1':'Gray L1'}
    table('gap_routing.tex','llrrrr',r'Topology & Circuit & CX & CX change & Depth change & CX wins',[
        f"{r['topology']} & {labels[r['method']]} & {r['routed_cx_first_median']:,.0f} & {100*r['routed_cx_relative_median']:+.1f}\\% & {100*r['routed_depth_relative_median']:+.1f}\\% & {r['routed_cx_wins']}/12\\\\" for r in routing])
    quality=[];quality_instances=[]
    for cells,training in [(4,'analytic'),(4,'finite2048'),(5,'analytic'),(6,'analytic')]:
        for spec in [r for r in p['quality'] if len(r['cells'])==cells]:
            record=dict(instance_id=spec['id'],family=spec['family'],cells=cells,sites=len(spec['sites']),training=training)
            for method in p['quality_methods']:
                group=[r for r in rows['quality'] if r['instance_id']==spec['id'] and r['method']==method and r['training']==training]
                for metric in ('optimal_probability','feasible_probability','evaluations','training_shots'):
                    record[method+'_'+metric]=st.mean(float(r[metric]) for r in group)
            record['delta']=record['token_optimal_probability']-record['penalty_row_xy_optimal_probability'];quality_instances.append(record)
        group=[r for r in quality_instances if r['cells']==cells and r['training']==training]
        rng=np.random.default_rng(p['bootstrap']['seed']);draws=np.zeros(p['bootstrap']['draws'])
        for family in sorted({r['family'] for r in group}):
            v=np.array([r['delta'] for r in group if r['family']==family]);draws+=v[rng.integers(len(v),size=(len(draws),len(v)))].sum(axis=1)
        lo,hi=np.percentile(draws/len(group),[2.5,97.5])
        quality.append(dict(cells=cells,sites=group[0]['sites'],training=training,instances=len(group),
            token_mean=st.mean(r['token_optimal_probability'] for r in group),penalty_mean=st.mean(r['penalty_row_xy_optimal_probability'] for r in group),
            token_legal=st.mean(r['token_feasible_probability'] for r in group),penalty_legal=st.mean(r['penalty_row_xy_feasible_probability'] for r in group),
            delta=st.mean(r['delta'] for r in group),positive=sum(r['delta']>1e-12 for r in group),ci_low=float(lo),ci_high=float(hi)))
    data['quality']=quality;write('gap_quality_summary.csv',quality);write('gap_quality_instances.csv',quality_instances)
    shot_sensitivity=[]
    for method in p['quality_methods']:
        ids=[r['id'] for r in p['quality'] if len(r['cells'])==4]
        pairs=[]
        for i in ids:
            values={mode:st.mean(float(r['optimal_probability']) for r in rows['quality'] if r['instance_id']==i and r['method']==method and r['training']==mode) for mode in ('analytic','finite2048')}
            pairs.append((values['finite2048'],values['analytic']))
        shot_sensitivity.append(dict(method=method,instances=12,analytic_mean=st.mean(b for a,b in pairs),finite_mean=st.mean(a for a,b in pairs),mean_delta=st.mean(a-b for a,b in pairs),finite_higher=sum(a>b+1e-12 for a,b in pairs)))
    data['shot_sensitivity']=shot_sensitivity;write('gap_shot_sensitivity.csv',shot_sensitivity)

    table('gap_quality.tex','llrrrrl',r'Size & Training & Token & Row-XY & Row legal & Wins & Difference interval',[
        f"{r['cells']}c/{r['sites']}s & {'2048 shots' if r['training']=='finite2048' else 'Exact'} & {r['token_mean']:.5f} & {r['penalty_mean']:.5f} & {r['penalty_legal']:.3f} & {r['positive']}/12 & [{r['ci_low']:.5f}, {r['ci_high']:.5f}]\\\\" for r in quality])
    from collect_penalty_calibration import OUT as CALIBRATION, validate as validate_calibration
    calibrated_rows=validate_calibration(CALIBRATION)['evaluation']
    selection=json.loads((CALIBRATION/'selection.json').read_text())
    calibration=[]
    for cells,training in [(4,'analytic'),(4,'finite2048'),(5,'analytic'),(6,'analytic')]:
        records=[]
        for spec in [r for r in p['quality'] if len(r['cells'])==cells]:
            token=next(r['token_optimal_probability'] for r in quality_instances if r['instance_id']==spec['id'] and r['training']==training)
            group=[r for r in calibrated_rows if r['instance_id']==spec['id'] and r['training']==training]
            penalty=st.mean(float(r['optimal_probability']) for r in group)
            records.append(dict(family=spec['family'],token=token,penalty=penalty,legal=st.mean(float(r['feasible_probability']) for r in group),delta=token-penalty))
        rng=np.random.default_rng(p['bootstrap']['seed']);draws=np.zeros(p['bootstrap']['draws'])
        for family in sorted({r['family'] for r in records}):
            values=np.array([r['delta'] for r in records if r['family']==family])
            draws+=values[rng.integers(len(values),size=(len(draws),len(values)))].sum(axis=1)
        lo,hi=np.percentile(draws/12,[2.5,97.5])
        calibration.append(dict(cells=cells,training=training,candidate=selection[str(cells)]['candidate'],
            token_mean=st.mean(r['token'] for r in records),penalty_mean=st.mean(r['penalty'] for r in records),penalty_legal=st.mean(r['legal'] for r in records),
            delta=st.mean(r['delta'] for r in records),positive=sum(r['delta']>1e-12 for r in records),ci_low=float(lo),ci_high=float(hi)))
    data['calibration']=calibration;write('gap_calibration_summary.csv',calibration)
    candidate_labels={'fixed5':'Fixed 5','quarter_bound':'$B/4$','strict_bound':'$1.000001B$','four_bound':'$4B$'}
    table('gap_calibration.tex','llrrrrl',r'Size & Training & Penalty & Row optimum & Row legal & Wins & Difference interval',[
        f"{r['cells']}c & {'2048 shots' if r['training']=='finite2048' else 'Exact'} & {candidate_labels[r['candidate']]} & {r['penalty_mean']:.5f} & {r['penalty_legal']:.3f} & {r['positive']}/12 & [{r['ci_low']:.5f}, {r['ci_high']:.5f}]\\\\" for r in calibration])
    controls=[]
    for cells in (4,5,6):
        for method in p['classical_methods']:
            for budget in p['classical_budgets']:
                group=[r for r in rows['classical'] if int(r['cells'])==cells and r['method']==method and int(r['budget'])==budget]
                controls.append(dict(cells=cells,method=method,budget=budget,ratio=st.mean(float(r['ratio']) for r in group),optimal_hit=st.mean(r['optimal_hit']=='True' for r in group)))
        for method in p['quality_methods']:
            for training in p['training_modes'][str(cells)]:
                for budget in p['classical_budgets']:
                    group=[r for r in rows['quality'] if int(r['cells'])==cells and r['method']==method and r['training']==training]
                    controls.append(dict(cells=cells,method=method+'_'+training,budget=budget,ratio=st.mean(float(r[f'expected_best_{budget}'])/float(r['exact_cost']) for r in group),optimal_hit=''))
    for cells in (4,5,6):
        for training in p['training_modes'][str(cells)]:
            for budget in p['classical_budgets']:
                group=[r for r in calibrated_rows if int(r['cells'])==cells and r['training']==training]
                controls.append(dict(cells=cells,method='calibrated_row_xy_'+training,budget=budget,
                    ratio=st.mean(float(r[f'expected_best_{budget}'])/float(r['exact_cost']) for r in group),optimal_hit=''))
    data['controls']=controls;write('gap_classical_summary.csv',controls)
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42});fig,axes=plt.subplots(1,3,figsize=(7,2.5))
    for ax,cells in zip(axes,(4,5,6)):
        methods=p['classical_methods']+[m+'_analytic' for m in p['quality_methods']]+['calibrated_row_xy_analytic']
        for method in methods:
            group=[r for r in controls if r['cells']==cells and r['method']==method]
            ax.plot([r['budget'] for r in group],[r['ratio'] for r in group],marker='.',label=method.replace('_analytic','').replace('_',' '))
        ax.set_xscale('log',base=2);ax.set_title(f'{cells} cells');ax.set_xlabel('Readouts / classical call cap');ax.set_ylabel('Best / exact cost')
    axes[0].legend(fontsize=5);fig.tight_layout();fig.savefig(PAPER/'figures/fig_gap_controls.pdf');plt.close(fig)
    finite=next(r for r in quality if r['training']=='finite2048')
    largest=next(r for r in quality if r['cells']==6)
    numerical_gray=[r for r in numeric if r['numerator']=='gray_zero' and r['denominator']=='beam_zero']
    text=("The generic-library and matched numeric comparisons are retained in the artifact; "
          "they are not pooled with symbolic counts. The Gray-zero implementation has "
          "median numeric CX changes relative to beam-zero of "
          +', '.join(f"{100*r['relative_median']:+.1f}\\% at {r['sites']} sites" for r in numerical_gray)+".\n\n"
)
    (PAPER/'gap_results.tex').write_text(text)
    (PAPER/'gap_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='controls'},indent=2))

if __name__=='__main__':main()
