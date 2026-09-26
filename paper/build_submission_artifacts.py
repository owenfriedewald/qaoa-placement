"""Generate submission-control claims, uncertainty and figures on Slurm."""
import json,os,statistics as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collect_submission_study import validate,OUT,read
from build_submission_existing_analysis import interval,tests,holm
from build_gap_artifacts import table,write
PAPER=Path(__file__).resolve().parent;M=OUT.parent


def relative_summary(pairs):
    values=np.array([a/b-1 for a,b in pairs]);q=np.percentile(values,[25,50,75])
    return dict(relative=float(q[1]),q25=float(q[0]),q75=float(q[2]),wins=sum(a<b for a,b in pairs),ties=sum(a==b for a,b in pairs),losses=sum(a>b for a,b in pairs))


def main():
    assert os.environ.get('SLURM_JOB_ID')
    data=validate();summary={};phase=data['phase'];summaries=[]
    for cohort,flow,m in [(a,b,c) for a in ('historical','fresh') for b in ('beam','gray','gray_numeric','library','pyzx') for c in (6,8,9,12,16)]:
        group=[r for r in phase if r['cohort']==cohort and r['flow']==flow and int(r['sites'])==m and float(r['eta'])==0]
        keyed={(r['instance_id'],r['policy']):r for r in group};ids=sorted({r['instance_id'] for r in group})
        for reference in ('zero','virtual','nearest','mean'):
            pairs=[(int(keyed[i,'l1']['two_qubit']),int(keyed[i,reference]['two_qubit'])) for i in ids]
            levels=np.percentile([a for a,b in pairs],[25,50,75]);base=np.percentile([b for a,b in pairs],[25,50,75])
            summaries.append(dict(cohort=cohort,flow=flow,sites=m,reference=reference,n=len(ids),l1_median=float(levels[1]),l1_q25=float(levels[0]),l1_q75=float(levels[2]),reference_median=float(base[1]),reference_q25=float(base[0]),reference_q75=float(base[2]),**relative_summary(pairs)))
    summary['extensions']=summaries;write('submission_extensions_summary.csv',summaries)
    preprocessing=[]
    for cohort,m,policy in [(a,b,c) for a in ('historical','fresh') for b in (6,8,9,12,16) for c in ('zero','l1','virtual','nearest','mean')]:
        g=[r for r in phase if r['cohort']==cohort and int(r['sites'])==m and r['policy']==policy and r['flow']=='gray']
        record=dict(cohort=cohort,sites=m,policy=policy,n=len(g),max_residual=max(float(r['valid_residual']) for r in g))
        for key in ('solve_seconds','synthesis_seconds','terms','weighted_norm','one_qubit','rz','two_qubit','depth'):
            q=np.percentile([float(r[key]) for r in g],[25,50,75])
            record[key]=float(q[1]);record[key+'_q25']=float(q[0]);record[key+'_q75']=float(q[2])
        preprocessing.append(record)
    summary['preprocessing']=preprocessing;write('submission_preprocessing.csv',preprocessing)
    table('submission_extensions.tex','llrrrrr',r'Corpus & $m$ & L1 CX [IQR] & vs zero & vs virtual & vs nearest & vs mean',[
        f"{'100-case' if a=='historical' else '60-case'} & {m} & "+
        f"{next(r for r in summaries if r['cohort']==a and r['flow']=='gray' and r['sites']==m)['l1_median']:.0f} "+
        f"[{next(r for r in summaries if r['cohort']==a and r['flow']=='gray' and r['sites']==m)['l1_q25']:.0f}, {next(r for r in summaries if r['cohort']==a and r['flow']=='gray' and r['sites']==m)['l1_q75']:.0f}] & "+
        ' & '.join(f"{100*next(r for r in summaries if r['cohort']==a and r['flow']=='gray' and r['sites']==m and r['reference']==b)['relative']:+.1f}\\%" for b in ('zero','virtual','nearest','mean'))+r'\\'
        for a in ('historical','fresh') for m in (6,9,12)])
    numeric=[]
    for cohort,m,policy in [(a,b,c) for a in ('historical','fresh') for b in (6,8,9,12,16) for c in ('zero','l1','virtual','nearest','mean')]:
        g=[r for r in phase if r['cohort']==cohort and int(r['sites'])==m and r['policy']==policy];keyed={(r['instance_id'],r['flow']):r for r in g}
        ids=sorted({r['instance_id'] for r in g})
        for flow in ('library','pyzx'):
            numeric.append(dict(cohort=cohort,sites=m,policy=policy,flow=flow,**relative_summary([(int(keyed[i,flow]['two_qubit']),int(keyed[i,'gray_numeric']['two_qubit'])) for i in ids])))
    summary['numeric']=numeric;write('submission_numeric_summary.csv',numeric)
    table('submission_numeric.tex','lrrr',r'Policy & $m$ & Library / Gray (\%) & PyZX / Gray (\%)',[
        f"{policy} & {m} & "+' & '.join(f"{100*next(r for r in numeric if r['cohort']=='fresh' and r['sites']==m and r['policy']==policy and r['flow']==flow)['relative']:+.1f}" for flow in ('library','pyzx'))+r'\\'
        for policy in ('l1','virtual') for m in (6,9,12)])
    truncation=[]
    for cohort,m,policy,eta in [(a,b,c,d) for a in ('historical','fresh') for b in (6,8,9,12,16) for c in ('l1','virtual') for d in (.001,.01,.05)]:
        g=[r for r in phase if r['cohort']==cohort and int(r['sites'])==m and r['flow']=='gray'];keyed={(r['instance_id'],r['policy']):r for r in g};ids=sorted({r['instance_id'] for r in g});name=f'{policy}_{eta}'
        truncation.append(dict(cohort=cohort,sites=m,policy=policy,eta=eta,max_residual=max(float(keyed[i,name]['valid_residual']) for i in ids),max_cost_error_bound=max(float(keyed[i,name]['cost_error_bound']) for i in ids),**relative_summary([(int(keyed[i,name]['two_qubit']),int(keyed[i,policy]['two_qubit'])) for i in ids])))
    summary['truncation']=truncation;write('submission_truncation_summary.csv',truncation)
    table('submission_truncation.tex','lrrrr',r'Policy & $m$ & $\eta$ & CX change (\%) & Max. valid residual',[
        f"{r['policy']} & {r['sites']} & {r['eta']:.3f} & {100*r['relative']:+.1f} & {r['max_residual']:.3g}\\\\"
        for r in truncation if r['cohort']=='fresh' and r['sites'] in (6,9,12) and r['eta'] in (.001,.05)])
    summary.update(resource_analysis(data))
    summary.update(quality_analysis(data))
    summary.update(structured_analysis(data))
    summary["numeric_precision"]=numeric_precision()
    summary["eda_nontrivial"]=eda_nontrivial()
    plots(summary)
    editorial_table_labels()
    (PAPER/'submission_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def structured_analysis(data):
    summary={}
    structured=data['structured'];summary['structured']=structured
    for r in structured:
        if r['kind']=='growth':
            width=int(r['width'])
            assert int(r['terms'])==5*(1<<width)-4*width-4,(width,r['terms'])
    table('submission_structured.tex','lrrrr',r'Coordinate set & $m$ & Zero CX & L1 CX & Structured CX',[
        f"{kind} & {m} & "+' & '.join(next(r['gray_cx'] for r in structured if r['kind']==kind and r['sites']==str(m) and r['policy']==policy) for policy in ('zero','l1','structured'))+r'\\' for kind in ('row','grid') for m in (6,8,9,12,16)])
    return summary


def resource_analysis(data):
    summary={}
    # All targets share the same 12 cases and Gray/L1 contract here.
    routes=[]
    for r in data['homogeneous']:
        if r['component']=='full':routes.append(dict(r))
    old=read(M/'acm_reviewer_controls_20260904/workspace/routing_run_level.csv')
    for r in old:
        if r['cohort']=='gap':routes.append(dict(instance_id=r['instance_id'],method=r['method'],topology=r['topology'],seed=r['seed'],two_qubit=r['routed_two_qubit'],depth=r['routed_depth'],active_qubits=r['active_qubits'],circuit_sha256=r['circuit_sha256']))
    for i,method in {(r['instance_id'],r['method']) for r in routes}:
        g=[r for r in routes if r['instance_id']==i and r['method']==method]
        assert len(g)==15 and {r['topology'] for r in g}=={'sherbrooke','line25','grid25'}
        assert len({r['circuit_sha256'] for r in g})==1,(i,method,'circuit contract differs by target')
        assert len({int(r['active_qubits']) for r in g})==1
    medians={}
    for target in ('sherbrooke','line25','grid25'):
        ids=sorted({r['instance_id'] for r in routes if r['topology']==target});assert len(ids)==12
        for i in ids:
            for method in ('serial','parallel','complete','ring'):
                g=[r for r in routes if r['topology']==target and r['instance_id']==i and r['method']==method]
                assert len(g)==5 and {int(r['seed']) for r in g}=={211,223,227,229,233}
                medians[target,i,method]={key:st.median(float(r[key]) for r in g) for key in ('two_qubit','depth')}
    resource=[]
    for target in ('sherbrooke','line25','grid25'):
        ids=sorted({i for t,i,m in medians if t==target})
        for method,reference in [('parallel','serial'),('serial','complete'),('serial','ring'),('parallel','complete'),('parallel','ring')]:
            for metric in ('two_qubit','depth'):
                pairs=[(medians[target,i,method][metric],medians[target,i,reference][metric]) for i in ids]
                resource.append(dict(target=target,method=method,reference=reference,metric=metric,n=12,**relative_summary(pairs)))
    summary['resources']=resource;write('submission_resources_summary.csv',resource)
    table('submission_resources.tex','lllrr',r'Target & Token & Reference & 2q (\%) [IQR] & Depth (\%) [IQR]',[
        f"{t} & {m} & {b} & "+' & '.join(f"{100*r['relative']:+.1f} [{100*r['q25']:+.1f}, {100*r['q75']:+.1f}]" for metric in ('two_qubit','depth') for r in resource if r['target']==t and r['method']==m and r['reference']==b and r['metric']==metric)+r'\\'
        for t in ('sherbrooke','line25','grid25') for m,b in [('parallel','serial'),('parallel','complete'),('parallel','ring')]])
    components=[]
    for target,method,component in [(a,b,c) for a in ('sherbrooke','line25','grid25') for b in ('serial','parallel','complete','ring') for c in ('prep','mixer','phase')]:
        g=[r for r in data['homogeneous'] if r['topology']==target and r['method']==method and r['component']==component];assert len(g)==60
        record=dict(target=target,method=method,component=component,active_qubits=int(g[0]['active_qubits']),ancillas=1 if method=='serial' else 3 if method=='parallel' else 0,n=12,seeds=5)
        for key in ('one_qubit','rz','two_qubit','depth','logical_one_qubit','logical_rz','logical_two_qubit','logical_depth'):
            v=[st.median(float(r[key]) for r in g if r['instance_id']==i) for i in sorted({r['instance_id'] for r in g})]
            record[key]=st.median(v);record[key+'_q25']=float(np.percentile(v,25));record[key+'_q75']=float(np.percentile(v,75))
        components.append(record)
    summary['components']=components;write('submission_components.csv',components)
    table('submission_components.tex','llrrrrrr',r'Method & Component & Data & Anc. & 1q & $R_Z$ & CX & Depth',[
        f"{r['method']} & {r['component']} & {r['active_qubits']-r['ancillas']} & {r['ancillas']} & "+' & '.join(f"{r[k]:.0f}" for k in ('logical_one_qubit','logical_rz','logical_two_qubit','logical_depth'))+r'\\' for r in components if r['target']=='sherbrooke'])
    coordinate=read(M/'acm_reviewer_controls_20260904/phase/phase_run_level.csv')
    coordinate_summary=[]
    for cohort,flow in [('historical','mask'),('gap','beam'),('gap','gray')]:
        for m in (6,9,12):
            g=[r for r in coordinate if r['cohort']==cohort and r['synthesis']==flow and int(r['sites'])==m]
            keyed={(r['instance_id'],r['policy']):r for r in g};ids=sorted({r['instance_id'] for r in g})
            pairs=[(int(keyed[i,'separable']['logical_cx']),int(keyed[i,'dense']['logical_cx'])) for i in ids]
            coordinate_summary.append(dict(cohort=cohort,flow=flow,sites=m,n=len(ids),norm=st.median(float(keyed[i,'separable']['objective'])/float(keyed[i,'dense']['objective'])-1 for i in ids),**relative_summary(pairs)))
    write('submission_coordinate_dispersion.csv',coordinate_summary)
    table('reviewer_coordinate.tex','llrrrl',r'Corpus & Flow & $m$ & CX change & IQR & W/T/L',[
        f"{'100-case' if r['cohort']=='historical' else '60-case'} & {r['flow']} & {r['sites']} & {100*r['relative']:+.1f}\\% & [{100*r['q25']:+.1f}, {100*r['q75']:+.1f}] & {r['wins']}/{r['ties']}/{r['losses']}\\\\" for r in coordinate_summary])
    return summary


def quality_analysis(data):
    summary={}
    quality=data['quality'];instances=[]
    for i,method,p in sorted({(r['instance_id'],r['method'],int(r['p'])) for r in quality}):
        g=[r for r in quality if r['instance_id']==i and r['method']==method and int(r['p'])==p];assert len(g)==6
        record=dict(instance_id=i,method=method,p=p,family=g[0]['family'])
        for key in ('optimal_probability','feasible_probability','conditional_optimal','expected_best_128','expected_best_512','evaluations'):
            record[key]=st.mean(float(r[key]) for r in g)
        instances.append(record)
    write('submission_quality_instances.csv',instances)
    selections=json.loads((OUT/'quality/selection.json').read_text());matched=[]
    for selection in selections:
        method=selection['method'];p=selection['p'];base=[r for r in instances if r['method']==method and r['p']==p];token={r['instance_id']:r for r in instances if r['method']=='token' and r['p']==3}
        for metric in ('optimal_probability','conditional_optimal','expected_best_128','feasible_probability'):
            values=np.array([token[r['instance_id']][metric]-r[metric] for r in base]);lo,hi=interval(values,[r['family'] for r in base]);record={k:v for k,v in selection.items() if k!='levels'}
            record.update(metric=metric,n=len(base),token_mean=st.mean(token[r['instance_id']][metric] for r in base),baseline_mean=st.mean(r[metric] for r in base),delta=float(values.mean()),median_delta=float(np.median(values)),ci_low=lo,ci_high=hi,**tests(values,metric.startswith('expected_best')));matched.append(record)
    for metric in ('optimal_probability','conditional_optimal','expected_best_128','feasible_probability'):
        for test in ('wilcoxon','sign'):holm([r for r in matched if r['metric']==metric],test)
    summary['matched']=matched;write('submission_matched_summary.csv',matched)
    table('submission_matched.tex','lllrrl',r'Target & Row graph & Budget & $p$ & Used (\%) & Token$-$Row optimum [95\% CI]',[
        f"{r['topology']} & {r['method']} & {'2q' if r['endpoint']=='two_qubit' else 'depth'} & {r['p']} & {100*r['row']/r['token']:.1f} & {r['delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]\\\\" for r in matched if r['metric']=='optimal_probability'])
    sweeps=[]
    for method,p in sorted({(r['method'],r['p']) for r in instances}):
        g=[r for r in instances if r['method']==method and r['p']==p];lo,hi=interval([r['optimal_probability'] for r in g],[r['family'] for r in g]);sweeps.append(dict(method=method,p=p,n=len(g),mean=st.mean(r['optimal_probability'] for r in g),ci_low=lo,ci_high=hi,mean_calls=st.mean(r['evaluations'] for r in g)))
    summary['sweep']=sweeps;write('submission_sweep.csv',sweeps)
    return summary


def numeric_precision():
    import hashlib,math,sys
    import pyzx as zx
    from qiskit import qasm2
    sys.path.insert(0,str(M));sys.path.insert(0,str(M.parent))
    from acm_gap_study import problem
    from acm_submission_methods import phase_from_coeff
    from scipy.linalg import hadamard
    records=[];distance_checks=[];walsh={}
    for path in sorted((OUT/'phase/cases').glob('*.json')):
        obj=json.loads(path.read_text());p=problem(obj['instance'])
        m=len(p.sites);k=(m-1).bit_length();dimension=1<<(2*k)
        if dimension not in walsh:walsh[dimension]=hadamard(dimension)
        indices=[a|(b<<k) for b in range(m) for a in range(m)]
        truth=np.array([sum(abs(p.sites[a][q]-p.sites[b][q]) for q in (0,1)) for b in range(m) for a in range(m)])
        for name,values in obj['coefficients'].items():
            vector=np.array(values);assert len(vector)==dimension and np.isfinite(vector).all()
            error=float(max(abs((walsh[dimension]@vector)[indices]-truth)))
            eta=float(name.rsplit('_',1)[1]) if '_' in name else 0.
            assert error<(1e-8 if eta==0 else eta*max(truth)+1e-10),(obj['instance']['id'],name,error)
            distance_checks.append(dict(instance_id=obj['instance']['id'],policy=name,eta=eta,valid_residual=error,cost_error_bound=error*sum(abs(w) for u,v,w in p.nets)))
        for policy in ('zero','l1','virtual','nearest','mean'):
            c=phase_from_coeff(p,.371,np.array(obj['coefficients'][policy]))
            parsed=zx.Circuit.from_qasm(qasm2.dumps(c))
            original=[float(op.operation.params[0]) for op in c.data if op.operation.name=='rz']
            phases=[float(g.phase)*math.pi for g in parsed.gates if g.name=='ZPhase']
            assert len(original)==len(phases)
            parities=[1<<q for q in range(c.num_qubits)];supports=[]
            for op in c.data:
                bits=[c.find_bit(q).index for q in op.qubits]
                if op.operation.name=='cx':parities[bits[1]]^=parities[bits[0]]
                elif op.operation.name=='rz':supports.append(parities[bits[0]])
                else:raise AssertionError(op.operation.name)
            assert parities==[1<<q for q in range(c.num_qubits)]
            assert len(supports)==len(set(supports))
            errors=[abs((a-b+math.pi)%(2*math.pi)-math.pi) for a,b in zip(original,phases)]
            bound=sum(errors)/2
            records.append(dict(instance_id=obj['instance']['id'],cohort='historical' if obj['instance']['id'].startswith('historical_') else 'fresh',sites=len(p.sites),policy=policy,rotations=len(phases),raw_gray_cx=c.count_ops().get('cx',0),global_supports_sha256=hashlib.sha256(json.dumps(sorted(supports)).encode()).hexdigest(),max_angle_error=max(errors,default=0),phase_operator_bound_up_to_global=bound))
    assert len(records)==800
    assert len(distance_checks)==1760
    write('submission_numeric_precision.csv',records)
    write('submission_distance_verification.csv',distance_checks)
    support_summary=[]
    for cohort,m in [(a,b) for a in ('historical','fresh') for b in (6,8,9,12,16)]:
        g=[r for r in records if r['cohort']==cohort and r['sites']==m and r['policy']=='zero']
        support_summary.append(dict(cohort=cohort,sites=m,n=len(g),terms_min=min(r['rotations'] for r in g),terms_median=st.median(r['rotations'] for r in g),terms_max=max(r['rotations'] for r in g),raw_cx_median=st.median(r['raw_gray_cx'] for r in g),distinct_support_sets=len({r['global_supports_sha256'] for r in g})))
    write('submission_support_summary.csv',support_summary)
    return dict(circuits=len(records),coefficient_vectors=len(distance_checks),max_exact_distance_residual=max(r['valid_residual'] for r in distance_checks if r['eta']==0),max_operator_bound=max(r['phase_operator_bound_up_to_global'] for r in records),interpretation='Triangle-inequality bound for QASM angle conversion; each extracted ZX circuit also passed the run-time identity check against its parsed input. All stored exact/approximate vectors are independently reconstructed on their valid domains.')


def eda_nontrivial():
    import sys
    sys.path.insert(0,str(M.parent))
    from eda_bridge.run_real_benchmark import load_landscape,greedy_search
    base=M.parent/'eda_bridge/results/orfs_nangate45_six_design_72window_v1'
    windows=json.loads((base/'benchmark_manifest.json').read_text())['windows']
    records={r['window_id']:r for r in read(base/'localization_and_greedy_analysis_v1/window_level.csv')}
    rows=[]
    for w in windows:
        model=load_landscape(w);initial=model.cost_by_assignment[model.initial]
        _,cost,calls=greedy_search(model,32)
        r=records[w['window_id']]
        assert int(float(r['initial_hpwl_dbu']))==initial
        rows.append(dict(window_id=w['window_id'],initial=initial,exact=model.exact_cost,nontrivial=initial>model.exact_cost,
            qaoa_hit=r['observed_any_exact_hit_4096']=='True',greedy_hit=cost==model.exact_cost,neighbor_queries=calls,total_with_initial=calls+1))
    assert len(rows)==72 and sum(r['qaoa_hit'] for r in rows)==71 and sum(r['greedy_hit'] for r in rows)==69
    selected=[r for r in rows if r['nontrivial']];assert len(selected)==48
    classical=read(base/'matched_classical_budget_analysis_v3/run_level.csv');ids={r['window_id'] for r in selected};subgroups=[]
    for method,budget in sorted({(r['method'],int(r['budget'])) for r in classical}):
        g=[r for r in classical if r['method']==method and int(r['budget'])==budget and r['window_id'] in ids]
        by_window={i:[r for r in g if r['window_id']==i] for i in ids}
        assert len(by_window)==48 and all(by_window.values())
        subgroups.append(dict(method=method,budget=budget,windows=48,mean_hit_fraction=st.mean(st.mean(r['exact_hit']=='True' for r in rs) for rs in by_window.values())))
    write('submission_eda_windows.csv',rows);write('submission_eda_classical.csv',subgroups)
    return dict(nontrivial=48,qaoa_hits=sum(r['qaoa_hit'] for r in selected),greedy_hits=sum(r['greedy_hit'] for r in selected),classical=subgroups)


def plots(summary):
    points=read(PAPER/'tables/reviewer_free_fraction.csv');fig,axes=plt.subplots(1,3,figsize=(7.2,2.9),sharey=True)
    for ax,(cohort,flow) in zip(axes,[('Historical','mask'),('Fresh','beam'),('Fresh','gray')]):
        groups=[[100*float(r['reduction']) for r in points if r['corpus']==cohort and r['synthesis']==flow and int(r['sites'])==m] for m in (6,8,9,12,16)]
        ax.boxplot(groups,tick_labels=[6,8,9,12,16],widths=.55,showfliers=True);ax.axhline(0,color='gray',lw=.5);ax.set_xlabel('Number of valid sites m');ax.set_title({'Historical':'Initial','Fresh':'Replication'}[cohort]+': '+{'mask':'fixed order','beam':'beam search','gray':'Gray'}[flow])
    axes[0].set_ylabel('CX reduction from zero extension (%)');fig.tight_layout();fig.savefig(PAPER/'figures/fig_reviewer_free_fraction.pdf');plt.close(fig)
    effects=read(PAPER/'tables/reviewer_cohort_effects.csv');fig,ax=plt.subplots(figsize=(7.2,3))
    for x,name in enumerate(['Development','Confirmation','Transfer']):
        g=[r for r in effects if r['cohort']==name];v=np.array([float(r['delta']) for r in g]);lo,hi=interval(v,['cohort']*len(v))
        ax.scatter(x+np.linspace(-.2,.2,len(v)),sorted(v),s=12,alpha=.6)
        ax.errorbar(x,float(v.mean()),yerr=[[v.mean()-lo],[hi-v.mean()]],fmt='s',color='black',capsize=4)
    ax.set_xticks(range(3),[f'{name}\nN={sum(r["cohort"]==name for r in effects)} instances' for name in ['Development','Confirmation','Transfer']]);ax.set_ylabel('Token − complete-XY optimum probability');ax.axhline(0,color='gray',lw=.5);fig.tight_layout();fig.savefig(PAPER/'figures/fig_reviewer_cohorts.pdf');plt.close(fig)
    hist=[]
    for p in (M/'acm_reviewer_controls_20260904/phase/phase').glob('*.json'):
        obj=json.loads(p.read_text())
        if 'instance' not in obj:continue
        m=len(obj['instance']['sites']);cohort='historical' if obj['instance']['id'].startswith('historical_') else 'fresh'
        for policy in ('zero','dense','separable'):
            for weight in range(1,2*(m-1).bit_length()+1):
                hist.append(dict(instance_id=obj['instance']['id'],cohort=cohort,sites=m,policy=policy,popcount=weight,terms=sum(abs(v)>1e-10 and mask.bit_count()==weight for mask,v in enumerate(obj[policy]))))
    assert len({r['instance_id'] for r in hist})==160
    write('submission_popcount.csv',hist);fig,axes=plt.subplots(2,3,figsize=(7.2,4.5))
    for row,cohort in enumerate(('historical','fresh')):
        for col,m in enumerate((6,9,12)):
            ax=axes[row,col];weights=list(range(1,2*(m-1).bit_length()+1))
            for offset,policy in enumerate(('zero','dense','separable')):
                values=[st.mean(r['terms'] for r in hist if r['cohort']==cohort and r['sites']==m and r['policy']==policy and r['popcount']==w) for w in weights]
                ax.bar(np.array(weights)+(offset-1)*.25,values,width=.25,label={'dense':'joint L1','separable':'separate L1','zero':'zero'}[policy])
            ax.set_title(f"{'Initial' if cohort=='historical' else 'Replication'}, m={m}",fontsize=10);ax.set_xlabel('Pauli weight');ax.set_ylabel('Mean nonzero terms')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=9)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(PAPER/'figures/fig_submission_popcount.pdf');plt.close(fig)
    quality_plot(summary)


def editorial_table_labels():
    """Use manuscript terminology in TeX labels without changing table values."""
    for name in ('submission_extensions', 'reviewer_coordinate',
                 'submission_resources', 'submission_components', 'submission_matched',
                 'submission_numeric', 'submission_truncation'):
        path = PAPER / 'tables' / (name + '.tex')
        text = path.read_text()
        for old, new in (
            ('Corpus & Flow', 'Cohort & Synthesis'), ('Corpus &', 'Cohort &'),
            ('Policy &', 'Extension &'), ('l1 &', 'L1 &'),
            ('Library / Gray', 'Generic / Gray'),
            ('sherbrooke &', 'Sherbrooke &'), ('line25 &', 'Path &'),
            ('grid25 &', 'Grid &'), (' & mask &', ' & fixed order &'),
            (' & gray &', ' & Gray &'), (' & beam &', ' & beam search &'),
            (' & prep &', ' & preparation &')):
            text = text.replace(old, new)
        path.write_text(text)


def quality_plot(summary):
    # Keep standalone and full-manuscript regeneration visually identical.
    with plt.rc_context({'font.size':10, 'pdf.fonttype':3}):
        fig,ax=plt.subplots(figsize=(6.5,3))
        for method in ('token','complete','ring','random_token'):
            g=sorted([r for r in summary['sweep'] if r['method']==method],key=lambda r:r['p'])
            assert g,method
            means=np.array([r['mean'] for r in g]);ax.errorbar([r['p'] for r in g],means,yerr=[means-[r['ci_low'] for r in g],[r['ci_high'] for r in g]-means],marker='D' if method=='random_token' else 'o',linestyle='none' if method=='random_token' else '-',capsize=3,label='random token' if method=='random_token' else method)
        ax.set_xlabel('Layers p (200-call cap)');ax.set_ylabel('Mean optimum probability');ax.legend();fig.tight_layout();fig.savefig(PAPER/'figures/fig_submission_depth.pdf');plt.close(fig)

if __name__=='__main__':main()
