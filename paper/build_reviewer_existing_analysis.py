"""Existing-data mechanism and cohort plots; run on Slurm only."""
import csv
import json
import os
from pathlib import Path
import statistics as st
import matplotlib.pyplot as plt
PAPER=Path(__file__).resolve().parent
MIXER=PAPER.parent/'experiments/qaoa_placement/mixer_reduction'


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Slurm required'
    historical=read(MIXER/'exact_phase_completion_20260904/run_level.csv')
    fresh=read(MIXER/'acm_gap_closure_20260904/phase_run_level.csv')
    points=[]
    for corpus,rows,syntheses in [('Historical',historical,['mask']),('Fresh',fresh,['beam','gray'])]:
        for synthesis in syntheses:
            selected=rows if corpus=='Historical' else [r for r in rows if r['synthesis']==synthesis and r['parameter_policy']=='symbolic']
            pairs={}
            for r in selected:
                key=(r['sites'],r['family'],r.get('repeat',r.get('instance_id')))
                pairs.setdefault(key,{})[r['policy']]=int(r['logical_cx'])
            for (m,f,i),v in pairs.items():
                m=int(m);assert set(v)=={'zero','l1'}
                points.append(dict(corpus=corpus,synthesis=synthesis,sites=m,family=f,instance_id=i,
                    free_fraction=1-m*m/(1<<(2*(m-1).bit_length())),zero_cx=v['zero'],l1_cx=v['l1'],reduction=1-v['l1']/v['zero']))
    write(PAPER/'tables/reviewer_free_fraction.csv',points)
    fig,axes=plt.subplots(1,3,figsize=(8,2.8),sharex=True,sharey=True)
    colors={6:'#2166ac',8:'#777777',9:'#b2182b',12:'#1b7837',16:'#333333'}
    for ax,(corpus,synthesis) in zip(axes,[('Historical','mask'),('Fresh','beam'),('Fresh','gray')]):
        group=[r for r in points if r['corpus']==corpus and r['synthesis']==synthesis]
        for m in (6,8,9,12,16):
            g=[r for r in group if r['sites']==m];x=g[0]['free_fraction']
            ax.scatter([x]*len(g),[100*r['reduction'] for r in g],s=10,alpha=.35,color=colors[m])
            ax.scatter(x,100*st.median(r['reduction'] for r in g),s=55,marker='_',color=colors[m],label=f'm={m}')
        ax.axhline(0,color='black',lw=.6);ax.set_title(f'{corpus}: {synthesis}');ax.set_xlabel('Unconstrained table fraction')
    axes[0].set_ylabel('Paired CX reduction (%)');axes[2].legend(fontsize=7,loc='upper left')
    fig.tight_layout();fig.savefig(PAPER/'figures/fig_reviewer_free_fraction.pdf');plt.close(fig)
    token=[r for r in read(MIXER/'token_permutation_p3_parameter_results.csv') if r['graph']=='line' and r['mixer_schedule']=='empty_prioritized' and r['optimization_objective']=='cvar_0.25' and r['parameter_protocol']=='optimized']
    row=[r for r in read(MIXER/'penalty_lambda_sweep_v1/run_level.csv') if float(r['penalty_lambda'])==5]
    keys=lambda rows:{(r['instance_id'],r['init_mode'],r['optimizer_seed']) for r in rows}
    assert keys(token)==keys(row) and len(keys(token))==len(token)==len(row)==24
    effects=[]
    for i in sorted({r['instance_id'] for r in token}):
        a=st.mean(float(r['optimal_probability']) for r in token if r['instance_id']==i)
        b=st.mean(float(r['optimal_probability']) for r in row if r['instance_id']==i)
        effects.append(dict(cohort='Development',instance_id=i,token=a,complete=b,delta=a-b,cap=40,starts=2,seeds=2))
    for name,path in [('Confirmation','confirmation_quality_instances.csv'),('Transfer','gap_quality_instances.csv')]:
        for r in read(PAPER/'tables'/path):
            if int(r['cells'])!=4 or r.get('training','analytic')!='analytic':continue
            effects.append(dict(cohort=name,instance_id=r['instance_id'],token=float(r['token_optimal_probability']),
                complete=float(r['penalty_row_xy_optimal_probability']),delta=float(r['delta']),cap=200,starts=3,seeds=4 if name=='Confirmation' else 2))
    write(PAPER/'tables/reviewer_cohort_effects.csv',effects)
    fig,ax=plt.subplots(figsize=(7.2,3));summaries=[]
    labels=[]
    for x,name in enumerate(['Development','Confirmation','Transfer']):
        g=[r for r in effects if r['cohort']==name];v=sorted(r['delta'] for r in g)
        # Deterministic horizontal spread is display only; no cross-cohort pairing.
        ax.scatter([x+(i-(len(v)-1)/2)*.5/max(1,len(v)-1) for i in range(len(v))],v,s=17,color='#2166ac',alpha=.7)
        mean=st.mean(v);ax.plot([x-.27,x+.27],[mean,mean],color='#b2182b',lw=2)
        labels.append(f"{name}\nN={len(g)}, cap {g[0]['cap']}")
        summaries.append(dict(cohort=name,instances=len(g),token=st.mean(r['token'] for r in g),complete=st.mean(r['complete'] for r in g),delta=mean))
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),labels);ax.set_ylabel('Token − complete-XY optimum mass')
    ax.set_title('Same size (4c/6s); different cohorts and protocols',fontsize=10)
    fig.tight_layout();fig.savefig(PAPER/'figures/fig_reviewer_cohorts.pdf');plt.close(fig)
    excluded=[r for r in effects if r['cohort']=='Confirmation' and r['instance_id']!='fresh4_sparse_grid_904000']
    data=dict(cohorts=summaries,confirmation_without_smoke=dict(instances=len(excluded),delta=st.mean(r['delta'] for r in excluded)),
        interpretation='Paired methods within instances; cohorts are unpaired and generator/protocol differences confound selection effects.')
    (PAPER/'reviewer_existing_summary.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
