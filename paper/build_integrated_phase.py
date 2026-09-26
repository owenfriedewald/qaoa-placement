"""Derive paired full-circuit completion summaries on the compute node."""
import json
import statistics
from collect_completed_phase import OUT, MIXER, read, validate


def build(paper, write):
    rows=read(OUT/'run_level.csv'); validate(rows,OUT)
    penalty=[r for r in read(MIXER/'acm_routing_qiskit252_20260904/routing_fresh_run_level.csv') if r['method']=='penalty_row_xy']
    zero=[r for r in read(MIXER/'acm_routing_qiskit252_20260904/routing_fresh_run_level.csv') if r['method']=='token']
    instances=[]
    metrics=('logical_cx','routed_ecr','routed_depth','routed_two_qubit_depth')
    for instance in sorted({r['instance_id'] for r in rows}):
        record=dict(instance_id=instance)
        for policy, group in [('zero',zero),('l1',rows),('penalty',penalty)]:
            for metric in metrics:
                record[policy+'_'+metric]=statistics.median(float(r[metric]) for r in group if r['instance_id']==instance)
        instances.append(record)
    summaries=[]
    for metric in metrics:
        a=[r['l1_'+metric] for r in instances]; b=[r['zero_'+metric] for r in instances]
        summaries.append(dict(metric=metric,instances=36,zero_median=statistics.median(b),
            completed_median=statistics.median(a),relative_median=statistics.median(x/y-1 for x,y in zip(a,b)),
            wins=sum(x<y for x,y in zip(a,b)),ties=sum(x==y for x,y in zip(a,b)),losses=sum(x>y for x,y in zip(a,b))))
    for result in summaries:
        metric=result['metric']
        pairs=[(r['l1_'+metric],r['penalty_'+metric]) for r in instances]
        result['versus_penalty_relative_median']=statistics.median(a/b-1 for a,b in pairs)
        result['versus_penalty_wins']=sum(a<b for a,b in pairs)
    write(paper/'tables/confirmation_completed_phase_instances.csv',instances)
    write(paper/'tables/confirmation_completed_phase_summary.csv',summaries)
    labels=dict(logical_cx='Logical CX',routed_ecr='Routed ECR',routed_depth='Total depth',routed_two_qubit_depth='Two-qubit depth')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Resource & Zero & Completed & Paired change & W/T/L\\',r'\midrule']
    for r in summaries:
        lines.append(f"{labels[r['metric']]} & {r['zero_median']:,.1f} & {r['completed_median']:,.1f} & {100*r['relative_median']:+.1f}\\% & {r['wins']}/{r['ties']}/{r['losses']}\\\\")
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (paper/'tables/confirmation_completed_phase.tex').write_text('\n'.join(lines)+'\n')
    return summaries
