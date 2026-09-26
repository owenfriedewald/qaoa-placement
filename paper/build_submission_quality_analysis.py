"""Validate and summarize completed budget/quality data while phase jobs run."""
import json,os
from pathlib import Path
from collect_submission_study import validate
from build_submission_artifacts import quality_analysis,resource_analysis,structured_analysis,plots,eda_nontrivial


def main():
    assert os.environ.get('SLURM_JOB_ID')
    data=validate(kinds=('budget','quality','homogeneous','structured'))
    summary=quality_analysis(data)
    summary.update(resource_analysis(data))
    summary.update(structured_analysis(data))
    summary['eda_nontrivial']=eda_nontrivial()
    plots(summary)
    result=dict(job=os.environ['SLURM_JOB_ID'],counts={k:len(v) for k,v in data.items()},**summary)
    (Path(__file__).resolve().parent/'submission_quality_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['counts']))


if __name__=='__main__':main()
