"""Verify deposited evidence or regenerate analyses in a separate directory."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def paths(root):
 return [root,root/'paper',root/'experiments/qaoa_placement',root/'experiments/qaoa_placement/mixer_reduction']

def verify(root):
 sys.path[:0]=[str(p) for p in paths(root)]
 from verify_artifact import verify as check
 result=check();print(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=('verify','figures','analyze','snapshot'))
 p.add_argument('--output-dir',type=Path);p.add_argument('--output',type=Path)
 a=p.parse_args()
 if a.command=='verify':verify(ROOT);return
 if a.command=='snapshot':
  if a.output is None:p.error('snapshot requires --output')
  manifest=json.loads((ROOT/'MANIFEST.json').read_text())
  with a.output.open('x') as f:
   for row in manifest['sources']:
    path=ROOT/row['path'];h=hashlib.sha256(path.read_bytes()).hexdigest()
    f.write(h+'  '+row['path']+'\n')
  return
 if not os.environ.get('SLURM_JOB_ID'):p.error('Run analysis/plotting inside a Slurm allocation')
 if a.output_dir is None:p.error('analysis/plotting requires --output-dir')
 out=a.output_dir.resolve()
 if out==ROOT or ROOT in out.parents:p.error('Use a new output directory outside this artifact')
 if out.exists():p.error('Output directory already exists')
 verify(ROOT)
 shutil.copytree(ROOT,out,ignore=shutil.ignore_patterns('.git','.venv','__pycache__','.cache','slurm-*.out','slurm-*.err'))
 env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(p) for p in paths(out)),MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
 if a.command=='analyze':
  for name in ('build_confirmation_artifacts','build_gap_artifacts','build_reviewer_existing_analysis','build_reviewer_artifacts','build_submission_existing_analysis','build_submission_artifacts'):
   subprocess.run([sys.executable,'-B',str(out/'paper'/(name+'.py'))],cwd=out,env=env,check=True)
 # Render the final presentation from stored or regenerated summaries.
 code="import json,matplotlib.pyplot as plt; from pathlib import Path; from build_submission_artifacts import plots,editorial_table_labels; plt.rcParams.update({'font.size':8,'pdf.fonttype':42}); plots(json.loads(Path('paper/submission_summary.json').read_text())); editorial_table_labels()"
 subprocess.run([sys.executable,'-B','-c',code],cwd=out,env=env,check=True)
 print('Generated outputs:',out/'paper')

if __name__=='__main__':main()
