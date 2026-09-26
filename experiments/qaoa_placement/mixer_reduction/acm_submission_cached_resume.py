"""Resume the frozen checkpoint runner with equivalent PyZX verification."""
import hashlib,sys
from pathlib import Path
from unittest.mock import patch
import acm_submission_checkpoint as driver
from acm_submission_fast_verify import UPSTREAM_SHA256
from acm_submission_cached_pivot import accelerate_pyzx as accelerate_verification


def main():
    original_run=driver.subprocess.run;original_dump=driver.dump
    this=str(Path(__file__).resolve());checkpoint=str(Path(driver.__file__).resolve())
    def run(command,*args,**kwargs):
        if isinstance(command,list) and len(command)>=3 and command[:3]==[sys.executable,'-B',checkpoint]:
            command=command.copy();command[2]=this
        return original_run(command,*args,**kwargs)
    def dump(path,value):
        if Path(path).name=='environment.json':
            value=dict(value,verification_mode='ordered-candidates-and-readonly-pivot-cache-equivalent',
                verification_runner_sha256=hashlib.sha256(Path(this).read_bytes()).hexdigest(),
                verification_matcher_upstream_sha256=UPSTREAM_SHA256)
        return original_dump(path,value)
    with accelerate_verification(),patch.object(driver.subprocess,'run',run),patch.object(driver,'dump',dump):
        driver.main()

if __name__=='__main__':main()
