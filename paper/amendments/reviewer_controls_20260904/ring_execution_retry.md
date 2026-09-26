# Ring adapter execution retry

Recorded after Slurm job 16985454 failed, before running the retry. All 72 gap
evaluation tasks and 72 development tasks completed. Each of the 36 historical
confirmation tasks failed with `KeyError: 'token'` in legacy selector-curve
postprocessing, because the single-method adapter removed that dictionary key.
No completed historical ring rows or aggregate were produced by this attempt.

The separate retry adapter supplies the required internal dictionary key while
retaining the same RingPlacement kernel. It changes no objective, parameter,
seed, instance, penalty, endpoint, or optimizer setting. The returned method label
remains ring. Discarded selector curves do not enter this amendment's endpoints.
Only the 36 failed confirmation tasks are recomputed. The 72 successful gap
tasks, development rows, and penalty selection are reused byte-for-byte.
The failed logs and original source remain available; the corrected aggregate
will identify both original and retry jobs and the new source checksum.
