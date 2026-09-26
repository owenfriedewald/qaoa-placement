"""Verify coverage, source integrity, and generated artifact synchronization.

Standard-library checks only. --allow-incomplete supports work-in-progress
inspection and never certifies a submission-ready manuscript.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
MIXER = ROOT / 'experiments/qaoa_placement/mixer_reduction'
BUNDLE = MIXER / 'acm_confirmation_20260904'
ROUTING = MIXER / 'acm_routing_qiskit252_20260904'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def verify(allow_incomplete=False):
    protocol = json.loads((BUNDLE/'protocol.json').read_text())
    for name, expected in {**protocol['code_sha256'], **protocol['selection_data']}.items():
        assert digest(ROOT/name) == expected, name
    manifest = json.loads((BUNDLE/'benchmark_manifest.json').read_text())['instances']
    expected = {(i['instance_id'], m, start, str(seed)) for i in manifest
                for m in ('token', 'penalty_row_xy') for start in protocol['initializations']
                for seed in protocol['optimizer_seeds']}
    rows = read(BUNDLE/'quality_run_level.csv')
    keys = [(r['instance_id'], r['method'], r['init_mode'], r['optimizer_seed']) for r in rows]
    assert len(keys) == len(expected) and set(keys) == expected
    for row in rows:
        for name in ('optimal_probability', 'feasible_probability', 'improvement_probability'):
            assert math.isfinite(float(row[name])) and -1e-10 <= float(row[name]) <= 1+1e-10
        assert 0 < int(row['evaluations']) <= protocol['budget']
        assert len(json.loads(row['theta'])) == 5
    disclosure=json.loads((BUNDLE/'validation_case_disclosure.json').read_text())
    smoke=PAPER/'hpc/evidence/quality-smoke'
    for name, expected_hash in disclosure['smoke_sha256'].items():
        assert digest(smoke/name)==expected_hash
    assert {r['instance_id'] for r in read(smoke/'quality_run_level.csv')}=={disclosure['validation_instance']}
    phase = MIXER/'exact_phase_completion_20260904'
    phase_protocol = json.loads((phase/'protocol.json').read_text())
    for name, expected_hash in phase_protocol['source_sha256'].items():
        assert digest(MIXER/name) == expected_hash, name
    phase_rows = read(phase/'run_level.csv')
    expected_phase = {(str(m), f, str(r), p) for m, f, r in phase_protocol['cases'] for p in ('zero','l1')}
    phase_keys = [(r['sites'], r['family'], r['repeat'], r['policy']) for r in phase_rows]
    assert len(phase_keys) == len(expected_phase) and set(phase_keys) == expected_phase
    assert max(float(r['legal_distance_max_error']) for r in phase_rows) < 1e-8
    routing_counts = {}
    for cohort, count in [('frozen',32), ('fresh',36)]:
        path = ROUTING/f'routing_{cohort}_run_level.csv'
        route = read(path) if path.exists() else []
        routing_counts[cohort] = len(route)
        keys = [(r['instance_id'],r['method'],r['routing_seed']) for r in route]
        assert len(keys) == len(set(keys)), 'duplicated routing row'
        if not allow_incomplete:
            assert len(keys) == count*40, f'incomplete {cohort} routing'
        for r in route:
            assert r['symbolic_parameters']=='5' and r['preparation']=='True'
            assert all(r[k]=='False' for k in ('first_phase','row_penalties','measurements'))
    if not allow_incomplete:
        assert json.loads((ROUTING/'routing_environment.json').read_text())['versions']['qiskit']=='2.5.2'
    integrated_rows = 0
    if not allow_incomplete:
        from collect_completed_phase import OUT, validate
        integrated = read(OUT/'run_level.csv')
        validate(integrated, OUT)
        integrated_rows = len(integrated)
    gap_counts = {}
    if not allow_incomplete:
        from collect_gap_study import OUT as GAP, validate as validate_gap
        gap_counts = {k:len(v) for k,v in validate_gap(GAP).items()}
        from collect_penalty_calibration import OUT as CALIBRATION, validate as validate_calibration
        gap_counts.update({'penalty_'+k:len(v) for k,v in validate_calibration(CALIBRATION).items()})
        from collect_classical_accounting import OUT as ACCOUNTING, validate as validate_accounting
        gap_counts['corrected_classical']=len(validate_accounting(ACCOUNTING))
    reviewer_counts = {}
    if not allow_incomplete:
        from collect_reviewer_study import validate as validate_reviewer
        reviewer_counts = {k:len(v) for k,v in validate_reviewer().items()}
    submission_counts = {}
    if not allow_incomplete:
        from collect_submission_study import validate as validate_submission
        submission_counts = {k:len(v) for k,v in validate_submission().items()}
    metadata = json.loads((ROOT/'MANIFEST.json').read_text())
    for row in metadata['sources']:
        assert digest(ROOT/row['path']) == row['sha256'], row['path']
    tex = (PAPER/'main.tex').read_text()
    for name in ('gap_study.tex','submission_methods.tex','submission_results.tex'):
        if (PAPER/name).exists(): tex += (PAPER/name).read_text()
    citations = {x.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for x in block.split(',')}
    bib = set(re.findall(r'@\w+\{([^,]+),',(PAPER/'references.bib').read_text()))
    assert citations <= bib, citations-bib
    assert tex.count('\\begin{figure}') == tex.count('\\Description{')
    if not allow_incomplete:
        assert (PAPER/'confirmation_summary.json').exists()
        for path in [PAPER/'main.tex', PAPER/'main.pdf', PAPER/'references.bib', PAPER/'confirmation_numbers.tex', PAPER/'gap_study.tex', PAPER/'gap_results.tex', PAPER/'submission_methods.tex', PAPER/'submission_results.tex']:
            assert path.is_file(), path.name
        assert metadata['schema'] == 'qaoa-placement-public-v1'
    return dict(submission_rows=submission_counts, reviewer_rows=reviewer_counts, gap_rows=gap_counts, integrated_phase_rows=integrated_rows, quality_rows=len(rows), phase_rows=len(phase_rows), routing_rows=routing_counts,
                checked_sources=len(metadata['sources']), complete=not allow_incomplete)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete',action='store_true')
    result=verify(parser.parse_args().allow_incomplete)
    print(json.dumps(result,indent=2))
