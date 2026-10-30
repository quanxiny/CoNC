#!/usr/bin/env python3
"""Recompute CoNC/Eva disagreement counts from released prediction records."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import gzip
import json

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ('11', '29', '47')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/case_studies/disagreements.json')
    args = parser.parse_args()
    cohorts = defaultdict(set)
    with (ROOT/'data/cohorts.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['cohort'] in ('source_common11342', 'mit_extended1059'):
                cohorts[r['dataset']].add(r['unit_id'])
    rows = defaultdict(dict)
    with gzip.open(ROOT/'data/predictions.csv.gz','rt',newline='') as f:
        for r in csv.DictReader(f):
            if r['dataset'] in cohorts and r['unit_id'] in cohorts[r['dataset']] and r['method'] in ('original_N', 'eva'):
                rows[r['dataset'],r['method'],r['seed']][r['unit_id']] = r
    reference = json.loads((ROOT/'data/case_studies/disagreement_summary.json').read_text())
    result = {'scope':'Saved predictions on fixed common IDs; no model or analyzer execution.', 'source':[], 'mit':[]}
    source_only, mit_errors = [], []
    for dataset, cohort in [('source','source'),('mit','mit')]:
        eva = rows[dataset,'eva','deterministic']
        assert set(eva) == cohorts[dataset]
        for seed in SEEDS:
            neural = rows[dataset,'original_N',seed]
            assert set(neural) == cohorts[dataset]
            counts = defaultdict(int); selected = set()
            for uid in sorted(neural):
                n, e = neural[uid], eva[uid]
                assert n['label'] == e['label'] and n['prediction'] in ('0','1') and e['prediction'] in ('0','1')
                y, npred, epred = map(int,(n['label'],n['prediction'],e['prediction']))
                if y == 1 and npred == 1 and epred == 0:
                    counts['conc_only_positive'] += 1; selected.add(uid)
                if y == 1 and npred == 0 and epred == 1: counts['eva_only_positive'] += 1
                if y == 0 and npred == 1 and epred == 0: counts['conc_only_false_alert'] += 1
                if y == 0 and npred == 0 and epred == 1: counts['eva_only_false_alert'] += 1
                if npred != y and epred == y:
                    counts['conc_wrong_eva_correct'] += 1
                    if dataset == 'mit': selected.add(uid)
            result[cohort].append({'seed':int(seed),'units':len(neural),**counts})
            (source_only if dataset == 'source' else mit_errors).append(selected)
    result['source_union_conc_only_positive'] = len(set.union(*source_only))
    result['source_stable_conc_only_positive'] = len(set.intersection(*source_only))
    result['mit_all_seed_wrong_eva_correct'] = len(set.intersection(*mit_errors))
    sr = reference['source']; mr = reference['mit']
    for actual, expected in zip(result['source'],sr['counts_by_seed']):
        for name, old in [('conc_only_positive','conc_only_positive'),('eva_only_positive','eva_only_positive'),('conc_only_false_alert','conc_only_false_alert'),('eva_only_false_alert','eva_only_false_alert')]:
            assert actual.get(name,0) == expected[old], (actual,expected,name)
    assert result['source_union_conc_only_positive'] == sr['union_conc_only_positive'] == 622
    assert result['source_stable_conc_only_positive'] == sr['stable_conc_only_positive'] == 594
    assert [r['conc_wrong_eva_correct'] for r in result['mit']] == [r['conc_wrong_eva_correct'] for r in mr['by_seed']]
    assert result['mit_all_seed_wrong_eva_correct'] == mr['all_seed_wrong_eva_correct'] == 221
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
