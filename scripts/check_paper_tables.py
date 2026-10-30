#!/usr/bin/env python3
"""Check mean F1 values against the paper's main and strict-study tables."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT/'outputs/replay/summary.csv',
                        help='summary.csv produced by scripts/replay_results.py')
    args = parser.parse_args()
    reference = json.loads((ROOT/'data/paper_table_f1.json').read_text())
    main_values = {(r['cohort'],r['method']):100*float(r['mean'])
                   for r in read(args.summary) if r['metric']=='F1'}
    strict = {(r['dataset'],r['method']):100*float(r['f1_mean'])
              for r in read(ROOT/'results/reference/mechanism_summary.csv')
              if r['cohort']=='own_coverage' and r['f1_mean']}
    checked = 0
    for scope, table, values in [('main',reference['main'],main_values),
                                 ('strict_supplement',reference['strict_supplement'],strict)]:
        for cohort, methods in table.items():
            for method, expected in methods.items():
                actual = values[cohort,method]
                if abs(actual - expected) >= 0.0051:
                    raise ValueError(f'{scope} {cohort}/{method}: {actual:.6f}% != {expected:.2f}%')
                checked += 1
    print(json.dumps({'status':'PASS','published_mean_f1_values_checked':checked,
                      'scope':'Published rounded means only; strict-study checks use saved summary values.'}))


if __name__ == '__main__':
    main()
