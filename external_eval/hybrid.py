"""Apply CoNC+Trace+Eva (H) and its inference-branch deletions."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from .boundary_optimization.script.hybrid_retrain_core import combine_decision


def state(value):
    if value in (None, '', 'UNKNOWN', 'NA'):
        return None
    if type(value) is bool or str(value) not in ('0', '1'):
        raise ValueError('Expected 0, 1, or UNKNOWN')
    return int(value)


def branches(neural, address, eva):
    """D known > Eva positive > N. Eva zero is never a safety certificate."""
    n, d, e = map(state, (neural, address, eva))
    full = combine_decision(n, 'UNKNOWN' if d is None else d, 'UNKNOWN' if e is None else e)['prediction']
    return {'full_H': state(full),
            'without_D': 1 if e == 1 else n,
            'without_Eva': d if d is not None else n,
            'N_only': n,
            'without_N': d if d is not None else 1 if e == 1 else None}


def self_test():
    for n in (None, 0, 1):
        for d in (None, 0, 1):
            for e in (None, 0, 1):
                result = branches(n, d, e)
                assert result['full_H'] == (d if d is not None else 1 if e == 1 else n)
                assert result['without_N'] == (d if d is not None else 1 if e == 1 else None)
    assert branches(1, 0, 1)['full_H'] == 0
    assert branches(None, None, 0)['full_H'] is None
    return {'status': 'PASS', 'decision_combinations': 27}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--input', type=Path, help='predictions.csv or predictions.csv.gz')
    parser.add_argument('--method', default='conc', help='conc is historical original_N; fallback_weighted is residual_N')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.method = {'conc': 'original_N', 'fallback_weighted': 'residual_N',
                   'ordinary_continuation': 'ordinary_N', 'class_matched': 'classmatch_N'}.get(args.method, args.method)
    if args.self_test:
        print(json.dumps(self_test()))
        return
    if args.input is None or args.output is None:
        parser.error('--input and --output are required unless --self-test is selected')
    opener = gzip.open if args.input.suffix == '.gz' else open
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with opener(args.input, 'rt', newline='') as stream, args.output.open('x', newline='') as output:
        reader = csv.DictReader(stream)
        writer = csv.DictWriter(output, fieldnames=['dataset', 'seed', 'unit_id', 'label', 'group', 'pair_id', 'policy', 'prediction'])
        writer.writeheader()
        for row in reader:
            if row['method'] != args.method:
                continue
            for policy, value in branches(row['prediction'], row['D_trace'], row['Eva']).items():
                writer.writerow({**{key: row.get(key, '') for key in ['dataset', 'seed', 'unit_id', 'label', 'group', 'pair_id']},
                                 'policy': policy, 'prediction': 'UNKNOWN' if value is None else value})
            count += 1
    if not count:
        raise ValueError('No rows match the selected neural method')
    names = json.loads((Path(__file__).resolve().parents[1] / 'experiment_names.json').read_text())['prediction_method_names']
    print(json.dumps({'input_rows': count, 'output_rows': count * 5, 'method': args.method,
                      'paper_name': names.get(args.method, args.method)}))


if __name__ == '__main__':
    main()
