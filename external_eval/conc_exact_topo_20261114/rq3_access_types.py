"""Reanalyse frozen predictions using the paper's existing RQ3 access groups.

No training, threshold selection, source execution, or new source analysis.
Also run the v1.1 arithmetic implementation on the historical numerical probe.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from . import campaign as c
from .model import exact_states

OUT = c.OUT / 'rq3_access_types'
OLD = c.ROOT / 'external_eval/rq34_source_20261107'
PROBE = c.ROOT / 'external_eval/numeric_precision_20261114'
SEEDS = (11, 29, 47)
ARMS = ('original', 'literal_only', 'exact_topo')
GROUPS = ('simple_array_read', 'simple_array_write', 'simple_pointer_write',
          'call_memcpy', 'call_memmove', 'call_strcpy', 'call_strncpy',
          'call_memset', 'other_or_multiple')
NAMES = ('Array read', 'Array write', 'Pointer write', 'memcpy', 'memmove',
         'strcpy', 'strncpy', 'memset', 'Other / multiple')
CHINESE = ('数组读取', '数组写入', '指针写入', 'memcpy', 'memmove',
           'strcpy', 'strncpy', 'memset', '其他或多种访问')


def csv_read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metrics(y, p):
    tn, fp, fn, tp = map(int, confusion_matrix(y, p, labels=[0, 1]).ravel())
    def ratio(a, b):
        return a / b if b else None
    # Positive-class F1 is uninformative in an all-negative group.
    f1 = ratio(2 * tp, 2 * tp + fp + fn) if tp + fn else None
    if f1 is not None:
        assert abs(f1 - f1_score(y, p)) < 1e-12
    return {'n': len(y), 'positive': tp + fn, 'negative': tn + fp,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'f1': f1,
            'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'specificity': ratio(tn, tn + fp), 'accuracy': ratio(tp + tn, len(y))}


def strict_probe():
    receipt = c.read(PROBE / 'exact_channel_results.json')
    predicted = {}
    for source in receipt['source_receipts']:
        path = Path(source['result_path']); tensor = Path(source['tensor_path'])
        assert c.sha(path) == source['result_sha256']
        assert c.sha(tensor) == source['tensor_sha256']
        result = c.read(path)
        with np.load(tensor, allow_pickle=False) as z:
            x = {k: torch.from_numpy(np.array(z[k], copy=True))
                 for k in ('ops', 'parents', 'bits', 'known', 'node_count')}
        values, known = exact_states(x)
        for row in result['slices']:
            sid = row['sample_id']; idx = int(row['tensor_row'])
            assert sid not in predicted
            count = int(x['node_count'][idx])
            predicted[sid] = {'sample_id': sid,
                              'values': values[idx, :count].tolist(),
                              'known': known[idx, :count].tolist()}
    assert len(predicted) == 1698
    # Only open reference values after committing the implementation's output.
    with (OUT / 'strict_probe_predictions.jsonl').open('w') as f:
        for sid in sorted(predicted):
            f.write(json.dumps(predicted[sid]) + '\n')
    scores, identity = [], None
    for ref in receipt['input_receipts']:
        path = Path(ref['path']); assert c.sha(path) == ref['sha256']
        rows = [r for r in csv_read(path) if r['layer'] == '3']
        ids = {(r['sample_id'], r['node'], r['reference_value']) for r in rows}
        if identity is None: identity = ids
        assert identity == ids
        for literal in (False, True):
            part = [r for r in rows if r['literal'] == str(literal)]
            known = correct = original = 0
            by_graph = defaultdict(list)
            for r in part:
                prediction = predicted[r['sample_id']]; node = int(r['node'])
                k = prediction['known'][node]
                ok = k and prediction['values'][node] == int(r['reference_value'])
                known += k; correct += ok; original += r['integer_exact'] == 'True'
                by_graph[r['sample_id']].append(ok)
            scores.append({'seed_alignment': ref['seed'], 'kind': 'literal' if literal else 'computed',
                           'nodes': len(part), 'v11_known': known, 'v11_correct': correct,
                           'original_neural_correct': original,
                           'v11_integer_exactness': correct / len(part),
                           'graphs': len(by_graph), 'graphs_all_correct': sum(all(v) for v in by_graph.values())})
    assert all(r['nodes'] == r['v11_correct'] for r in scores)
    result = {'status': 'PASS', 'model_code_sha256': c.sha(c.HERE / 'model.py'),
              'historical_probe_receipt_sha256': c.sha(PROBE / 'exact_channel_results.json'),
              'deterministic_runs': 1, 'scores': scores,
              'limitation': 'Actual v1.1 arithmetic channel, not neural Q; same supplied-graph reference contract; historical exposed probe.'}
    c.save_json(OUT / 'strict_probe_v11.json', result)
    return result


def paired_interval(y, predictions, families, rng):
    groups, index = np.unique(families, return_inverse=True)
    counts = np.zeros((3, 3, len(groups), 4), dtype=np.int64)
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            p = predictions[(arm, seed)]
            slot = np.where(y == 1, np.where(p == 1, 0, 3), np.where(p == 1, 1, 2))
            np.add.at(counts[ai, si], (index, slot), 1)
    def f1(count):
        tp, fp, tn, fn = np.moveaxis(count, -1, 0)
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)
    boot = []
    for _ in range(40):
        draws = rng.integers(0, len(groups), size=(50, len(groups)))
        multiplicity = np.stack([np.bincount(d, minlength=len(groups)) for d in draws])
        summed = np.einsum('bf,asfc->basc', multiplicity, counts, optimize=True)
        boot.append(f1(summed).mean(axis=2))
    sampled = np.concatenate(boot); point = f1(counts.sum(axis=2)).mean(axis=1)
    intervals = []
    for ai, comparator in enumerate(ARMS[:2]):
        values = sampled[:, 2] - sampled[:, ai]
        valid = values[np.isfinite(values)]
        intervals.append({'comparator': comparator, 'difference': float(point[2] - point[ai]),
                          'ci95_lower': float(np.quantile(valid, .025)),
                          'ci95_upper': float(np.quantile(valid, .975)),
                          'families': len(groups), 'replicates': 2000, 'valid_replicates': len(valid)})
    return intervals


def main():
    c.setup(); c.verify_freeze(); OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'name': 'CoNC-ExactTopo-v1.1 RQ3 saved-prediction stratification',
                'scope': 'Original nine target syntax groups; common11342 primary, full11577 auxiliary',
                'model_selection': 'Reuse frozen selection.json; no training or threshold changes',
                'seeds': list(SEEDS), 'arms': list(ARMS), 'bootstrap': {'unit': 'paired family', 'replicates': 2000, 'seed': 20261114},
                'subgroup_inference': 'Exploratory pointwise intervals, no multiplicity-adjusted significance claims',
                'no_positive_group': 'F1 NA; report false positives and specificity',
                'numeric_check': 'Actual v1.1 exact_states on same historical7178 probe; full TEST coverage stratified from independently verified states',
                'semantics': 'Target syntax, not verified CWE subtypes or exact overflow boundary ground truth'}
    pp = OUT / 'protocol.json'
    if pp.exists(): assert c.read(pp) == protocol
    else: c.save_json(pp, protocol)
    provenance = {}; access_receipt = c.read(OLD / 'access_type_receipt.json')
    for filename in ('target_type_manifest.csv', 'access_type_metrics.csv'):
        assert c.sha(OLD / filename) == access_receipt['outputs'][filename]
    manifest = {r['sample_id']: r for r in csv_read(OLD / 'target_type_manifest.csv')}
    assert len(manifest) == 11577
    old = {(r['target_group'], r['scope'], r['method'], r['seed']): r
           for r in csv_read(OLD / 'access_type_metrics.csv')}
    data_rows = c.read(c.OUT / 'data/test_rows.json'); ids = [r['sample_id'] for r in data_rows]
    assert set(ids) == set(manifest)
    y = np.array([r['label'] for r in data_rows]); groups = np.array([manifest[s]['access_group'] for s in ids])
    families = np.array([manifest[s]['family_id'] for s in ids])
    bags = {}; common = None
    for arm in ARMS:
        for seed in SEEDS:
            path = c.OUT / 'predictions' / f'{arm}_s{seed}.csv'; receipt = c.read(path.with_suffix('.json'))
            assert c.sha(path) == receipt['sha256']
            assert receipt['selection'] == c.sha(c.OUT / 'selection.json')
            records = csv_read(path); assert [r['sample_id'] for r in records] == ids
            for i, r in enumerate(records):
                assert int(r['label']) == y[i] == int(manifest[ids[i]]['label'])
                assert r['group'] == families[i] == data_rows[i]['group']
            mask = np.array([r['common11342'] == 'True' for r in records])
            if common is None: common = mask
            assert np.array_equal(common, mask)
            bags[(arm, seed)] = np.array([int(r['prediction']) for r in records])
            provenance[str(path.relative_to(c.ROOT))] = c.sha(path)
    actual_common = {r['scene_id'] for r in csv_read(c.ROOT / 'paper/results/revision/comparison_common_ids.csv')
                     if r['cohort'] == 'source_common11342'}
    assert {sid for sid, keep in zip(ids, common) if keep} == actual_common
    assert int(common.sum()) == 11342
    verification = c.read(c.OUT / 'verification.json'); assert verification['status'] == 'PASS'
    graph_path = c.OUT / 'data/test.pt'
    assert c.sha(graph_path) == c.read(c.OUT / 'preparation.json')['files']['data/test.pt']
    data = torch.load(graph_path, map_location='cpu', weights_only=True)
    assert data['source_row'].tolist() == [r['source_row'] for r in data_rows]
    valid = torch.arange(data['ops'].shape[1])[None] < data['node_count'][:, None]
    computed = valid & (data['ops'].eq(2) | data['ops'].eq(3) | data['ops'].eq(4))
    numerical_rows, metric_rows, summary, differences = [], [], [], []
    checks = 0; rng = np.random.default_rng(protocol['bootstrap']['seed'])
    for scope, scope_mask in [('common11342', common), ('full11577', np.ones(len(ids), dtype=bool))]:
        for group in ('ALL',) + GROUPS:
            keep = scope_mask & ((groups == group) if group != 'ALL' else True)
            count = int(keep.sum()); positive = int(y[keep].sum()); indices = np.where(keep)[0]
            for arm in ARMS:
                parts = []
                for seed in SEEDS:
                    result = metrics(y[keep], bags[(arm, seed)][keep])
                    row = {'scope': scope, 'access_group': group, 'arm': arm, 'seed': seed, **result}
                    metric_rows.append(row); parts.append(result)
                    if arm == 'original':
                        previous = old[(group, 'common' if scope == 'common11342' else 'own', 'CoNC', str(seed))]
                        assert all(result[k] == int(previous[k]) for k in ('tp', 'fp', 'tn', 'fn'))
                        checks += 1
                row = {'scope': scope, 'access_group': group, 'arm': arm, 'n': count, 'positive': positive}
                for key in ('f1', 'precision', 'recall', 'specificity', 'accuracy', 'fp', 'fn'):
                    values = [p[key] for p in parts]
                    row[key + '_mean'] = float(np.mean(values)) if all(v is not None for v in values) else None
                    row[key + '_sd'] = float(np.std(values, ddof=1)) if all(v is not None for v in values) else None
                summary.append(row)
            num = {'scope': scope, 'access_group': group, 'n': count}
            for op, name in [(None, 'computed'), (2, 'copy'), (3, 'add'), (4, 'sub')]:
                comp = computed[indices] if op is None else (valid & data['ops'].eq(op))[indices]
                total = int(comp.sum()); known = int((comp & data['exact_known'][indices]).sum())
                num[name + '_total'] = total; num[name + '_known'] = known
                num[name + '_coverage'] = known / total if total else None
            num['known_integer_exactness'] = 1.0 if num['computed_known'] else None
            numerical_rows.append(num)
            if scope == 'common11342' and positive:
                predictions = {k: v[keep] for k, v in bags.items()}
                for interval in paired_interval(y[keep], predictions, families[keep], rng):
                    differences.append({'scope': scope, 'access_group': group, **interval})
    for name, rows in [('per_seed.csv', metric_rows), ('summary.csv', summary),
                       ('numeric_coverage.csv', numerical_rows), ('paired_differences.csv', differences)]:
        write_csv(name, rows)
    for p in (OLD / 'target_type_manifest.csv', OLD / 'access_type_metrics.csv', c.OUT / 'selection.json',
              c.OUT / 'verification.json', c.OUT / 'data/test.pt', c.OUT / 'data/test_rows.json',
              c.ROOT / 'paper/manuscript.tex', c.ROOT / 'paper/results/revision/comparison_common_ids.csv'):
        provenance[str(p.relative_to(c.ROOT))] = c.sha(p)
    lookup = {(r['scope'], r['access_group'], r['arm']): r for r in summary}
    comparison = []
    for group in GROUPS + ('ALL',):
        new = lookup[('common11342', group, 'exact_topo')]
        row = {'access_group': group, 'n': new['n'], 'positive': new['positive']}
        for method in ('CoNC', 'VDTransformer', 'DeepWukong', 'DeepDFA', 'Cppcheck', 'Clang', 'Eva'):
            parts = [v for k, v in old.items() if k[:3] == (group, 'common', method)]
            vals = [float(p['f1']) for p in parts if p['f1'] != '']
            row[method + '_f1'] = float(np.mean(vals)) if len(vals) == len(parts) and new['positive'] else None
        row['literal_only_f1'] = lookup[('common11342', group, 'literal_only')]['f1_mean']
        row['exact_topo_f1'] = new['f1_mean']
        row['exact_minus_original_pp'] = 100 * (new['f1_mean'] - row['CoNC_f1']) if new['positive'] else None
        comparison.append(row)
    write_csv('paper_rq3_comparison.csv', comparison)
    probe = strict_probe()
    result = {'status': 'COMPLETE', 'historical_original_confusion_checks': checks,
              'sklearn_confusion_and_f1_checks': len(metric_rows), 'strict_probe_status': probe['status'],
              'common_n': 11342, 'full_n': 11577, 'comparisons': comparison,
              'subgroup_intervals': differences, 'numerical_coverage': numerical_rows,
              'inputs_sha256': provenance, 'script_sha256': c.sha(__file__),
              'protocol_sha256': c.sha(pp),
              'interpretation': 'Retrospective access-syntax strata; pointwise exploratory intervals; no new fitting or C execution.'}
    c.save_json(OUT / 'results.json', result)
    print(json.dumps({'status': 'COMPLETE', 'comparisons': comparison,
                      'subgroup_intervals': differences,
                      'numeric_common': [r for r in numerical_rows if r['scope'] == 'common11342'],
                      'strict_probe': probe['scores']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
