"""Run the exact arithmetic channel on frozen graph inputs, then load score targets."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from exact_channel import evaluate_graph, numeric_features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'external_eval/tifs_runs/tifs_mechanism_20261019'
FIELDS = ('node_count', 'ops', 'parents', 'bits', 'known')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    protocol_hash = sha(HERE/'protocol.json')
    predictions, sources, unknowns = {}, [], Counter()
    with (BASE/'diagnostics/ledger.csv').open() as f:
        ledger = list(csv.DictReader(f))
    for entry in ledger:
        if entry['status'] != 'PASS':
            continue
        result_path = Path(entry['result_path'])
        assert sha(result_path) == entry['result_sha256']
        result = json.loads(result_path.read_text())
        tensor_path = result_path.parent/'tensors.npz'
        assert sha(tensor_path) == result['tensor_sha256']
        sources.append({'result_path':str(result_path), 'result_sha256':entry['result_sha256'],
                        'tensor_path':str(tensor_path), 'tensor_sha256':result['tensor_sha256']})
        with np.load(tensor_path, allow_pickle=False) as z:
            for sl in result['slices']:
                assert Path(sl['tensor_cache']).resolve() == tensor_path.resolve()
                x = {k:z[k][int(sl['tensor_row'])] for k in FIELDS}
                sid = sl['sample_id']
                assert sid not in predictions
                sync, topo = evaluate_graph(x, 4), evaluate_graph(x)
                predictions[sid] = (sync, topo)
                for i, state in enumerate(topo):
                    if int(x['ops'][i]) in (2,3,4) and state.value is None:
                        unknowns[state.reason] += 1
    # Persist predictions before opening any historical reference-value rows.
    pred_path = HERE/'exact_channel_predictions.jsonl'
    with pred_path.open('w') as f:
        for sid, (sync, topo) in sorted(predictions.items()):
            f.write(json.dumps({'sample_id':sid,
                                'four_round_values':[s.value for s in sync],
                                'topological_values':[s.value for s in topo],
                                'topological_reasons':[s.reason for s in topo]})+'\n')
    # Additional exact features are available for a future trained fusion model.
    # They are not injected into, or substituted for, the historical detector Q.
    ids = sorted(predictions)
    maximum = max(len(predictions[sid][1]) for sid in ids)
    features = np.zeros((len(ids), maximum, 33), dtype=np.float32)
    counts = []
    for i, sid in enumerate(ids):
        state = predictions[sid][1]; counts.append(len(state))
        features[i, :len(state)] = numeric_features(state)
    np.savez_compressed(HERE/'exact_features.npz', sample_ids=np.array(ids),
                        node_count=np.array(counts), exact_numeric=features)

    rows, graph_rows, input_receipts, expected_identities = [], [], [], None
    for request_path in sorted((BASE/'predictions').glob('*/request.json')):
        req = json.loads(request_path.read_text())
        if req['variant'] != 'full' or req['dataset'] != 'diagnostic':
            continue
        p = request_path.parent/'numeric_state_observations.csv'
        done = json.loads((request_path.parent/'completion.json').read_text())
        assert sha(p) == done['artifacts'][p.name]
        input_receipts.append({'seed':req['seed'], 'path':str(p), 'sha256':sha(p)})
        with p.open() as f:
            reference = [r for r in csv.DictReader(f) if r['layer']=='3']
        identities = {(r['sample_id'], int(r['node']), int(r['reference_value'])) for r in reference}
        if expected_identities is None:
            expected_identities = identities
        assert identities == expected_identities
        computation = [r for r in reference if r['literal']=='False']
        literals = [r for r in reference if r['literal']=='True']
        assert len(computation)==7178 and len(literals)==7507
        by_graph = defaultdict(list)
        for r in computation:
            by_graph[r['sample_id']].append(r)
        for arm in ['original_saved_neural_Q','four_round_exact','topological_exact']:
            complete = 0
            for sid, part in by_graph.items():
                if arm == 'original_saved_neural_Q':
                    complete += all(r['integer_exact']=='True' for r in part)
                else:
                    predicted = predictions[sid][0 if arm=='four_round_exact' else 1]
                    complete += all(predicted[int(r['node'])].value is not None and
                                    predicted[int(r['node'])].value==int(r['reference_value']) for r in part)
            graph_rows.append({'seed_alignment':req['seed'], 'arm':arm, 'graphs':len(by_graph),
                               'all_computed_nodes_exact':complete, 'rate':complete/len(by_graph)})
        for kind, rr in [('computed', computation), ('literal', literals)]:
            for grouping in ['all','operation','dependency_depth']:
                buckets = defaultdict(list)
                for r in rr:
                    buckets['ALL' if grouping=='all' else r[grouping]].append(r)
                for group, part in sorted(buckets.items()):
                    for arm in ['original_saved_neural_Q','four_round_exact','topological_exact']:
                        known, correct = 0, 0
                        for r in part:
                            if arm=='original_saved_neural_Q':
                                # Original probe decodes every latent state without abstention.
                                known += 1; correct += r['integer_exact']=='True'
                            else:
                                state = predictions[r['sample_id']][0 if arm=='four_round_exact' else 1][int(r['node'])]
                                known += state.value is not None
                                correct += state.value is not None and state.value==int(r['reference_value'])
                        rows.append({'seed_alignment':req['seed'],'kind':kind,'grouping':grouping,'group':group,
                                     'arm':arm,'total':len(part),'returned_values':known,'exact_correct':correct,
                                     'coverage':known/len(part),'exact_over_all':correct/len(part),
                                     'exact_given_returned':correct/known if known else None})
    with (HERE/'exact_channel_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (HERE/'whole_graph_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(graph_rows[0]));w.writeheader();w.writerows(graph_rows)
    summary={'status':'COMPLETE','protocol_sha256':protocol_hash,
             'input_graphs':len(predictions),'reference_computed_node_instances':7178,
             'all_computation_nodes_without_exact_value':dict(unknowns),
             'results':[r for r in rows if r['kind']=='computed' and r['grouping']=='all'],
             'whole_graph_results':graph_rows,
             'literal_results':[r for r in rows if r['kind']=='literal' and r['grouping']=='all'],
             'input_receipts':input_receipts,'source_receipts':sources,
             'files_sha256':{name:sha(HERE/name) for name in ['exact_channel.py','evaluate_exact_channel.py','exact_channel_predictions.jsonl','exact_features.npz','exact_channel_metrics.csv','whole_graph_metrics.csv','kernel_verification.json']},
             'interpretation':'Exact arithmetic component evaluated against supplied-graph references; not C execution, full-model retraining, blind test, or detector improvement.'}
    (HERE/'exact_channel_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ['status','input_graphs','all_computation_nodes_without_exact_value','results']}))


if __name__=='__main__':
    main()
