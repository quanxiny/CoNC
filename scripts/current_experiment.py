#!/usr/bin/env python3
"""Evaluate or train the strict-split exact-fusion branch from frozen inputs."""
import argparse
import csv
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def replay():
    for split in ('strict', 'family'):
        for arm in ('original', 'literal_only', 'exact_topo'):
            rows = []
            for seed in (11, 29, 47):
                path = ROOT / 'data/current' / split / f'{arm}_s{seed}.csv.gz'
                with gzip.open(path, 'rt', newline='') as f:
                    predictions = list(csv.DictReader(f))
                if split == 'family':
                    predictions = [r for r in predictions if r['common11342'] == 'True']
                tp = sum(r['label'] == '1' and r['prediction'] == '1' for r in predictions)
                fp = sum(r['label'] == '0' and r['prediction'] == '1' for r in predictions)
                fn = sum(r['label'] == '1' and r['prediction'] == '0' for r in predictions)
                rows.append(2 * tp / (2 * tp + fp + fn))
            print(f'{split:6} {arm:13} n={len(predictions)} F1={statistics.mean(rows)*100:.4f}% +/- {statistics.stdev(rows)*100:.4f}%')
    with gzip.open(ROOT/'data/current/strict/trace.jsonl.gz','rt') as f:
        trace={r['sample_id']:r['D_trace'] for r in map(json.loads,f)}
    with gzip.open(ROOT/'data/current/strict/frama_c_eva.csv.gz','rt',newline='') as f:
        eva={r['sample_id']:r['prediction'] for r in csv.DictReader(f)}
    hybrid=[]; static=[]
    for seed in (11,29,47):
        with gzip.open(ROOT/f'data/current/strict/exact_topo_s{seed}.csv.gz','rt',newline='') as f:
            predictions=list(csv.DictReader(f))
        counts=[[0,0,0],[0,0,0]]
        for r in predictions:
            sid=r['sample_id']; t=trace[sid]; e=eva.get(sid)
            fixed=t if t in (0,1) else 1 if e=='1' else None
            for i,pred in enumerate((int(r['prediction']) if fixed is None else fixed, 0 if fixed is None else fixed)):
                label=int(r['label']);counts[i][0]+=pred==1 and label==1
                counts[i][1]+=pred==1 and label==0;counts[i][2]+=pred==0 and label==1
        for dest,(tp,fp,fn) in zip((hybrid,static),counts): dest.append(2*tp/(2*tp+fp+fn))
    for name,values in (('CoNC+Trace+Eva',hybrid),('Trace+Eva+0',static)):
        print(f'strict {name:13} n={len(predictions)} F1={statistics.mean(values)*100:.4f}% +/- {statistics.stdev(values)*100:.4f}%')


def evaluate(args):
    import torch
    from external_eval.conc_exact_topo_20261114.model import ExactFusion, GRAPH_FIELDS, exact_states
    torch.set_num_threads(4)
    directory = ROOT / 'data/current_assets/strict'
    data = torch.load(directory/'data/test.pt', map_location='cpu', weights_only=True)
    cache = torch.load(directory/f'context/test_s{args.seed}.pt', map_location='cpu', weights_only=True)
    assert torch.equal(data['source_row'], cache['source_row'])
    model = ExactFusion('literal_only' if args.arm == 'literal_only' else 'exact_topo')
    model.load_state_dict(torch.load(directory/f'heads/{args.arm}_s{args.seed}.pt', map_location='cpu', weights_only=True)['head'])
    model.to(args.device).eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(data['labels']), args.batch_size):
            ix = slice(start, start + args.batch_size)
            x = {k:data[k][ix].to(args.device) for k in GRAPH_FIELDS}
            if args.arm == 'literal_only':
                values, known = exact_states(x, False)
            else:
                prefix = 'four' if args.arm == 'four_round' else 'exact'
                values = data[prefix+'_values'][ix].to(args.device)
                known = data[prefix+'_known'][ix].to(args.device)
            out = model(x, cache['context'][ix].to(args.device), cache['logits'][ix].to(args.device), values, known)
            probabilities.extend(out['logits'].softmax(-1)[:, 1].cpu().tolist())
    with gzip.open(ROOT/f'data/current/strict/{args.arm}_s{args.seed}.csv.gz', 'rt', newline='') as f:
        reference = list(csv.DictReader(f))
    assert len(reference) == len(probabilities)
    mismatches = sum((p >= .5) != bool(int(r['prediction'])) for p, r in zip(probabilities, reference))
    max_difference = max(abs(p - float(r['risk'])) for p, r in zip(probabilities, reference))
    print(json.dumps({'rows':len(reference),'arm':args.arm,'seed':args.seed,'class_mismatches':mismatches,'max_probability_difference':max_difference}, indent=2))
    if mismatches: raise SystemExit('Predicted classes differ from the saved reference')


def live(args):
    import torch
    from external_eval.tifs import common
    from external_eval.tifs.models import make_model
    from external_eval.conc_exact_topo_20261114.model import CoNCExactTopo
    torch.set_num_threads(4)
    asset = ROOT/'data/current_assets/strict'
    package = Path(args.data_root).resolve()/'transformer_gru_v76_v2'
    common.CACHE = package/'external/overflowgen/joint_graph_study_v1'
    common.METADATA = package/'assets/comparisons/overflowgen/overflow_interfaces_v51/output/row_metadata.jsonl.gz'
    base = make_model('full', 2, args.seed)
    state = torch.load(asset/f'backbones/full_s{args.seed}_stage2.pt', map_location='cpu', weights_only=True)
    base.load_state_dict(state['model'], strict=True)
    model = CoNCExactTopo(base).to(args.device).eval()
    head = torch.load(asset/f'heads/exact_topo_s{args.seed}.pt', map_location='cpu', weights_only=True)
    model.adapter.load_state_dict(head['head'], strict=True)
    data = torch.load(asset/'data/test.pt', map_location='cpu', weights_only=True)
    source = common.SourceData()
    with gzip.open(ROOT/f'data/current/strict/exact_topo_s{args.seed}.csv.gz', 'rt', newline='') as f:
        reference = list(csv.DictReader(f))[:args.limit]
    mismatch = 0
    max_difference = 0.0
    with torch.inference_mode():
        for start in range(0, len(reference), args.batch_size):
            part = reference[start:start+args.batch_size]
            indices = [int(r['source_row']) for r in part]
            x, labels = source.batch(indices, args.device)
            out = model(x)
            scores = out['risk'].cpu().tolist()
            mismatch += sum((v >= .5) != bool(int(r['prediction'])) for v,r in zip(scores,part))
            max_difference = max(max_difference, *(abs(v-float(r['risk'])) for v,r in zip(scores,part)))
    print(json.dumps({'rows':len(reference),'seed':args.seed,'class_mismatches':mismatch,'max_probability_difference':max_difference},indent=2))
    if mismatch: raise SystemExit('Predicted classes differ from saved reference')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('replay', help='Recompute F1 from saved predictions (standard library only)')
    q = sub.add_parser('evaluate', help='Recompute strict fusion predictions from cached graph and backbone context')
    q.add_argument('--seed', type=int, choices=(11,29,47), default=11)
    q.add_argument('--arm', choices=('exact_topo','literal_only','four_round'), default='exact_topo')
    q.add_argument('--device', default='cpu')
    q.add_argument('--batch-size', type=int, default=512)
    l = sub.add_parser('live', help='Run full strict model on encoded inputs')
    l.add_argument('--seed', type=int, choices=(11,29,47), default=11)
    l.add_argument('--device', default='cpu')
    l.add_argument('--batch-size', type=int, default=16)
    l.add_argument('--limit', type=int, default=64)
    l.add_argument('--data-root', default=str(ROOT), help='Root containing extracted data attachment')
    a = p.parse_args()
    {'replay': replay, 'evaluate': lambda: evaluate(a), 'live': lambda: live(a)}[a.command]()

if __name__ == '__main__':
    main()
