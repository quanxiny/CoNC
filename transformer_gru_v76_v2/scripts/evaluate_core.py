"""Evaluate CoNC and retained controls on the historical family split."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from portable_runtime import ROOT, configure_imports, read, sha, check_output

MODELS = ['conc', 'full', 'vd', 'source_backbone', 'no_refiner', 'residual_refiner',
          'lexical_context', 'numerical_structure', 'joint_representation', 'fixed_encoder_no_dropout']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODELS, default='conc', help='conc is CoNC; full is its historical alias')
    ap.add_argument('--seed', type=int, choices=[11, 29, 47], default=11)
    ap.add_argument('--split', choices=['train', 'dev', 'test'], default='test')
    ap.add_argument('--checkpoint', type=Path, help='A checkpoint and its .pt.sha256 sidecar')
    ap.add_argument('--data-dir', type=Path, default=ROOT/'external/overflowgen/joint_graph_study_v1')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--limit-rows', type=int, help='Explicitly request a partial evaluation')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    requested_model = args.model
    if args.model == 'conc': args.model = 'full'
    if args.batch_size < 1 or (args.limit_rows is not None and args.limit_rows < 1):
        ap.error('batch-size and limit-rows must be positive')
    configure_imports()
    import numpy as np
    import torch
    from common70 import KEYS
    from model70 import AdaptiveGraph70
    from model71 import FrozenResidual71
    from model75 import FrozenEncoder75
    from model76 import NoRefiner76
    from all_visible52 import AllVisible
    from overflow51.baselines import RiskBridge, LegacyPureGraph, NativeVD
    from overflow51.overflowgen import original_vd_batch
    from overflow51.training import load_own_checkpoint
    from overflow51.metrics import risk_metrics
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_factory = {'full': lambda: AdaptiveGraph70('liquid_transfer_frozen'), 'vd': lambda: RiskBridge(NativeVD('family')),
                     'source_backbone': lambda: RiskBridge(AllVisible()), 'no_refiner': NoRefiner76,
                     'residual_refiner': FrozenResidual71, 'fixed_encoder_no_dropout': FrozenEncoder75,
                     'lexical_context': lambda: RiskBridge(LegacyPureGraph('semantic')),
                     'numerical_structure': lambda: RiskBridge(LegacyPureGraph('numeric')),
                     'joint_representation': lambda: RiskBridge(LegacyPureGraph('joint'))}
    checkpoint = args.checkpoint
    if checkpoint is None:
        base = ROOT/'assets/comparisons/overflowgen'
        if args.model == 'full':
            checkpoint = base/f'adaptive_graph_liquid_v70/runs/liquid_transfer_frozen_s{args.seed}/best.pt'
        elif args.model == 'vd':
            checkpoint = base/f'full_overflowgen_v52/runs/family_vd_s{args.seed}_lr2e-05/best.pt'
        elif args.model == 'source_backbone':
            checkpoint = ROOT/f'assets/initialization/v52_all_visible_s{args.seed}.pt'
        else:
            ap.error('--checkpoint is required for this control')
    cache = {}
    manifest = read(ROOT/'configs/source_data.json')
    keys = ['tokens', 'numbers', 'marks', 'line_count', 'key'] if args.model == 'vd' else KEYS
    for key in keys + ['labels', 'family_split']:
        path = args.data_dir/(key+'.npy')
        if sha(path) != manifest['sha256'][path.name]:
            raise ValueError('Source tensor checksum mismatch: ' + path.name)
        cache[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    rows = np.flatnonzero(cache['family_split'] == {'train': 0, 'dev': 1, 'test': 2}[args.split])
    expected = {'train': 94179, 'dev': 11940, 'test': 11577}[args.split]
    if len(rows) != expected:
        raise ValueError('Unexpected split membership')
    if args.limit_rows is not None:
        rows = rows[:args.limit_rows]
    model = model_factory[args.model]()
    state = load_own_checkpoint(checkpoint)
    model.load_state_dict(state['model'], strict=True)
    del state
    model = model.to(args.device).eval()
    output = check_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    all_labels, all_scores = [], []
    total_loss = 0.
    with gzip.open(output/'predictions.csv.gz', 'wt') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row', 'label', 'score', 'prediction'])
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                selected = rows[start:start+args.batch_size]
                if args.model == 'vd':
                    inputs = {k: v.to(args.device) for k, v in original_vd_batch(cache, selected).items()}
                else:
                    inputs = {k: torch.from_numpy(np.array(cache[k][selected], copy=True)).to(args.device) for k in KEYS}
                    lines, nodes = int(inputs['line_count'].max()), int(inputs['node_count'].max())
                    for key in ['tokens', 'numbers', 'marks']:
                        inputs[key] = inputs[key][:, :lines]
                    inputs['edges'] = inputs['edges'][:, :lines, :lines]
                    for key in ['ops', 'parents', 'node_line', 'condition', 'unit', 'known', 'bits', 'target']:
                        inputs[key] = inputs[key][:, :nodes]
                    inputs['numbers'] = inputs['numbers'].long()
                labels = torch.from_numpy(np.array(cache['labels'][selected], copy=True)).long().to(args.device)
                result = model(inputs)
                if not torch.isfinite(result['logits']).all():
                    raise FloatingPointError('Nonfinite logits')
                total_loss += float(torch.nn.functional.cross_entropy(result['logits'], labels, reduction='sum'))
                scores = result['risk'].cpu().tolist()
                truths = labels.cpu().tolist()
                all_labels.extend(truths)
                all_scores.extend(scores)
                writer.writerows((int(row), y, score, int(score >= .5)) for row, y, score in zip(selected, truths, scores))
    report = {'model': requested_model, 'model_id': args.model, 'seed': args.seed, 'split': args.split, 'rows': len(rows),
              'partial_evaluation': args.limit_rows is not None, 'checkpoint_sha256': sha(checkpoint),
              'native_NLL': total_loss/len(rows), **risk_metrics(np.asarray(all_labels), np.asarray(all_scores)),
              'threshold': 0.5, 'comparison_scope': 'Historical family split, all eligible source slices; not the common static-analysis cohort',
              'prediction_sha256': sha(output/'predictions.csv.gz')}
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
