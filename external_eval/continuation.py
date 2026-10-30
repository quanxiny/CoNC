"""Source-only ordinary, fallback-weighted and class-matched CoNC continuation."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).parent / 'boundary_optimization/script'
sys.path[:0] = [str(ROOT), str(SCRIPT)]
import hybrid_retrain_train as trainer
from hybrid_retrain_core import validate_gate, supervised_ce
from classmatch_control import constants, classmatch_ce
from external_eval.tifs.common import (PACKAGE, CACHE, METADATA, SourceData, atomic_json,
                                       read, sha, digest, stable_id, setup)

ARMS = ('ordinary_continue', 'residual_weighted', 'class_prior_matched')
DATA = Path(os.environ.get('OVERFLOW_DATA_DIR', ROOT / 'data')) / 'hybrid'


def load_model(seed, device='cpu'):
    _, torch = setup(seed)
    from model70 import AdaptiveGraph70
    from overflow51.training import load_own_checkpoint
    from external_eval.mit_numeric_ood import patch_numeric_range
    checkpoint = PACKAGE / f'assets/comparisons/overflowgen/adaptive_graph_liquid_v70/runs/liquid_transfer_frozen_s{seed}/best.pt'
    model = AdaptiveGraph70('liquid_transfer_frozen')
    state = load_own_checkpoint(checkpoint)
    model.load_state_dict(state['model'], strict=True)
    patch_numeric_range(model, torch)
    return model.to(device), {'checkpoint': str(checkpoint.relative_to(ROOT)), 'sha256': sha(checkpoint)}


class GateData:
    def __init__(self, data, cfg):
        self.data = data
        self.book = {}
        self.rows = {}
        manifest = read(DATA / 'manifest.json')
        for name, expected in manifest['files'].items():
            if sha(DATA / name) != expected:
                raise ValueError('Gate or schedule hash mismatch: ' + name)
        with (DATA.parent / 'mechanism/source_manifest.csv').open() as stream:
            inputs = {int(row['row']): row for row in csv.DictReader(stream)}
        for split, expected_count in [('train', 94179), ('val', 11940)]:
            path = DATA / 'gates' / (split + '.jsonl')
            records = [json.loads(line) for line in path.read_text().splitlines() if line]
            expected = {i for i, meta in enumerate(data.metadata) if meta['family_split'] == split}
            if {record['row'] for record in records} != expected or len(records) != expected_count:
                raise ValueError('Gate membership differs from historical family split')
            for record in records:
                row = record['row']
                meta, inp = data.metadata[row], inputs[row]
                if record['unit_id'] != stable_id(meta) or record['unit_id'] != inp['sample_id']:
                    raise ValueError('Gate row identity mismatch')
                consumed = record.get('consumed_input_sha256', record.get('input_sha256'))
                if consumed != inp['input_sha256'] or int(inp['label']) != int(data.a['labels'][row]):
                    raise ValueError('Gate input or source label identity mismatch')
                self.book[row] = {**validate_gate(record), 'unit_id': record['unit_id']}
            self.rows[split] = sorted(expected)
        self.p_R = sum(self.book[i]['R'] for i in self.rows['train']) / len(self.rows['train'])
        self.mu = 1 + self.p_R
        if abs(self.mu - cfg['mu']) > 1e-15:
            raise ValueError('Residual normalization changed')


def class_constants(gates):
    total = [0, 0]
    residual = [0, 0]
    for row in gates.rows['train']:
        label = int(gates.data.a['labels'][row])
        total[label] += 1
        residual[label] += int(gates.book[row]['R'])
    return constants(total, residual)


def prepare(out):
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Continuation outputs must stay in this repository outputs directory')
    out.mkdir(parents=True, exist_ok=True)
    manifest = read(DATA / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha(DATA / name) != expected:
            raise ValueError('Data artifact mismatch: ' + name)
        if name.startswith(('gates/', 'schedules/')):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != expected:
                raise ValueError('Existing continuation artifact changed')
            if not target.exists():
                shutil.copyfile(DATA / name, target)
    cfg = read(DATA / 'config.json')
    cfg['arms'] = list(ARMS)
    gates = GateData(SourceData(), cfg)
    cfg['classmatch_constants'] = class_constants(gates)
    atomic_json(out / 'config.json', cfg, immutable=True)
    bases = {}
    for seed in cfg['seeds']:
        checkpoint = PACKAGE / f'assets/comparisons/overflowgen/adaptive_graph_liquid_v70/runs/liquid_transfer_frozen_s{seed}/best.pt'
        bases[str(seed)] = {'checkpoint': str(checkpoint.relative_to(ROOT)), 'sha256': sha(checkpoint)}
    model, _ = load_model(cfg['seeds'][0])
    mask = digest(trainer.parameter_manifest(model))
    del model
    files = [Path(__file__), SCRIPT / 'hybrid_retrain_train.py', SCRIPT / 'hybrid_retrain_core.py',
             SCRIPT / 'hybrid_retrain_state.py', SCRIPT / 'classmatch_control.py', DATA / 'manifest.json',
             DATA.parent / 'mechanism/source_manifest.csv', METADATA]
    source_manifest = read(PACKAGE / 'configs/source_data.json')
    for name, expected in source_manifest['sha256'].items():
        path = CACHE / name
        if sha(path) != expected:
            raise ValueError('Source tensor checksum mismatch: ' + name)
        files.append(path)
    freeze = {'schema': 'portable-continuation-v1', 'base_models': bases,
              'parameter_contract': {'mask_sha256': mask}, 'config_sha256': sha(out / 'config.json'),
              'files': {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha(path) for path in files}}
    atomic_json(out / 'freeze.json', freeze, immutable=True)
    return cfg, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS + ('fallback_weighted',), default='ordinary_continue',
                        help='fallback_weighted is the paper name for historical residual_weighted')
    parser.add_argument('--seed', type=int, choices=[11, 29, 47], default=11)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/hybrid_continuation')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two TRAIN examples, forward/backward only; no checkpoint')
    args = parser.parse_args()
    if args.arm == 'fallback_weighted': args.arm = 'residual_weighted'
    paper_arm = {'ordinary_continue': 'ordinary continuation',
                 'residual_weighted': 'fallback-weighted continuation',
                 'class_prior_matched': 'class-matched continuation'}[args.arm]
    cfg, gates = prepare(args.out)
    if args.prepare_only:
        print(json.dumps({'status': 'READY', 'paper_arm': paper_arm, 'arm': args.arm,
                          'directory': str(args.out), 'mu': gates.mu}))
        return
    c = cfg['classmatch_constants']
    def loss(logits, labels, residual, arm, mu):
        if arm == 'class_prior_matched':
            return classmatch_ce(logits, labels, residual, arm, mu, c['class_weights'])
        return supervised_ce(logits, labels, residual, arm, mu)
    if args.smoke:
        _, torch = setup(args.seed)
        model, _ = load_model(args.seed, 'cpu')
        candidates = gates.rows['train'][::max(1, len(gates.rows['train']) // 128)][:128]
        measured = []
        with torch.no_grad():
            for start in range(0, len(candidates), 16):
                selected = candidates[start:start + 16]
                xx, yy = gates.data.batch(selected, 'cpu')
                errors = torch.nn.functional.cross_entropy(model(xx)['logits'], yy, reduction='none')
                measured.extend(zip(errors.tolist(), selected))
        rows = [row for _, row in sorted(measured, reverse=True)[:2]]
        x, y = gates.data.batch(rows, 'cpu')
        residual = torch.tensor([gates.book[i]['R'] for i in rows], dtype=torch.bool)
        value, _ = loss(model(x)['logits'], y, residual, args.arm, gates.mu)
        value.backward()
        if not torch.isfinite(value) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite continuation smoke result')
        print(json.dumps({'status': 'PASS', 'paper_arm': paper_arm, 'arm': args.arm, 'examples': 2,
                          'screened_train_examples': len(candidates), 'loss': float(value.detach()), 'weights_saved': False}))
        return
    trainer.D = args.out.resolve()
    trainer.ARMS = ARMS
    trainer.GateData = GateData
    trainer.load_model = load_model
    trainer.supervised_ce = loss
    if args.arm == 'class_prior_matched':
        original_audit = trainer.gate_label_audit
        def audit(gate_data):
            result = original_audit(gate_data)
            for split in ['train', 'val']:
                item = result[split]
                negative = item['R0_label0'] + item['R1_label0']
                positive = item['R0_label1'] + item['R1_label1']
                item['residual_reference_weighted_positive_fraction'] = item['weighted_positive_fraction']
                item['weighted_positive_fraction'] = positive * c['class_weights'][1] / (negative * c['class_weights'][0] + positive * c['class_weights'][1])
                item['weight_definition'] = 'Class constants fitted on full source TRAIN only'
            result['classmatch_constants'] = c
            return result
        trainer.gate_label_audit = audit
    trainer.run(args.arm, args.seed, args.device)


if __name__ == '__main__':
    main()
