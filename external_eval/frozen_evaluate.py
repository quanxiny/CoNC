#!/usr/bin/env python3
"""Frozen six-checkpoint evaluation. Saved slice scores are the sole statistics input.

No training, parser, replacement model, calibration, or source-code access occurs here.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import fcntl
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'transformer_gru_v76_v2'
PROTOCOL = HERE / 'frozen_evaluation_protocol.json'
KEYS = ['tokens', 'numbers', 'marks', 'line_count', 'key', 'ops', 'parents', 'node_line',
        'condition', 'unit', 'known', 'bits', 'target', 'node_count', 'edges']
LINE_KEYS = ['tokens', 'numbers', 'marks']
NODE_KEYS = ['ops', 'parents', 'node_line', 'condition', 'unit', 'known', 'bits', 'target']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def signed_read(path, field):
    value = json.loads(Path(path).read_text())
    expected = value.pop(field, None)
    if digest(value) != expected:
        raise ValueError('signed artifact hash mismatch: ' + str(path))
    return value


def atomic_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def checked_child(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('tensor cache path escapes preparation directory')
    return path


def load_prepared(directory):
    directory = Path(directory).resolve()
    receipt = signed_read(directory / 'preparation_receipt.json', 'result_sha256')
    if receipt['status'] != 'COMPLETE':
        raise ValueError('preparation must finish its full registered unit inventory')
    for filename in ['prepared_scenes.jsonl', 'slices.jsonl']:
        if sha(directory / filename) != receipt['views'][filename]:
            raise ValueError('prepared view changed')
    scenes, slices = jsonl(directory / 'prepared_scenes.jsonl'), jsonl(directory / 'slices.jsonl')
    if len({s['scene_id'] for s in scenes}) != len(scenes) or len({s['slice_id'] for s in slices}) != len(slices):
        raise ValueError('duplicate scene/slice identity')
    artifacts = {}
    for unit in receipt['completed'].values():
        for path, expected in unit.get('artifact_sha256', {}).items():
            p = checked_child(directory, path)
            if sha(p) != expected:
                raise ValueError('prepared tensor hash mismatch')
            artifacts[str(p)] = expected
    by_scene = {s['scene_id']: s for s in scenes}
    for row in slices:
        path = checked_child(directory, row['tensor_cache'])
        if str(path) not in artifacts or row['status'] != 'encoded' or len(row['owner_scene_ids']) != 1:
            raise ValueError('tensor row is not uniquely owned and hash verified')
        if row['owner_scene_ids'][0] not in by_scene:
            raise ValueError('slice owner absent from scene manifest')
        row['tensor_cache'] = str(path)
    return {'directory': str(directory), 'dataset': receipt['request']['dataset'], 'scenes': scenes,
            'slices': sorted(slices, key=lambda x: x['slice_id']),
            'identity': {'preparation_receipt_sha256': sha(directory / 'preparation_receipt.json'),
                         'scene_manifest_sha256': receipt['views']['prepared_scenes.jsonl'],
                         'slice_manifest_sha256': receipt['views']['slices.jsonl'],
                         'tensor_artifacts': artifacts}}


class ScoreStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS scores (slice_id TEXT PRIMARY KEY, status TEXT NOT NULL, risk REAL, logit0 REAL, logit1 REAL, reason TEXT)')
        self.db.commit()

    def rows(self):
        return [dict(zip(['slice_id', 'status', 'risk', 'logit0', 'logit1', 'reason'], row))
                for row in self.db.execute('SELECT * FROM scores ORDER BY slice_id')]

    def insert(self, records):
        with self.db:
            self.db.executemany('INSERT INTO scores VALUES (?,?,?,?,?,?)',
                [(r['slice_id'], r['status'], r.get('risk'), r.get('logit0'), r.get('logit1'), r.get('reason')) for r in records])

    def close(self):
        self.db.close()


def scene_predictions(scenes, score_rows):
    scores = {r['slice_id']: r for r in score_rows}
    output = []
    for scene in sorted(scenes, key=lambda s: s['scene_id']):
        inventory = scene.get('slice_inventory') or []
        observed = [scores[s['slice_id']]['risk'] for s in inventory
                    if s.get('slice_id') in scores and scores[s['slice_id']]['status'] == 'PASS']
        complete = bool(inventory) and bool(scene.get('slice_inventory_complete')) and bool(scene.get('frontend_processable'))
        complete = complete and all(s.get('status') == 'encoded' and scores.get(s['slice_id'], {}).get('status') == 'PASS' for s in inventory)
        maximum = max(observed, default=None)
        decision = 1 if maximum is not None and maximum >= .5 else 0 if complete else None
        output.append({'scene_id': scene['scene_id'], 'label': scene.get('label'),
                       'record_kind': scene.get('record_kind', 'scene'),
                       'family_id': scene.get('family_id'), 'template_group': scene.get('template_group'),
                       'variant': scene.get('variant'), 'task_applicability': scene.get('task_applicability'),
                       'preparation_status': scene.get('preparation_status'),
                       'manifest_status': scene.get('manifest_status', scene.get('status')), 'complete': bool(complete),
                       'slice_inventory_complete': bool(scene.get('slice_inventory_complete')),
                       'expected_slice_count': len(inventory), 'observed_slice_count': len(observed),
                       'max_observed_risk': maximum, 'prediction': decision,
                       'primary_prediction': decision if complete else None,
                       'decision_scope': 'complete_scene' if complete else 'partial_diagnostic_only'})
    return output


def write_csv(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['scene_id'])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def initialize_run(output, request, prepared):
    run_id = digest(request)
    path = output / run_id
    path.mkdir(parents=True, exist_ok=True)
    request_path = path / 'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('run ID request collision')
    else:
        if any(path.iterdir()):
            raise ValueError('untracked run output exists')
        atomic_json(request_path, request)
    scores = ScoreStore(path / 'slice_scores.sqlite3')
    rows = scores.rows()
    wanted = {s['slice_id'] for s in prepared['slices']}
    if not {r['slice_id'] for r in rows}.issubset(wanted):
        raise ValueError('score database has unknown slice IDs')
    receipt_path = path / 'run_receipt.json'
    if receipt_path.exists():
        receipt = signed_read(receipt_path, 'receipt_sha256')
        if receipt.get('run_id') != run_id:
            raise ValueError('run receipt identity mismatch')
        if receipt['status'] == 'COMPLETE':
            if receipt['slice_scores_sha256'] != digest(rows) or len(rows) != len(wanted):
                raise ValueError('completed score ledger changed')
            if sha(path / 'scene_predictions.csv') != receipt['scene_predictions_sha256']:
                raise ValueError('completed scene predictions changed')
            return {'run_id': run_id, 'path': path, 'store': scores, 'skip': True, 'prepared': prepared, 'request': request}
    return {'run_id': run_id, 'path': path, 'store': scores, 'skip': False, 'prepared': prepared, 'request': request}


def receipt(run, status):
    rows = run['store'].rows()
    value = {'status': status, 'run_id': run['run_id'], 'request': run['request'],
             'slice_scores_sha256': digest(rows), 'committed_slices': len(rows),
             'successful_slices': sum(r['status'] == 'PASS' for r in rows),
             'failed_slices': sum(r['status'] != 'PASS' for r in rows), 'training': False,
             'threshold': .5, 'aggregation': 'scene max; partial positives diagnostic only'}
    if status == 'COMPLETE':
        predictions = scene_predictions(run['prepared']['scenes'], rows)
        write_csv(run['path'] / 'scene_predictions.csv', predictions)
        value['scene_predictions_sha256'] = sha(run['path'] / 'scene_predictions.csv')
        value['complete_scenes'] = sum(p['complete'] for p in predictions)
    value['receipt_sha256'] = digest(value)
    atomic_json(run['path'] / 'run_receipt.json', value)


def tensor_batch(records, device, np, torch):
    tensors = {key: [] for key in KEYS}
    groups = {}
    for record in records:
        path = record['tensor_cache']
        if path not in groups:
            groups[path] = np.load(path, allow_pickle=False)
    try:
        for record in records:
            arrays = groups[record['tensor_cache']]
            if set(arrays.files) != set(KEYS):
                raise ValueError('tensor schema differs from original 15-field input')
            for key in KEYS:
                tensors[key].append(arrays[key][record['tensor_row']])
        x = {key: torch.from_numpy(np.stack(values)).to(device) for key, values in tensors.items()}
    finally:
        for arrays in groups.values():
            arrays.close()
    L, N = int(x['line_count'].max()), int(x['node_count'].max())
    if not (0 < L <= 12 and 0 < N <= 64):
        raise ValueError('input caps violated')
    for key in LINE_KEYS:
        x[key] = x[key][:, :L]
    for key in NODE_KEYS:
        x[key] = x[key][:, :N]
    x['edges'] = x['edges'][:, :L, :L]
    x['numbers'] = x['numbers'].long()  # Exact original SelectedData.batch cast.
    return x


def select_device(requested, torch):
    if not torch.cuda.is_available():
        raise ValueError('CUDA A100 is required for this frozen execution protocol')
    if requested == 'auto':
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            index = next((i for i in range(torch.cuda.device_count())
                          if 'A100' in torch.cuda.get_device_name(i) and torch.cuda.mem_get_info(i)[0] / torch.cuda.mem_get_info(i)[1] >= .95), None)
        else:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                    check=True, text=True, capture_output=True)
            choices = [line.split(',') for line in result.stdout.splitlines()]
            index = next((int(row[0]) for row in choices if 'A100' in row[1] and int(row[2]) < 1024 and int(row[3]) == 0), None)
        if index is None:
            raise ValueError('no idle A100 available')
        requested = 'cuda:' + str(index)
    device = torch.device(requested)
    if device.type != 'cuda' or 'A100' not in torch.cuda.get_device_name(device):
        raise ValueError('selected device is not an A100')
    free, total = torch.cuda.mem_get_info(device)
    if free / total < .95:
        raise ValueError('selected A100 is not sufficiently idle')
    return device


def own_models(np, torch):
    import asset_smoke
    contract = json.loads((ROOT / 'configs/MIGRATION_CONTRACT_v2.json').read_text())['sha256']
    hashes = {}
    for relative in asset_smoke.SOURCES:
        actual = sha(ROOT / relative)
        if actual != contract[relative]:
            raise ValueError('frozen model dependency changed: ' + relative)
        hashes[relative] = actual
    sources = json.loads((ROOT / 'configs/portable_sources.json').read_text())
    checkpoints = []
    for method, section, key in [('v76', 'trained_v76', 'sha256'), ('v52', 'transfer_sources', 'checkpoint_sha256')]:
        for seed in [11, 29, 47]:
            item = sources[section][str(seed)]
            actual = sha(ROOT / item['checkpoint'])
            if actual != item[key]:
                raise ValueError('trained checkpoint hash mismatch')
            checkpoints.append({'method': method, 'seed': seed, 'metadata': item,
                                'checkpoint_sha256': actual, 'checkpoint': item['checkpoint']})
    sys.path.insert(0, str(ROOT / 'scripts'))
    from portable_runtime import configure_imports
    configure_imports()
    from model76 import NoRefiner76
    from all_visible52 import AllVisible
    from overflow51.baselines import RiskBridge
    def load(checkpoint, device):
        safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype('uint32'))]
        with torch.serialization.safe_globals(safe):
            state = torch.load(ROOT / checkpoint['checkpoint'], map_location='cpu', weights_only=True)
        if not isinstance(state.get('model'), dict):
            raise ValueError('trained model state missing')
        if state.get('config', {}).get('seed') != checkpoint['seed']:
            raise ValueError('checkpoint seed mismatch')
        if checkpoint['method'] == 'v52' and state.get('manifest_sha256') != checkpoint['metadata']['data_manifest_sha256']:
            raise ValueError('source training manifest mismatch')
        model = NoRefiner76() if checkpoint['method'] == 'v76' else RiskBridge(AllVisible())
        model.load_state_dict(state['model'], strict=True)
        model = model.float().to(device).eval()
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('model is not FP32')
        return model
    return checkpoints, hashes, load


def infer_run(run, model, device, np, torch):
    completed = {r['slice_id'] for r in run['store'].rows()}
    pending = [s for s in run['prepared']['slices'] if s['slice_id'] not in completed]
    receipt(run, 'RUNNING')
    for start in range(0, len(pending), 64):
        batch = pending[start:start + 64]
        x = tensor_batch(batch, device, np, torch)
        with torch.inference_mode():
            out = model(x)
        logits = out['logits']
        if logits.shape != (len(batch), 2):
            raise ValueError('invalid frozen logits shape')
        risks = logits.softmax(-1)[:, 1]
        if 'risk' in out and not torch.allclose(out['risk'], risks, equal_nan=True):
            raise ValueError('original risk readout disagrees with class-1 softmax')
        logits, risks = logits.cpu(), risks.cpu()
        records = []
        for item, pair, risk in zip(batch, logits, risks):
            valid = bool(torch.isfinite(pair).all() and torch.isfinite(risk))
            records.append({'slice_id': item['slice_id'], 'status': 'PASS' if valid else 'REJECTED',
                            'risk': float(risk) if valid else None,
                            'logit0': float(pair[0]) if valid else None, 'logit1': float(pair[1]) if valid else None,
                            'reason': None if valid else 'nonfinite_model_output'})
        run['store'].insert(records)
        receipt(run, 'RUNNING')
    receipt(run, 'COMPLETE')


def self_test():
    scene = {'scene_id': 'x', 'label': 1, 'record_kind': 'scene', 'task_applicability': 'target',
             'slice_inventory_complete': True, 'frontend_processable': True,
             'slice_inventory': [{'slice_id': 'a', 'status': 'encoded'}, {'slice_id': 'b', 'status': 'encoded'}]}
    rows = [{'slice_id': 'a', 'status': 'PASS', 'risk': .6, 'logit0': 0., 'logit1': 1., 'reason': None}]
    pred = scene_predictions([scene], rows)[0]
    assert pred['prediction'] == 1 and not pred['complete'] and pred['primary_prediction'] is None
    assert scene_predictions([scene], [dict(rows[0], risk=.2)])[0]['prediction'] is None
    assert scene_predictions([dict(scene, slice_inventory=[])], [])[0]['prediction'] is None
    with tempfile.TemporaryDirectory(prefix='score-handfixture-') as temp:
        path = Path(temp)
        prepared = {'scenes': [scene], 'slices': [{'slice_id': 'a'}, {'slice_id': 'b'}]}
        request = {'method': 'fixture', 'seed': 11, 'input': 'handwritten'}
        run = initialize_run(path, request, prepared)
        run['store'].insert(rows)
        receipt(run, 'RUNNING')
        run['store'].close()
        resumed = initialize_run(path, request, prepared)
        assert len(resumed['store'].rows()) == 1 and not resumed['skip']
        resumed['store'].insert([dict(rows[0], slice_id='b', risk=.3)])
        receipt(resumed, 'COMPLETE')
        resumed['store'].close()
        again = initialize_run(path, request, prepared)
        assert again['skip'] and len(again['store'].rows()) == 2
        again['store'].close()
    print(json.dumps({'status': 'PASS', 'scope': 'handwritten aggregation and SQLite recovery only', 'model_calls': 0, 'external_source_reads': 0, 'script_sha256': sha(__file__)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--prepared', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.prepared or not args.output:
        parser.error('--prepared and --output are required')
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['threshold'] != .5 or protocol['seeds'] != [11, 29, 47] or protocol['batch_size'] != 64:
        raise ValueError('frozen protocol settings changed')
    inference_protocol = {k: protocol[k] for k in ['methods', 'seeds', 'threshold', 'comparison', 'dtype', 'batch_size', 'tf32',
        'device_family', 'primary_population', 'partial_prediction', 'source_policy', 'checkpoint_policy']}
    prepared = [load_prepared(p) for p in args.prepared]
    if len({p['dataset'] for p in prepared}) != len(prepared):
        raise ValueError('duplicate prepared dataset')
    checkpoints, dependency_hashes, load_model = own_models(np, torch)
    # CUDA is only selected if actual missing slice scores require a model call.
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    device = None
    with (args.output / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for checkpoint in checkpoints:
            runs = []
            model = None
            try:
                for dataset in prepared:
                    request = {'method': checkpoint['method'], 'seed': checkpoint['seed'], 'dataset': dataset['dataset'],
                               'checkpoint': checkpoint['checkpoint'], 'checkpoint_sha256': checkpoint['checkpoint_sha256'],
                               'input': dataset['identity'], 'inference_protocol': inference_protocol,
                               'inference_protocol_sha256': digest(inference_protocol),
                               'inference_script_sha256': sha(__file__), 'dependency_hashes': dependency_hashes,
                               'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                                           'dtype': 'float32', 'tf32': False, 'device_family': 'NVIDIA A100', 'batch_size': 64}}
                    run = initialize_run(args.output, request, dataset)
                    runs.append(run)
                    if not run['skip']:
                        committed = {r['slice_id'] for r in run['store'].rows()}
                        has_pending = any(s['slice_id'] not in committed for s in dataset['slices'])
                        if has_pending and model is None:
                            device = device or select_device(args.device, torch)
                            model = load_model(checkpoint, device)
                        infer_run(run, model, device, np, torch)
                    index.append({'run_id': run['run_id'], 'method': checkpoint['method'], 'seed': checkpoint['seed'],
                                  'dataset': dataset['dataset'], 'directory': str(run['path'].resolve()),
                                  'prepared_directory': dataset['directory'],
                                  'receipt_sha256': sha(run['path'] / 'run_receipt.json')})
            finally:
                for run in runs:
                    run['store'].close()
                del model
                gc.collect()
                if device is not None:
                    torch.cuda.empty_cache()
        result = {'status': 'COMPLETE', 'runs': index, 'inference_protocol_sha256': digest(inference_protocol), 'script_sha256': sha(__file__),
                  'datasets': sorted(p['dataset'] for p in prepared), 'training': False}
        result['index_sha256'] = digest(result)
        destination = args.output / ('evaluation_' + digest([r['run_id'] for r in index]) + '.json')
        if destination.exists() and json.loads(destination.read_text()) != result:
            raise ValueError('existing evaluation index differs')
        if not destination.exists():
            atomic_json(destination, result)
        print(json.dumps({'status': 'COMPLETE', 'index': str(destination), 'runs': len(index)}))


if __name__ == '__main__':
    main()
