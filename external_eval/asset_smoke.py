"""Bounded CPU inference audit of supplied v76/v52 assets; no external data.

Run with the local research_agent Python. Never calls validate_contract(),
training, archived preparation scripts, or the upstream C generator.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'transformer_gru_v76_v2'
STORE = 'src/research_project/ubuntu_local_research_store/'
CACHE = STORE + 'hybrid_numeric_semantic_v59/baseline_dev60/'
KEYS = ['tokens', 'numbers', 'marks', 'line_count', 'key', 'ops', 'parents',
        'node_line', 'condition', 'unit', 'known', 'bits', 'target', 'node_count', 'edges']
SOURCES = [
    'scripts/portable_runtime.py', 'configs/portable_sources.json',
    STORE + 'no_refiner_frozen_v76/model76.py',
    STORE + 'no_refiner_frozen_v76/common76.py',
    STORE + 'adaptive_graph_liquid_v70/common70.py',
    STORE + 'adaptive_graph_liquid_v70/model70.py',
    STORE + 'full_overflowgen_v52/all_visible52.py',
    'external/overflowgen/joint_graph_study_v1/implementation_family_isolated_v2/models.py',
    CACHE + 'MANIFEST60.json',
] + [STORE + 'overflow_interfaces_v51/overflow51/' + n + '.py' for n in
     ['__init__', 'baselines', 'full_data', 'overflowgen', 'numeric', 'model', 'grouping']]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_result(path, payload):
    payload['result_sha256'] = digest(payload)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=2, choices=range(1, 9))
    args = parser.parse_args()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    # Read the manifest as metadata, then hash ONLY this explicit allowlist.
    contract = read(ROOT / 'configs/MIGRATION_CONTRACT_v2.json')['sha256']
    hashes = {}
    for name in SOURCES:
        actual = sha(ROOT / name)
        if actual != contract[name]:
            raise ValueError('Selected inference dependency hash mismatch: ' + name)
        hashes[name] = actual
    sources = read(ROOT / 'configs/portable_sources.json')
    manifest = read(ROOT / CACHE / 'MANIFEST60.json')
    for name in KEYS + ['row', 'split', 'targets']:
        rel = CACHE + name + '.npy'
        actual = sha(ROOT / rel)
        if actual != manifest['array_sha256'][name + '.npy']:
            raise ValueError('Selected cache hash mismatch: ' + name)
        hashes[rel] = actual
    checkpoints = []
    for family, section, hashkey in [('v76', 'trained_v76', 'sha256'),
                                      ('v52', 'transfer_sources', 'checkpoint_sha256')]:
        for seed in [11, 29, 47]:
            item = sources[section][str(seed)]
            rel = item['checkpoint']
            actual = sha(ROOT / rel)
            if actual != item[hashkey]:
                raise ValueError('Checkpoint hash mismatch: ' + rel)
            hashes[rel] = actual
            checkpoints.append((family, seed, item))
    request = {'script_sha256': sha(__file__), 'hashes': hashes, 'rows': args.rows,
               'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
               'device': 'cpu', 'threads': 4}
    request_hash = digest(request)
    destination = HERE / 'asset_smoke_result.json'
    prior = None
    if destination.exists():
        prior = read(destination)
        prior_hash = prior.pop('result_sha256', None)
        if prior.get('request_sha256') != request_hash or digest(prior) != prior_hash:
            raise ValueError('Existing smoke result identity/hash differs; refusing rerun or overwrite')
        if prior.get('status') == 'PASS':
            print(json.dumps({'status': 'SKIP_VERIFIED_PASS', 'result': str(destination),
                              'result_sha256': prior_hash}))
            return
        if prior.get('status') != 'RUNNING':
            raise ValueError('Existing result is not resumable')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from portable_runtime import configure_imports
    configure_imports()
    from common70 import SelectedData
    from model76 import NoRefiner76
    from all_visible52 import AllVisible
    from overflow51.baselines import RiskBridge
    data = SelectedData()
    selected = data.rows('val')[:args.rows]
    x, y = data.batch(selected, 'cpu')
    if set(x) != set(KEYS):
        raise ValueError('Input schema mismatch')
    schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in x.items()}
    records = prior['checkpoints'] if prior else []
    completed = {(r['family'], r['seed']) for r in records}
    allowed = {(family, seed) for family, seed, _ in checkpoints}
    if len(completed) != len(records) or not completed.issubset(allowed):
        raise ValueError('Invalid completed-checkpoint ledger')
    result = {'status': 'RUNNING', 'scope': 'bounded supplied synthetic DEV inference only',
              'request_sha256': request_hash, 'request': request, 'schema': schema,
              'selected_cache_indices': selected.tolist(),
              'original_row_ids': data.a['row'][selected].tolist(),
              'targets': y.tolist(), 'checkpoints': records, 'TEST_rows': 0,
              'training_performed': False, 'external_source_read': False,
              'full_DEV_evaluation': False, 'external_generalization_verified': False,
              'historical_prediction_equality_verified': False}
    write_result(destination, result)
    # Historical own checkpoints contain NumPy RNG states. Use the restricted
    # weights loader with only their required NumPy reconstruction primitives;
    # never fall back to weights_only=False or allow arbitrary checkpoint globals.
    safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype('uint32'))]
    for family, seed, item in checkpoints:
        if (family, seed) in completed:
            continue
        torch.manual_seed(0)
        with torch.serialization.safe_globals(safe):
            state = torch.load(ROOT / item['checkpoint'], map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not isinstance(state.get('model'), dict):
            raise ValueError('Expected own checkpoint with model state dictionary')
        model = NoRefiner76() if family == 'v76' else RiskBridge(AllVisible())
        model.load_state_dict(state['model'], strict=True)
        config = state.get('config', {})
        if isinstance(config, dict) and 'seed' in config and config['seed'] != seed:
            raise ValueError('Checkpoint seed mismatch')
        if family == 'v52' and state.get('manifest_sha256') != item['data_manifest_sha256']:
            raise ValueError('v52 source training manifest identity mismatch')
        model.eval()
        with torch.inference_mode():
            out = model(x)
        if out['logits'].shape != (args.rows, 2) or not torch.isfinite(out['logits']).all():
            raise ValueError('Invalid or nonfinite logits')
        if not torch.allclose(out['risk'], out['logits'].softmax(-1)[:, 1]):
            raise ValueError('Risk readout mismatch')
        records.append({'family': family, 'seed': seed, 'checkpoint': item['checkpoint'],
                        'checkpoint_keys': sorted(state), 'config': config,
                        'steps': state.get('steps'), 'earlystop': state.get('earlystop'),
                        'manifest_sha256': state.get('manifest_sha256'),
                        'parameter_count': sum(p.numel() for p in model.parameters()),
                        'state_namespace': sorted({k.split('.')[0] for k in state['model']}),
                        'logits': out['logits'].tolist(), 'risk': out['risk'].tolist(),
                        'finite': True, 'strict_state_load': True})
        result.pop('result_sha256', None)
        write_result(destination, result)
        del model, state, out
        gc.collect()
    result['status'] = 'PASS'
    result.pop('result_sha256', None)
    write_result(destination, result)
    print(json.dumps({'status': 'PASS', 'result': str(destination),
                      'checkpoint_count': len(records), 'rows_per_checkpoint': args.rows}))


if __name__ == '__main__':
    main()
