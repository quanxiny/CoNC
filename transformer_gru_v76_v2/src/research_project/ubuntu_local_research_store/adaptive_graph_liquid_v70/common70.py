"""Closed selected artificial TRAIN/DEV loader; no global metadata constructor."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STORE = ROOT.parent
OUT = ROOT / 'output'
CACHE = STORE / 'hybrid_numeric_semantic_v59/baseline_dev60'
KEYS = ['tokens', 'numbers', 'marks', 'line_count', 'key', 'ops', 'parents',
        'node_line', 'condition', 'unit', 'known', 'bits', 'target', 'node_count', 'edges']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def protocol():
    p = read(ROOT / 'EXPERIMENT_PROTOCOL70.json')
    for path, expected in p['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    if not (STORE / 'selector_recovery_v53/STOP_END_TO_END53').exists():
        raise RuntimeError('Historical test hold absent')
    return p


class SelectedData:
    def __init__(self):
        m = read(CACHE / 'MANIFEST60.json')
        self.a = {}
        for name in KEYS + ['row', 'split', 'targets']:
            path = CACHE / (name + '.npy')
            if sha(path) != m['array_sha256'][path.name]:
                raise ValueError('Selected-cache hash mismatch: ' + name)
            self.a[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        if len(self.a['row']) != 106119 or set(np.unique(self.a['split'])) != {0, 1}:
            raise ValueError('TRAIN/DEV-only cache contract failed')
        if self.a['numbers'].dtype.kind not in 'iu':
            raise TypeError('Integer cache required')

    def rows(self, split):
        if split not in ['train', 'val']:
            raise ValueError('Only TRAIN and DEV exist in this loader')
        ix = np.flatnonzero(self.a['split'] == (0 if split == 'train' else 1))
        return ix[np.argsort(self.a['row'][ix])]

    def batch(self, rows, device='cpu'):
        x = {k: torch.from_numpy(np.array(self.a[k][rows], copy=True)).to(device) for k in KEYS}
        L, N = int(x['line_count'].max()), int(x['node_count'].max())
        for k in ['tokens', 'numbers', 'marks']:
            x[k] = x[k][:, :L]
        x['edges'] = x['edges'][:, :L, :L]
        for k in ['ops', 'parents', 'node_line', 'condition', 'unit', 'known', 'bits', 'target']:
            x[k] = x[k][:, :N]
        x['numbers'] = x['numbers'].long()
        y = torch.from_numpy(np.array(self.a['targets'][rows], copy=True)).long().to(device)
        return x, y
