"""Only pinned selected artificial TRAIN/DEV tensors and existing NN artifacts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
STORE = ROOT.parent
sys.path.insert(0, str(STORE/'adaptive_graph_liquid_v70'))
from common70 import SelectedData, KEYS, CACHE, sha, read, write, np, torch
OUT = ROOT/'output'

def protocol():
    p = read(ROOT/'EXPERIMENT_PROTOCOL76.json')
    for path, expected in p['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Pinned dependency changed: '+path)
    if not (STORE/'selector_recovery_v53/STOP_END_TO_END53').exists():
        raise RuntimeError('Historical test hold absent')
    return p

def validate_execution_freeze():
    f = read(ROOT/'EXECUTION_FREEZE76.json')
    if f['protocol_sha256'] != sha(ROOT/'EXPERIMENT_PROTOCOL76.json'):
        raise ValueError('Execution protocol changed')
    for path, expected in f['files'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen implementation changed: '+path)

def restore_legacy_cpu_rng(seed):
    p = read(ROOT/'EXPERIMENT_PROTOCOL76.json')
    item = p['initialization_rng'][str(seed)]
    path = ROOT/item['file']
    if sha(path) != item['sha256']:
        raise ValueError('Initialization RNG artifact changed')
    state = np.load(path, allow_pickle=False)
    if state.dtype != np.uint8 or state.ndim != 1:
        raise ValueError('Invalid CPU RNG state')
    torch.set_rng_state(torch.from_numpy(state.copy()))

def repair_logs(dest, offsets):
    """Explicit resume only; preserve every uncommitted byte before truncating."""
    import time
    for name, offset in offsets.items():
        path = dest/name
        if not path.exists() or path.stat().st_size < offset:
            raise ValueError('Missing committed log bytes')
        if path.stat().st_size == offset:
            continue
        with path.open('rb') as f:
            f.seek(offset)
            tail = f.read()
        with (dest/(name+'.uncommitted_'+str(time.time_ns()))).open('xb') as f:
            f.write(tail)
        with path.open('r+b') as f:
            f.truncate(offset)
