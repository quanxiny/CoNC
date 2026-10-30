"""Scope stays at the hash-pinned artificial TRAIN/DEV cache."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STORE=ROOT.parent
sys.path.insert(0,str(STORE/'adaptive_graph_liquid_v70'))
from common70 import SelectedData, KEYS, CACHE, sha, read, write, np, torch
OUT=ROOT/'output'
def protocol():
    p=read(ROOT/'EXPERIMENT_PROTOCOL75.json')
    for name,expected in p['source_sha256'].items():
        if sha(name)!=expected:raise ValueError('Frozen dependency changed: '+name)
    if not (STORE/'selector_recovery_v53/STOP_END_TO_END53').exists():
        raise RuntimeError('Historical test hold absent')
    return p

def validate_execution_freeze():
    f=read(ROOT/'EXECUTION_FREEZE75.json')
    if f['protocol_sha256']!=sha(ROOT/'EXPERIMENT_PROTOCOL75.json'):
        raise ValueError('Protocol differs from execution freeze')
    for name,expected in f['sources'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Execution source changed: '+name)

def repair_logs(dest,offsets):
    """Preserve uncommitted tails before truncating to an explicit checkpoint.

    Only a user-requested --resume enters this function. Never discard evidence.
    Training steps after the checkpoint are replayed, and their cost is separate.
    """
    import time
    for name,offset in offsets.items():
        path=dest/name
        if not path.exists() or path.stat().st_size<offset:raise ValueError('Missing committed log bytes')
        if path.stat().st_size==offset:continue
        archived=dest/(name+'.uncommitted_'+str(time.time_ns()))
        with path.open('rb') as f:
            f.seek(offset); tail=f.read()
        with archived.open('xb') as f:f.write(tail)
        with path.open('r+b') as f:f.truncate(offset)
