"""Read-only v70 lineage and identical selected synthetic TRAIN/DEV loader."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
V70=ROOT.parent/'adaptive_graph_liquid_v70'
sys.path.insert(0,str(V70))
from common70 import *
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'
V70=ROOT.parent/'adaptive_graph_liquid_v70'


def protocol():
    p=read(ROOT/'EXPERIMENT_PROTOCOL71.json')
    assert sha(V70/'EXPERIMENT_PROTOCOL70.json')==p['v70_protocol_sha256']
    original=read(V70/'EXPERIMENT_PROTOCOL70.json')
    for path,digest in {**original['source_sha256'],**p['source_sha256']}.items():
        if sha(path)!=digest:raise ValueError('Frozen lineage source changed: '+path)
    for ref in p['liquid_frozen_references'].values():
        for key in ['config','completion','DEV_predictions']:
            if sha(ref[key])!=ref[key+'_sha256']:raise ValueError('v70 reference changed: '+key)
    assert (STORE/'selector_recovery_v53/STOP_END_TO_END53').exists()
    return p
