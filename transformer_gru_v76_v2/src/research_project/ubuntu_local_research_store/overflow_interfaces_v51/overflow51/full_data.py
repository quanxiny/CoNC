"""Read the versioned synthetic cache with its shared manifest; never arbitrary data."""
import gzip,json
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from .numeric import ObjectVersion,Access
from .model import Example,collate
from .baselines import OVERFLOW_ROOT
from .overflowgen import original_vd_batch

ROOT=Path(__file__).resolve().parents[1]
_VERIFIED=set()

def verify_file(path,expected):
    path=Path(path);key=(str(path),expected,path.stat().st_size,path.stat().st_mtime_ns)
    if key in _VERIFIED:return
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    if h.hexdigest()!=expected:raise ValueError(f'input integrity failure: {path.name}')
    _VERIFIED.add(key)

def deserialize(row):
    return Example(tuple(row['token_ids']),tuple(map(tuple,row['edges'])),tuple(row['literals']),
        tuple(ObjectVersion(**o) for o in row['objects']),
        tuple(Access(**(a|{'candidate_versions':tuple(a['candidate_versions'])})) for a in row['accesses']),
        row['coverage_complete'],tuple(map(tuple,row['node_text'])))

class FullData:
    def __init__(self,track,arm):
        if track not in {'author','family'}:raise ValueError(track)
        self.track=track;self.arm=arm
        self.manifest=json.loads((ROOT/'output/COMMON_FROZEN_MANIFEST.json').read_text())
        integrity=json.loads((ROOT/'output/CACHE_INPUT_VERIFICATION.json').read_text())
        verify_file(ROOT/'output/row_metadata.jsonl.gz',self.manifest['row_metadata_sha256'])
        with gzip.open(ROOT/'output/row_metadata.jsonl.gz','rt') as f:self.metadata=[json.loads(s) for s in f]
        self.graphs=None;self.cache={}
        if arm.startswith('roles'):
            verify_file(ROOT/'output/public_graphs.jsonl.gz',self.manifest['public_graphs_sha256'])
            with gzip.open(ROOT/'output/public_graphs.jsonl.gz','rt') as f:self.graphs=[deserialize(json.loads(s)) for s in f]
        else:
            keys=['tokens','numbers','marks','line_count','key']
            if arm!='vd':keys+=['ops','parents','node_line','condition','unit','known','bits','target','node_count','edges']
            for k in keys:
                path=OVERFLOW_ROOT/'joint_graph_study_v1'/f'{k}.npy'
                verify_file(path,integrity['original_all_npy_sha256'][k+'.npy'])
                self.cache[k]=np.load(path,mmap_mode='r',allow_pickle=False)
        if len(self.metadata)!=117696 or (self.graphs is not None and len(self.graphs)!=117696):raise ValueError('incomplete full pool')
    def indices(self,split):
        key='original_split' if self.track=='author' else 'family_split'
        return np.array([i for i,r in enumerate(self.metadata) if r[key]==split])
    def batch(self,rows,device='cpu'):
        rows=np.asarray(rows,dtype=np.int64)
        if self.graphs is not None:x=collate([self.graphs[i] for i in rows]).to(device)
        elif self.arm=='vd':x={k:v.to(device) for k,v in original_vd_batch(self.cache,rows).items()}
        else:
            x={k:torch.from_numpy(np.array(v[rows],copy=True)).to(device) for k,v in self.cache.items()}
            l=int(x['line_count'].max());n=int(x['node_count'].max())
            for k in ['tokens','numbers','marks']:x[k]=x[k][:,:l]
            x['edges']=x['edges'][:,:l,:l]
            for k in ['ops','parents','node_line','condition','unit','known','bits','target']:x[k]=x[k][:,:n]
        return x,torch.tensor([self.metadata[i]['label'] for i in rows],dtype=torch.long,device=device)

def make_model(arm,track,width=1472):
    from .model import RoleModel,ModelConfig
    from .baselines import NativeVD,LegacyPureGraph,RiskBridge
    if arm=='vd':return RiskBridge(NativeVD(track))
    if arm in {'semantic','numeric','legacy_joint'}:return RiskBridge(LegacyPureGraph('joint' if arm=='legacy_joint' else arm))
    cfg=ModelConfig(vocabulary=128,width=width,steps=8,cell='gru' if arm=='roles_gru' else 'cfc',arm='roles_free_head' if arm=='roles_free' else 'roles')
    if arm not in {'roles','roles_gru','roles_free'}:raise ValueError(arm)
    return RoleModel(cfg)
