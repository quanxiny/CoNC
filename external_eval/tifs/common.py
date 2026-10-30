from __future__ import annotations
import csv, gzip, hashlib, json, os, sys, tempfile, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'transformer_gru_v76_v2'
STORE = PACKAGE / 'src/research_project/ubuntu_local_research_store'
CACHE = PACKAGE / 'external/overflowgen/joint_graph_study_v1'
METADATA = PACKAGE / 'assets/comparisons/overflowgen/overflow_interfaces_v51/output/row_metadata.jsonl.gz'
KEYS = ['tokens','numbers','marks','line_count','key','ops','parents','node_line','condition','unit','known','bits','target','node_count','edges']
NODE_KEYS = ['ops','parents','node_line','condition','unit','known','bits','target']
LINE_KEYS = ['tokens','numbers','marks']
DEFAULT_OUT = ROOT / 'external_eval/tifs_runs'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(p): return json.loads(Path(p).read_text())
def config():
    value = read(HERE/'config.json')
    value['runtime']['python'] = os.environ.get('OVERFLOW_PYTHON', sys.executable)
    value['runtime']['deepdfa_python'] = os.environ.get('DEEPDFA_PYTHON', sys.executable)
    return value

def atomic_json(p,x,immutable=False):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if immutable and p.exists():
        if read(p)!=x: raise ValueError('Refusing changed immutable artifact: '+str(p))
        return
    raw=json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
    with os.fdopen(fd,'w') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)

def write_csv(p,rows,fields=None):
    rows=list(rows); fields=fields or (list(rows[0]) if rows else ['status'])
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(p)

def load_metadata():
    with gzip.open(METADATA,'rt') as f: return [json.loads(s) for s in f]

def imports():
    sys.dont_write_bytecode=True
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ.setdefault(name,'4')
    sys.path.insert(0,str(PACKAGE/'scripts'))
    from portable_runtime import configure_imports
    configure_imports()
    sys.path.insert(0,str(ROOT/'external_eval'))

def code_identity():
    files=list(HERE.glob('*.py'))+[HERE/'config.json']
    files += [STORE/'full_overflowgen_v52/all_visible52.py', STORE/'adaptive_graph_liquid_v70/model70.py',
              CACHE/'implementation_family_isolated_v2/models.py', CACHE/'implementation_family_isolated_v2/graph_input.py',
              STORE/'overflow_interfaces_v51/overflow51/training.py', STORE/'overflow_interfaces_v51/overflow51/baselines.py',
              STORE/'overflow_interfaces_v51/overflow51/overflowgen.py', PACKAGE/'scripts/portable_runtime.py',
              ROOT/'external_eval/legacy_frontend.py', ROOT/'external_eval/mit_numeric_ood.py',
              ROOT/'external_eval/frozen_evaluate.py', ROOT/'external_eval/mit_baseline_frozen_evaluate.py',
              ROOT/'external_eval/baselines/deepdfa/training.py', ROOT/'external_eval/baselines/deepdfa/native_model.py',
              ROOT/'external_eval/baselines/deepdfa/evaluation.py',
              ROOT/'external_eval/paper_baselines/deepdfa/protocol.json',
              ROOT/'external_eval/paper_baselines/deepdfa/cpu_runtime.json',
              ROOT/'external_eval/paper_baselines/deepdfa/evaluation_contract.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}

def verify_identity(files):
    for name,h in files.items():
        if sha(ROOT/name)!=h: raise ValueError('Frozen dependency mismatch: '+name)

def output_dir(p):
    p=Path(p).resolve()
    if not p.is_relative_to(DEFAULT_OUT): raise ValueError('Outputs must stay in external_eval/tifs_runs')
    return p

def campaign_root(): return DEFAULT_OUT/config()['protocol']

def setup(seed,threads=4):
    imports()
    import numpy as np, random, torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return np,torch

def canonical_input(x):
    """All consumed values, canonical lengths; ignores padding and unused query key."""
    import numpy as np
    h=hashlib.sha256(); l=int(x['line_count']); n=int(x['node_count'])
    for k in KEYS:
        if k=='key': continue  # model forward uses target, never key
        a=np.asarray(x[k])
        if k in LINE_KEYS:a=a[:l]
        elif k in NODE_KEYS:a=a[:n]
        elif k=='edges':a=a[:l,:l]
        # Cache int16 vs inference int64 is the same integer input.
        a=np.asarray(a,dtype='<i8',order='C')
        h.update(k.encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()

class SourceData:
    def __init__(self):
        import numpy as np
        self.a={k:np.load(CACHE/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in KEYS+['labels']}
        self.metadata=load_metadata()
        if len(self.metadata)!=len(self.a['labels']):raise ValueError('Row count mismatch')
    def batch(self,rows,device='cpu',vd=False):
        import numpy as np,torch
        if vd:
            from overflow51.overflowgen import original_vd_batch
            x={k:v.to(device) for k,v in original_vd_batch(self.a,rows).items()}
        else:
            x=stack_inputs([{k:self.a[k][i] for k in KEYS} for i in rows],device)
        y=torch.from_numpy(np.asarray(self.a['labels'][rows],dtype='int64').copy()).to(device)
        return x,y

def stack_inputs(items,device='cpu'):
    import numpy as np,torch
    x={k:torch.from_numpy(np.stack([r[k] for r in items])).to(device) for k in KEYS}
    l=int(x['line_count'].max());n=int(x['node_count'].max())
    for k in LINE_KEYS:x[k]=x[k][:,:l]
    for k in NODE_KEYS:x[k]=x[k][:,:n]
    x['edges']=x['edges'][:,:l,:l];x['numbers']=x['numbers'].long()
    return x

def stable_id(m):return 'og24455320:'+m['original_split']+':'+str(m['original_sample_id'])

def require_freeze(out):
    f=read(out/'freeze.json');verify_identity(f['code'])
    if f['config']!=config():raise ValueError('Config changed after freeze')
    for name,h in f['artifacts'].items():
        if sha(out/name)!=h:raise ValueError('Frozen artifact changed: '+name)
    if f.get('schema') == 'source-only-release-v1':
        verify_identity(f['dependencies'])
    else:
        verify_identity(read(out/'preflight.json')['dependencies'])
    return f
