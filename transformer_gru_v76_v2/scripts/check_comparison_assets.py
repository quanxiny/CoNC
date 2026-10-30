"""Offline CPU verification of recovered historical weights; never trains.

Inputs to every forward are constructed tensors, not source code or data records.
Archived sources are immutable. One declared PACKAGE path literal is rebound in
memory for the real-data model adapter; no model equation or state key is edited.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, importlib.util, json, os, statistics, sys, time, types
from pathlib import Path
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/comparison_recovery'
STORE = ROOT/'src/research_project/ubuntu_local_research_store'
_HASH_CACHE = {}

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    path = Path(path)
    before = path.stat()
    cache_key = (str(path), before.st_size, before.st_mtime_ns)
    if cache_key not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8<<20), b''):
                h.update(block)
        if (path.stat().st_size, path.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError('File changed while hashing: '+str(path))
        _HASH_CACHE[cache_key] = h.hexdigest()
    return _HASH_CACHE[cache_key]

def module_from_file(name, path, package_rebind=False):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    source = Path(path).read_text()
    if package_rebind:
        literal = "PACKAGE = ROOT/'transformer_gru_v76_v2'"
        if source.count(literal) != 1:
            raise ValueError('Unexpected real adapter path declaration')
        source = source.replace(literal, 'PACKAGE = Path('+repr(str(ROOT))+')', 1)
    exec(compile(source, str(path), 'exec'), mod.__dict__)
    return mod

def setup():
    sys.path.insert(0, str(ROOT/'scripts'))
    from portable_runtime import configure_imports
    configure_imports()
    import torch, numpy as np
    torch.set_num_threads(4)
    return torch, np

def safe_load(path, torch, np):
    # Restricted NumPy RNG constructors only. Never fall back to weights_only=False.
    from numpy.core.multiarray import _reconstruct
    allow = [_reconstruct, np.ndarray, np.dtype, type(np.dtype('uint32'))]
    with torch.serialization.safe_globals(allow):
        return torch.load(path, map_location='cpu', weights_only=True)

def fixture(torch):
    b,l,t,n = 2,3,32,4
    x = {'tokens':torch.full((b,l,t), 3, dtype=torch.long),
         'numbers':torch.zeros((b,l,t), dtype=torch.long),
         'marks':torch.ones((b,l,t), dtype=torch.long),
         'line_count':torch.full((b,), l, dtype=torch.long),
         'key':torch.full((b,), l-1, dtype=torch.long),
         'ops':torch.zeros((b,n), dtype=torch.long),
         'parents':torch.zeros((b,n,2), dtype=torch.long),
         'node_line':torch.zeros((b,n), dtype=torch.long),
         'condition':torch.zeros((b,n), dtype=torch.long),
         'unit':torch.zeros((b,n), dtype=torch.long),
         'known':torch.ones((b,n), dtype=torch.long),
         'bits':torch.zeros((b,n,32), dtype=torch.long),
         'target':torch.zeros((b,n), dtype=torch.long),
         'node_count':torch.full((b,), n, dtype=torch.long),
         'edges':torch.eye(l, dtype=torch.long).expand(b,-1,-1).clone()}
    x['numbers'][:,:,2] = 10
    x['marks'][:,:,2] = 0
    x['target'][:,-1] = 1
    x['node_line'][:,-1] = l-1
    return x

_REAL_ADAPTER = None
_REAL_NATIVE = {}

def make_real(run, cfg):
    global _REAL_ADAPTER
    base = ROOT/'assets/comparisons/real_reproduction'
    native_path = ROOT/run['native_implementation']
    expected = cfg['implementation']['native_baselines.py']
    assert sha(native_path) == expected
    if expected not in _REAL_NATIVE:
        native = module_from_file('native_baselines_'+expected[:10], native_path)
        native.PACKAGE = ROOT
        _REAL_NATIVE[expected] = native
    native = _REAL_NATIVE[expected]
    # make_model imports native_baselines lazily, selected by the recorded source hash.
    sys.modules['native_baselines'] = native
    if _REAL_ADAPTER is None:
        adapter_path = base/'scripts/models_adapted.py'
        assert sha(adapter_path) == cfg['implementation']['models_adapted.py']
        _REAL_ADAPTER = module_from_file('recovered_models_adapted', adapter_path, package_rebind=True)
    assert sha(base/'scripts/models_adapted.py') == cfg['implementation']['models_adapted.py']
    nc = dict(cfg['config'].get('native', {}))
    if run['arm'] == 'linevul':
        nc['weights'] = str(base/'vendor/codebert')
    vocab_size = cfg['architecture'].get('vocabulary_size') or 8192
    model, _ = _REAL_ADAPTER.make_model(run['arm'], vocab_size, run['seed'], nc)
    return model

def make_historical(run, cfg):
    from overflow51.baselines import RiskBridge, NativeVD
    group, arm = run['group'], run['arm']
    if group == 'full_overflowgen_v52':
        if arm == 'numeric_all_visible':
            from all_visible52 import AllVisible
            return RiskBridge(AllVisible())
        from overflow51.full_data import make_model
        return make_model(arm, cfg['track'])
    if group == 'adaptive_graph_liquid_v70':
        from model70 import AdaptiveGraph70
        return AdaptiveGraph70(arm)
    if group == 'no_refiner_frozen_v76':
        from model76 import NoRefiner76
        return NoRefiner76()
    if group == 'adaptive_graph_frozen_control_v71':
        sys.path.insert(0, str(STORE/group))
        from model71 import FrozenResidual71
        return FrozenResidual71(arm)
    if group == 'frozen_encoder_dropout_v75':
        sys.path.insert(0, str(STORE/group))
        from model75 import FrozenEncoder75
        return FrozenEncoder75(arm)
    names = {'full_liquid_graph_v72r2':('model72.py','FullLiquidGraph72','Architecture72'),
             'full_liquid_retention_v73r1':('model73.py','FullLiquidGraph73','Architecture73'),
             'full_liquid_line_pair_v74':('model74.py','FullLiquidGraph74','Architecture73')}
    filename, classname, configclass = names[group]
    module = module_from_file('recovered_'+group, STORE/group/filename)
    architecture = getattr(module, configclass)(**cfg['architecture'])
    return getattr(module, classname)(arm, architecture)

def forward(model, run, x, torch, np):
    if run['scope']=='real_reproduction' and run['arm']=='linevul':
        batch = {'input_ids':torch.tensor([[0,5,6,2,1,1,1,1],[0,7,8,2,1,1,1,1]],dtype=torch.long)}
    elif run['scope']=='real_reproduction' and run['arm']=='deepdfa':
        batch = {'features':torch.zeros((2,4),dtype=torch.long),
                 'edge_index':torch.tensor([[0,1],[0,1]],dtype=torch.long),
                 'graph_index':torch.tensor([0,1],dtype=torch.long),'num_graphs':2}
    elif run['group']=='full_overflowgen_v52' and run['arm']=='vd':
        from overflow51.overflowgen import original_vd_batch
        cache = {key:x[key].numpy() for key in ['tokens','numbers','marks','line_count','key']}
        batch = original_vd_batch(cache, np.arange(2))
    else:
        batch = x
    model.eval()
    with torch.inference_mode():
        out = model(batch)
    logits = out['logits']
    assert logits.shape == (2,2) and torch.isfinite(logits).all()
    assert torch.isfinite(out['risk']).all()
    return {'rows':2,'input':'fabricated tensors only','logits_shape':list(logits.shape),
            'logits':logits.tolist(),'finite':True,'optimizer_steps':0,'real_code_records':0}

def audit_run(run, torch, np, file_by_path):
    started = time.monotonic()
    result = {'id':run['id'],'scope':run['scope'],'seed':run['seed'],'checkpoint':run['checkpoint']}
    try:
        checkpoint = ROOT/run['checkpoint']
        assert sha(checkpoint) == run['sha256'], 'Checkpoint hash mismatch'
        for path in [run['config'],run['completion']]:
            assert sha(ROOT/path) == file_by_path[path]['sha256'], 'Evidence changed: '+path
        cfg = read(ROOT/run['config'])
        completion = read(ROOT/run['completion'])
        state = safe_load(checkpoint, torch, np)
        model = make_real(run,cfg) if run['scope']=='real_reproduction' else make_historical(run,cfg)
        expected = model.state_dict()
        actual = state['model']
        assert set(expected)==set(actual), 'State key mismatch'
        assert all(expected[k].shape==actual[k].shape for k in expected), 'State shape mismatch'
        assert all(expected[k].dtype==actual[k].dtype for k in expected), 'State dtype mismatch'
        model.load_state_dict(actual, strict=True)
        count = sum(p.numel() for p in model.parameters())
        recorded_count = cfg.get('parameters',cfg.get('architecture',{}).get('parameters'))
        assert count==recorded_count, f'Parameter count mismatch {count} != {recorded_count}'
        if run['scope']=='real_reproduction':
            assert state['signature']==cfg['signature']==completion['signature']
            assert state['epoch']==completion['best_epoch']
            selection = read((ROOT/run['completion']).parent/'validation_selection.json')
            assert selection['checkpoint_sha256']==run['sha256']
            assert selection['threshold']==run['threshold']
            result['config_signature_match']=True
            result['threshold']=run['threshold']
            result['native_source_sha256_match']=True
        else:
            assert state['config']==cfg['config'], 'Training config mismatch'
            assert state['manifest_sha256']==cfg.get('manifest',cfg.get('manifest_sha256')), 'Data/experiment manifest mismatch'
            offset = 1 if run['group']=='full_overflowgen_v52' else 0
            assert state['earlystop']['best_epoch']+offset==completion['best_epoch'], 'Selected epoch mismatch'
            result['config_and_data_manifest_match']=True
            result['selected_epoch_match']=True
            if 'metrics_receipt' in run:
                assert read(ROOT/run['metrics_receipt'])['checkpoint_sha256']==run['sha256']
                result['historical_metric_checkpoint_match']=True
        result.update(status='PASS',safe_load='weights_only=True with restricted NumPy RNG allowlist',
                      strict_state_dict='PASS',parameters=count,state_tensors=len(actual),
                      forward=forward(model,run,fixture(torch),torch,np))
        del state,model,expected,actual
    except Exception as exc:
        result.update(status='FAIL',error=type(exc).__name__+': '+str(exc))
    gc.collect()
    result['seconds']=round(time.monotonic()-started,3)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['overflowgen_family','overflowgen_author_split','overflowgen_exploratory','real_reproduction'])
    parser.add_argument('--method',help='Exact historical arm/model name')
    parser.add_argument('--seed',type=int,choices=[11,29,47])
    parser.add_argument('--only',help='Substring in the unique run id')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--verify-files',action='store_true',help='Verify every copied file against recorded SHA256 and size')
    parser.add_argument('--report',type=Path,default=EVIDENCE/'verification.json')
    args = parser.parse_args()
    manifest = read(EVIDENCE/'asset_manifest.json')
    selected = [r for r in manifest['runs'] if (not args.scope or r['scope']==args.scope)
                and (not args.method or r['arm']==args.method) and (args.seed is None or r['seed']==args.seed)
                and (not args.only or args.only in r['id'])]
    if not selected:
        raise SystemExit('No matching completed checkpoint; see missing entries in asset_manifest.json')
    if args.list:
        for run in selected:
            print('\t'.join([run['id'],str(run['seed']),run['checkpoint']]))
        return
    torch,np = setup()
    file_by_path = {f['path']:f for f in manifest['files']}
    file_errors = []
    if args.verify_files:
        for f in manifest['files']:
            p = ROOT/f['path']
            if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:
                file_errors.append(f['path'])
        print('FILE HASH AUDIT',len(manifest['files']),'errors',len(file_errors),flush=True)
    results = []
    report = {'manifest_sha256':sha(EVIDENCE/'asset_manifest.json'),'script_sha256':sha(Path(__file__)),
              'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu',
              'optimizer_steps':0,'real_code_records':0,'files_verified':len(manifest['files']) if args.verify_files else 0,
              'file_errors':file_errors,'runtime_path_adjustment':'Only real adapter PACKAGE literal rebound to this package root in memory; archived source unmodified',
              'results':results}
    for run in selected:
        result = audit_run(run,torch,np,file_by_path)
        results.append(result)
        print(result['status'],run['id'],result.get('error',''),flush=True)
        # Save progress so an interruption does not claim uncompleted checks.
        report['status']='IN_PROGRESS'
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    report['status']='PASS' if not file_errors and all(r['status']=='PASS' for r in results) else 'PARTIAL_FAILURE'
    report['passed']=sum(r['status']=='PASS' for r in results)
    report['checked']=len(results)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','passed','checked','files_verified','file_errors']},ensure_ascii=False))
    if report['status']!='PASS':
        raise SystemExit(1)

if __name__=='__main__':
    main()
