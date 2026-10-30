"""Existing tool predictions only; uncovered new TEST IDs remain UNKNOWN."""
from collections import defaultdict
from .common import *

def assets():
    source=ROOT/'external_eval/static_baselines/overflowgen_family_test/results';external=ROOT/'external_eval/static_baselines/results'
    s=read(source/'validation.json');e=read(external/'result.json')
    if sha(source/'slice_predictions.csv')!=s['outputs']['slice_predictions.csv']:raise ValueError('Historical source rule hash mismatch')
    if sha(external/'all_native_predictions.csv')!=e['artifacts']['all_native_predictions.csv']:raise ValueError('Historical external rule hash mismatch')
    return {'source':source/'slice_predictions.csv','external':external/'all_native_predictions.csv'}

def predictions(out):
    from .analyze import normalize,csv_rows
    from .evaluate import records_for
    paths=assets();raw=defaultdict(dict);inputs={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    for scope,p in paths.items():
        for r in csv_rows(p):
            dataset='source' if scope=='source' else r['dataset'];method=r.get('method') or r['tool'];raw[(dataset,method)][r['scene_id']]=r
    sets={}
    for dataset in ['source','mit','formai']:
        records,failed,_=records_for(out,dataset);meta={r['scene_id']:r for r in records};meta.update(failed)
        for method in ['cppcheck','clang_sa','frama_c_eva']:
            rows=[]
            for sid,m in sorted(meta.items()):
                r=raw[(dataset,method)].get(sid,{})
                if r and int(r['label'])!=int(m['label']):raise ValueError('Reused static label changed')
                rows.append(normalize({'scene_id':sid,'label':m['label'],'group':m.get('group') or m.get('family_id') or sid,
                                       'prediction':r.get('prediction') if r.get('status')=='PASS' else None,'source_case':m.get('source_case','')}))
            sets[(dataset,method,0)]=rows
    return sets,inputs
