"""Reuse the existing frozen external input implementations and their integrity checks."""
from collections import defaultdict
from functools import lru_cache
from .common import *

@lru_cache(maxsize=4)
def tensor_file(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in KEYS}

def tensor(record):
    z=tensor_file(str(record['tensor_cache']))
    return {k:z[k][int(record['tensor_row'])] for k in KEYS}

def external(name):
    imports()
    import frozen_evaluate as frozen
    import mit_baseline_frozen_evaluate as mit
    if name=='mit':
        d=mit.load_inputs();records=d['extended_slices'];scenes=d['scenes'];ids=d['extended_ids'];strict=d['strict_ids']
        identity={'protocol':sha(ROOT/'external_eval/mit_final_external_protocol.json'),'strict':d['strict']['identity'],'recovered':d['recovered']['receipt']}
    elif name=='formai':
        d=frozen.load_prepared(ROOT/'external_eval/formai_pool/adaptation/test_inputs')
        scenes={r['scene_id']:r for r in d['scenes']};ids={s for s,r in scenes.items() if mit.processable(r)};strict=ids
        records=[r for r in d['slices'] if r['owner_scene_ids'][0] in ids];identity=d['identity']
    else:raise ValueError(name)
    out=[]
    for r in sorted(records,key=lambda r:r['slice_id']):
        sid=r['owner_scene_ids'][0];s=scenes[sid]
        out.append({**r,'sample_id':r['slice_id'],'scene_id':sid,'label':s['label'],
                    'group':s.get('base_case_id') or s.get('base_id') or s.get('family_id') or sid,
                    'strict_eligible':sid in strict,'dataset':name})
    return out,scenes,ids,identity

def input_audit(out):
    report={};ledgers=[]
    for name in ['mit','formai']:
        records,scenes,ids,identity=external(name);scenehashes=defaultdict(list)
        for r in records:
            x=tensor(r);h=canonical_input(x);scenehashes[r['scene_id']].append(h)
            candidate_count=sum(int(x['target'][i]) and int(x['ops'][i]) in (16,17,18) for i in range(int(x['node_count'])))
            ledgers.append({'dataset':name,'sample_id':r['sample_id'],'scene_id':r['scene_id'],'label':r['label'],'group':r['group'],
                            'input_sha256':h,'candidate_count':candidate_count,'structurally_bound_candidates':sum(int(x['target'][i]) and int(x['ops'][i]) in (16,17,18) and all(int(v)>0 for v in x['parents'][i]) for i in range(int(x['node_count']))),
                            'information_sufficiency':'UNVERIFIED_C_SEMANTICS','note':'nonzero parent ids do not certify object or guard correctness'})
        buckets=defaultdict(list)
        for sid in ids:buckets[digest(sorted(set(scenehashes[sid])))].append(sid)
        conflicts=[]
        for h,sids in buckets.items():
            pos=sum(scenes[s]['label'] for s in sids);neg=len(sids)-pos
            if pos and neg:conflicts.append({'input_set_sha256':h,'positive':pos,'negative':neg,'minimum_errors':min(pos,neg),'scene_ids':json.dumps(sorted(sids))})
        write_csv(out/(name+'_scene_collisions.csv'),conflicts,['input_set_sha256','positive','negative','minimum_errors','scene_ids'])
        report[name]={'target_scenes':len(scenes),'processable':len(ids),'unknown':len(scenes)-len(ids),'conflict_groups':len(conflicts),
                      'unavoidable_errors_lower_bound':sum(c['minimum_errors'] for c in conflicts),'identity':identity,
                      'aggregation':'max probability; identical unordered input sets imply identical scene scores'}
    write_csv(out/'external_information_ledger.csv',ledgers)
    atomic_json(out/'external_input_audit.json',report,immutable=True);return report
