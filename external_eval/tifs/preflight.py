"""Source asset identity, group isolation and a genuinely label-blind assignment."""
from collections import Counter,defaultdict
import csv,gzip,json,time
from .common import *

class UnionFind:
    def __init__(self,n):self.p=list(range(n));self.size=[1]*n
    def find(self,a):
        while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.p[b]=a;self.size[a]+=self.size[b]

def partition(metadata,input_hashes,spec):
    uf=UnionFind(len(metadata));seen={}
    for i,m in enumerate(metadata):
        for field in spec['group_fields']+['input_sha256']:
            value=input_hashes[i] if field=='input_sha256' else m[field]
            key=(field,str(value))
            if key in seen:uf.union(i,seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i in range(len(metadata)):groups[uf.find(i)].append(i)
    ordered=sorted(groups.values(),key=lambda v:(-len(v),digest([spec['seed'],v[0]])))
    targets=[len(metadata)*v for v in spec['fractions']];counts=[0,0,0]
    assignment={};members={}
    # No labels are read by the allocator. Large components stay intact.
    for rows in ordered:
        dest=max(range(3),key=lambda j:(targets[j]-counts[j],-j))
        name=['train','val','test'][dest];gid='component:'+str(rows[0])
        members[gid]=rows
        for i in rows:assignment[i]=(name,gid)
        counts[dest]+=len(rows)
    summary={};failures=[]
    for name in ['train','val','test']:
        rows=[i for i,(s,g) in assignment.items() if s==name]
        c=Counter(assignment[i][1] for i in rows);pos=sum(metadata[i]['label'] for i in rows)
        summary[name]={'n':len(rows),'positive':pos,'negative':len(rows)-pos,'groups':len(c),
                       'largest_group':max(c.values(),default=0),'largest_group_fraction':max(c.values(),default=0)/max(len(rows),1)}
        if name!='train':
            if len(c)<spec['minimum_eval_groups']:failures.append(name+':too_few_independent_groups')
            if min(pos,len(rows)-pos)<spec['minimum_eval_per_class']:failures.append(name+':too_few_per_class')
            if summary[name]['largest_group_fraction']>spec['maximum_eval_group_fraction']:failures.append(name+':dominant_component')
    if summary['train']['n']/len(metadata)>spec['maximum_train_fraction']:failures.append('train:dominates_pool')
    for field in spec['group_fields']+['input_sha256']:
        buckets=defaultdict(set)
        for i,m in enumerate(metadata):buckets[assignment[i][0]].add(input_hashes[i] if field=='input_sha256' else m[field])
        for a,b in [('train','val'),('train','test'),('val','test')]:
            if buckets[a]&buckets[b]:raise AssertionError('Leaked grouping field '+field)
    return assignment,summary,failures

def run(out):
    import numpy as np
    out=output_dir(out);out.mkdir(parents=True,exist_ok=True)
    cfg=config();start=time.monotonic();receipt=read(CACHE/'CACHE_RECEIPT.json')
    dependencies={str(CACHE.relative_to(ROOT)/name):h for name,h in receipt['cache_hashes'].items()}
    dependencies[str(METADATA.relative_to(ROOT))]=read(ROOT/'reproducibility/data_split_check.json')['metadata_sha256']
    dependencies[str(HERE.relative_to(ROOT)/'config.json')]=sha(HERE/'config.json')
    dependencies[str(HERE.relative_to(ROOT)/'preflight.py')]=sha(__file__)
    dependencies[str(HERE.relative_to(ROOT)/'common.py')]=sha(HERE/'common.py')
    verify_identity(dependencies)
    if (out/'preflight.json').exists():
        prior=read(out/'preflight.json')
        if prior['dependencies']!=dependencies:raise ValueError('P0 identity changed; preserve prior attempt explicitly')
        for p,h in prior['artifacts'].items():
            if sha(out/p)!=h:raise ValueError('P0 output changed: '+p)
        return prior
    data=SourceData();metadata=data.metadata;hashes=[];collisions=defaultdict(list)
    rowsout=[]
    for i,m in enumerate(metadata):
        if m['row']!=i or int(data.a['labels'][i])!=m['label']:raise ValueError('Row/label binding mismatch')
        x={k:data.a[k][i] for k in KEYS};h=canonical_input(x);hashes.append(h);collisions[h].append(i)
        if i%10000==0:print(json.dumps({'stage':'input_census','rows':i,'elapsed_s':time.monotonic()-start}),flush=True)
    assignment,summary,failures=partition(metadata,hashes,cfg['split'])
    for i,m in enumerate(metadata):
        split,group=assignment[i]
        rowsout.append({'row':i,'sample_id':stable_id(m),'label':m['label'],'split':split,'group':group,
                        'source_case':m['source_case'],'family_id':m['family_id'],'historical_split':m['family_split'],
                        'input_sha256':hashes[i],'normalized_slice_sha256':m['normalized_slice_sha256']})
    write_csv(out/'source_manifest.csv',rowsout)
    collisions_out=[]
    for h,rows in collisions.items():
        pos=sum(metadata[r]['label'] for r in rows);neg=len(rows)-pos
        if pos and neg:collisions_out.append({'input_sha256':h,'n':len(rows),'positive':pos,'negative':neg,'unavoidable_errors':min(pos,neg),'rows':json.dumps(rows)})
    write_csv(out/'source_input_collisions.csv',collisions_out,['input_sha256','n','positive','negative','unavoidable_errors','rows'])
    # Original source magnitudes, inspected without predictions.
    max_token=max(int(data.a['numbers'][i][data.a['marks'][i]==0].max(initial=0)) for i in range(len(metadata)))
    report={'status':'PASS' if not failures else 'BLOCKED_STRICT_SPLIT','failures':failures,'source':'overflowgen_24455320',
            'dependencies':dependencies,'groups':summary,'n':len(metadata),'new_split_not_pristine_unseen':True,
            'maximum_literal':max_token,'mixed_input_groups':len(collisions_out),
            'unavoidable_errors_lower_bound':sum(r['unavoidable_errors'] for r in collisions_out),
            'no_collision_does_not_prove_semantic_sufficiency':True,'near_clone_scope':'existing normalized lexical identities only; no semantic clone certification',
            'artifacts':{p.name:sha(p) for p in [out/'source_manifest.csv',out/'source_input_collisions.csv']},
            'wall_s':time.monotonic()-start,'model_forward_calls':0}
    atomic_json(out/'preflight.json',report,immutable=True)
    return report

def real_pair_feasibility(out):
    """Census all existing paired cases; hash-select review rows without scores."""
    source=ROOT/'external_eval/primevul_v0_1/manifest/scenes.jsonl'
    status=ROOT/'external_eval/primevul_v0_1/results/all_scene_status.csv'
    with source.open() as f:scenes=[json.loads(s) for s in f]
    with status.open() as f:ledger=list(csv.DictReader(f))
    byid={r['scene_id']:r for r in ledger};groups=defaultdict(list)
    for s in scenes:
        group=s.get('pair_id') or s.get('family_id')
        if group is None:raise ValueError('Missing real-pair identity')
        groups[str(group)].append(s)
    rows=[]
    for gid,items in groups.items():
        rows.append({'group':gid,'scenes':json.dumps([s['scene_id'] for s in items]),'labels':json.dumps([s['label'] for s in items]),
                     'existing_status':json.dumps([byid.get(s['scene_id'],{}) for s in items],sort_keys=True),
                     'selection_hash':digest([config()['split']['seed'],gid]),'independently_reviewed':False})
    rows.sort(key=lambda r:r['selection_hash'])
    write_csv(out/'real_pair_ledger.csv',rows)
    write_csv(out/'real_pair_review_queue.csv',rows[:config()['real_pairs']['maximum_feasibility_pairs']])
    atomic_json(out/'real_pair_feasibility.json',{'status':'EXISTING_ASSET_CENSUS_ONLY','groups':len(rows),'scenes':len(scenes),
        'inputs':{str(source.relative_to(ROOT)):sha(source),str(status.relative_to(ROOT)):sha(status)},
        'new_label_or_context_verification':False,'confirmatory_real_world_test':'BLOCKED_PENDING_PAIR_LEVEL_CONTEXT_AND_LABEL_VERIFICATION',
        'prediction_based_selection':False,'new_downloads':False},immutable=True)
