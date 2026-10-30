"""ONLY the authorized finite author-token OverflowGen pool, not natural-source files.

The existing bounded token builder is a candidate frontend, not a C analyzer.
No gold labels, flaw-line truth, teacher states or chosen object enters Example.
Existing key_line defines the author's public candidate-slice query for every arm.
"""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import gzip,json,re
import numpy as np
from .baselines import OVERFLOW_ROOT
from .numeric import ObjectVersion,Access
from .model import Example

BUILDER_SHA='13d6a44201cccdbe72018c2587d838dc3f9f802b895015b7f3d002fd56d1c76f'
_builder=None

def _builder_class():
    global _builder
    if _builder is None:
        p=OVERFLOW_ROOT/'joint_graph_study_v1/implementation_family_isolated_v2/graph_input.py'
        data=p.read_bytes()
        if sha256(data).hexdigest()!=BUILDER_SHA:raise ValueError('candidate frontend source changed')
        ns={};exec(compile(data,str(p),'exec'),ns);_builder=ns['Builder']
    return _builder

def author_rows(split):
    if split not in ('train','val','test'):raise ValueError('fixed authorized split only')
    path=OVERFLOW_ROOT/'author_data/data/overflow_gen'/f'{split}_data.json'
    # Constant allowlist path; there is no arbitrary dataset path argument.
    return json.loads(path.read_text())

def normalized_token(token):return '<NUM>' if re.fullmatch(r'[+-]?\d+',token) else token

def fit_vocabulary(rows,train_ids):
    words=set()
    for rid in train_ids:
        for line in rows[rid]['line_code']:words.update(normalized_token(t) for t in line)
    # 0 PAD, 1 UNK; 2..31 event op IDs; text IDs begin at 32.
    return {w:i+32 for i,w in enumerate(sorted(words))}

def from_public_tokens(lines,key,vocabulary):
    if not lines or not 0<=key<len(lines):raise ValueError('invalid public slice query')
    if len(lines)>40 or any(len(line)>32 for line in lines):
        raise ValueError('outside author context budget; no silent truncation in graph arm')
    builder=_builder_class()(lines,key,max_nodes=4096);g=builder.build();nodes=g['nodes']
    objects=[];versions=Counter()
    for ni,node in enumerate(nodes):
        if node['op_name'] not in {'ARRAY','MALLOC'}:continue
        raw=lines[node['line']]
        marker='[' if node['op_name']=='ARRAY' else '='
        at=raw.index(marker) if marker in raw else -1
        name=raw[at-1] if at>0 else f'unknown_declaration_{ni}'
        version=versions[name];versions[name]+=1
        cap=nodes[node['parents'][0]]['exact'] if node['parents'] else None
        unit='bytes' if node['unit']==1 else 'elements' if node['unit']==2 else 'unknown'
        objects.append(ObjectVersion(name,version,cap,unit,ni))
    accesses=[]
    for c in g['rules']['evidence']:
        node=nodes[c['node']];unit='bytes' if node['unit']==1 else 'elements' if node['unit']==2 and c['kind']=='index' else 'unknown'
        # Retain ALL structurally preceding object versions, including ambiguous
        # earlier/later versions of a symbol. No selected capacity / gold binding.
        candidate=tuple(i for i,o in enumerate(objects) if o.node<c['node'])
        accesses.append(Access(c['node'],nodes[c['a']]['exact'],'index' if c['kind']=='index' else 'count',unit,candidate,
                               True if node['condition']==1 else None))
    edges=tuple((p,i) for i,n in enumerate(nodes) for p in n['parents'] if p!=i)
    complete=bool(accesses) and not any(g['stats'].values())
    text=tuple(tuple(vocabulary.get(normalized_token(t),1) for t in lines[n['line']]) for n in nodes)
    example=Example(tuple(n['op'] for n in nodes),edges,tuple(n['literal'] for n in nodes),tuple(objects),tuple(accesses),complete,text)
    return example,{'frontend': 'bounded_author_token_candidates_v51', 'stats':g['stats'],
                    'not_supported':['full C semantics','alias proof','loop invariants','verified path reachability'],
                    'node_count':len(nodes),'candidate_object_versions':len(objects),'accesses':len(accesses)}

def shared_index():
    p=OVERFLOW_ROOT/'joint_graph_study_v1/sample_ids.jsonl.gz'
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]

def original_vd_batch(cache,idx):
    import torch
    counts=torch.as_tensor(np.array(cache['line_count'][idx],copy=True),dtype=torch.long)
    line_mask=torch.arange(int(counts.max()))[None,:]<counts[:,None]
    def get(key):return torch.as_tensor(np.array(cache[key][idx,:int(counts.max())],copy=True),dtype=torch.long)[line_mask]
    return dict(token_ids=get('tokens'),number_ids=get('numbers'),number_mark=get('marks'),line_counts=counts,
                key_padding_mask=~line_mask,key_lines=torch.as_tensor(np.array(cache['key'][idx],copy=True),dtype=torch.long))

def real_source_frontend(*args,**kwargs):
    raise NotImplementedError('No real C/C++ frontend is implemented or authorized in v51')
