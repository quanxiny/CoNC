"""Internal conditional tensor rule. It is not a C verifier or an independent published tool."""
from collections import Counter
from .common import *

def decide(x):
    n=int(x['node_count']);values=[None]*n;unknown=[];states=[]
    for i in range(n):
        op=int(x['ops'][i]);a,b=map(int,x['parents'][i]);cond=int(x['condition'][i])
        if op==1 and x['known'][i]:
            values[i]=sum(int(v)<<k for k,v in enumerate(x['bits'][i]))
        elif op==2 and 0<a<i:values[i]=values[a]
        elif op in (3,4) and 0<a<i and 0<b<i and values[a] is not None and values[b] is not None:
            value=values[a]+values[b] if op==3 else values[a]-values[b]
            if -(2**31)<=value<2**31:values[i]=value
        if not int(x['target'][i]) or op not in (16,17,18):continue
        if cond!=1:unknown.append('conditional_access_path_unresolved');continue
        if not (0<a<i and 0<b<i) or values[a] is None or values[b] is None:unknown.append('operand_or_capacity_unknown');continue
        if int(x['unit'][i])==0 or (op in (17,18) and int(x['unit'][i])!=1):unknown.append('unit_unknown');continue
        index,capacity=values[a],values[b]
        states.append(int(index<0 or capacity<0 or (index>=capacity if op==16 else index>capacity)))
    if 1 in states:return {'prediction':1,'status':'CONDITIONAL_ALERT','reason':'represented_bound_violated'}
    if states and not unknown:return {'prediction':0,'status':'CONDITIONAL_NO_VIOLATION','reason':'all_represented_bounds_satisfied'}
    return {'prediction':None,'status':'UNKNOWN','reason':';'.join(sorted(set(unknown))) or 'no_target_access'}

def run(out):
    import numpy as np
    require_freeze(out)
    if (out/'tensor_rule_receipt.json').exists():
        prior=read(out/'tensor_rule_receipt.json')
        if prior['freeze_sha256']!=sha(out/'freeze.json') or sha(out/'tensor_rule_slice_predictions.csv')!=prior['prediction_sha256']:raise ValueError('Rule prediction contract changed')
        return prior
    from .datasets import external,tensor
    data=SourceData()
    with (out/'source_manifest.csv').open() as f:manifest=list(csv.DictReader(f))
    rows=[]
    for r in manifest:
        if r['split']!='test':continue
        i=int(r['row']);x={k:data.a[k][i] for k in KEYS}
        rows.append({'dataset':'source','sample_id':r['sample_id'],'scene_id':r['sample_id'],'label':int(r['label']),'group':r['group'],**decide(x)})
    for name in ['mit','formai']:
        records,scenes,ids,identity=external(name)
        if identity!=read(out/'external_input_audit.json')[name]['identity']:raise ValueError('External rule input changed')
        for r in records:rows.append({k:r[k] for k in ['dataset','sample_id','scene_id','label','group']}|decide(tensor(r)))
    if (out/'diagnostics/ledger.csv').exists():
        with (out/'diagnostics/ledger.csv').open() as f:diagnostics=list(csv.DictReader(f))
        for r in diagnostics:
            if r['status']!='PASS':continue
            result=read(r['result_path'])
            for s in result['slices']:
                rows.append({'dataset':'diagnostic','sample_id':s['sample_id'],'scene_id':r['sample_id'],'label':int(r['label']),'group':r['group'],**decide(tensor(s))})
    write_csv(out/'tensor_rule_slice_predictions.csv',rows)
    atomic_json(out/'tensor_rule_receipt.json',{'status':'COMPLETE','rows':len(rows),'counts':dict(Counter(r['status'] for r in rows)),
        'freeze_sha256':sha(out/'freeze.json'),
        'prediction_sha256':sha(out/'tensor_rule_slice_predictions.csv'),'code_sha256':sha(__file__),
        'inputs':'only the original 15 tensors; no Builder exact intermediates, labels or source oracle enter decide',
        'scope':'conditional candidate graph bound check under represented object binding, units and int32 arithmetic; not C safety proof',
        'unknown_is_not_safe':True,'rank_metrics':'NA: hard decisions have no continuous scores'},immutable=True)
