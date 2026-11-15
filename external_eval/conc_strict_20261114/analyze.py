"""Strict source comparisons from frozen per-sample predictions, without fitting."""
import argparse, gzip, json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from . import campaign as c

SEEDS = (11,29,47)
ARMS = ('original','literal_only','four_round','exact_topo')
GROUPS = ('simple_array_read','simple_array_write','simple_pointer_write','call_memcpy',
          'call_memmove','call_strcpy','call_strncpy','call_memset','other_or_multiple')
DISPLAY = {'original':'CoNC backbone','literal_only':'Literal-only fusion','four_round':'Four-round exact fusion',
           'exact_topo':'CoNC','vd':'VDTransformer','deepwukong':'DeepWukong-GCN','deepdfa':'DeepDFA',
           'cppcheck':'Cppcheck','clang_sa':'Clang Static Analyzer','frama_c_eva':'Frama-C Eva',
           'H':'CoNC+Trace+Eva','static_control':'Trace+Eva+0'}


def metrics(y, pred, score=None):
    y=np.asarray(y); pred=np.asarray(pred)
    tp=int(((y==1)&(pred==1)).sum()); tn=int(((y==0)&(pred==0)).sum())
    fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
    ratio=lambda a,b:a/b if b else None
    return {'n':len(y),'positive':tp+fn,'negative':tn+fp,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'f1':ratio(2*tp,2*tp+fp+fn) if tp+fn else None,'precision':ratio(tp,tp+fp),
            'recall':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),'accuracy':ratio(tp+tn,len(y)),
            'mcc':float(matthews_corrcoef(y,pred)) if len(np.unique(y))==2 else None,
            'ap':float(average_precision_score(y,score)) if score is not None and tp+fn else None,
            'auroc':float(roc_auc_score(y,score)) if score is not None and len(np.unique(y))==2 else None}


def aggregate(rows):
    result={'seeds':len(rows),'n':rows[0]['n'],'positive':rows[0]['positive'],'negative':rows[0]['negative']}
    for k in ['f1','precision','recall','specificity','accuracy','mcc','ap','auroc']:
        values=[r[k] for r in rows]
        result[k]=None if any(v is None for v in values) else {'mean':float(np.mean(values)),
                   'sd':float(np.std(values,ddof=1)) if len(values)>1 else 0.0}
    return result


def main(core=False):
    c.identity(); c.setup(); meta=c.read(c.OUT/'data/test_rows.json'); ids=[r['sample_id'] for r in meta]
    truth={r['sample_id']:r for r in meta}; predictions={}; receipts={}
    def add(name,seed,rows,idkey='sample_id',scorekey='score'):
        kept={}
        for r in rows:
            sid=r[idkey]
            if sid not in truth: continue
            assert int(r['label'])==int(truth[sid]['label'])
            if r.get('prediction') in (None,''): continue
            assert sid not in kept
            kept[sid]={'prediction':int(r['prediction']), 'score':float(r[scorekey]) if scorekey and r.get(scorekey) not in (None,'') else None}
        predictions[(name,seed)]=kept
    for arm in ARMS:
        for seed in SEEDS:
            path=c.OUT/'predictions'/f'{arm}_s{seed}.csv'; receipt=c.read(path.with_suffix('.json'))
            assert c.sha(path)==receipt['sha256']; receipts[str(path)]=receipt['sha256']
            add(arm,seed,c.csvread(path)); assert len(predictions[(arm,seed)])==11769
    for path in (c.PRIOR/'predictions').glob('*/request.json'):
        request=c.read(path)
        if request['dataset']!='source': continue
        variant=request['variant']; seed=request['seed']
        if variant=='full': continue
        completion=c.read(path.parent/'completion.json'); prediction=path.parent/'scene_predictions.csv'
        assert c.sha(prediction)==completion['artifacts'][prediction.name]
        receipts[str(prediction)]=c.sha(prediction)
        add(variant,seed,c.csvread(prediction),'scene_id')
    if not core:
        for seed in SEEDS:
            folder=c.PRIOR/'deepdfa/runs'/f'seed{seed}'; receipt=c.read(folder/'source_evaluation.json')
            path=folder/'source_slice_predictions.jsonl.gz'; assert c.sha(path)==receipt['slice_prediction_sha256']
            receipts[str(path)]=c.sha(path)
            with gzip.open(path,'rt') as f: add('deepdfa',seed,[json.loads(line) for line in f],scorekey='logit')
            folder=c.OUT/'deepwukong'/f's{seed}'; path=folder/'predictions.csv'
            receipt=c.read(folder/'evaluation.json'); assert c.sha(path)==receipt['sha256']
            receipts[str(path)]=c.sha(path);add('deepwukong',seed,c.csvread(path))
        for tool in ['cppcheck','clang_sa','frama_c_eva']:
            folder=c.OUT/'static'/tool;path=folder/'predictions.csv';receipt=c.read(folder/'completion.json')
            assert c.sha(path)==receipt['sha256'];receipts[str(path)]=c.sha(path)
            add(tool,0,c.csvread(path),scorekey=None)
    variants=sorted({a for a,s in predictions})
    seeds=lambda name:[s for a,s in predictions if a==name]
    def ids_for(names):
        common=set(ids)
        for a in names:
            for seed in seeds(a):common &= predictions[(a,seed)].keys()
        return [sid for sid in ids if sid in common]
    def summarize(names,cohort,tag):
        per_seed=[]; means=[];y=[truth[s]['label'] for s in cohort]
        for a in names:
            values=[]
            for seed in seeds(a):
                vals=[predictions[(a,seed)][sid] for sid in cohort]
                scores=[r['score'] for r in vals]
                m=metrics(y,[r['prediction'] for r in vals],None if None in scores else scores)
                values.append(m);per_seed.append({'cohort':tag,'method':a,'seed':seed,**m})
            means.append({'cohort':tag,'method':a,'display':DISPLAY.get(a,a),**aggregate(values)})
        return per_seed,means
    def bootstrap(names,cohort,comparisons,tag):
        y=np.array([truth[s]['label'] for s in cohort]); groups,index=np.unique([truth[s]['group'] for s in cohort],return_inverse=True)
        counts=np.zeros((len(names),3,len(groups),4),dtype=np.int64)
        for ai,a in enumerate(names):
            for si,s in enumerate(SEEDS):
                mapping=predictions[(a,s if (a,s) in predictions else 0)]
                p=np.array([mapping[sid]['prediction'] for sid in cohort])
                slot=np.where(y==1,np.where(p==1,0,3),np.where(p==1,1,2))
                np.add.at(counts[ai,si],(index,slot),1)
        def f1(x):
            den=2*x[...,0]+x[...,1]+x[...,3]
            return np.divide(2*x[...,0],den,out=np.full(den.shape,np.nan),where=den>0)
        rng=np.random.default_rng(20261114); boot=[]
        for _ in range(40):
            draws=rng.integers(0,len(groups),(50,len(groups)))
            weights=np.stack([np.bincount(d,minlength=len(groups)) for d in draws])
            boot.append(f1(np.einsum('bg,asgc->basc',weights,counts,optimize=True)).mean(axis=2))
        sampled=np.concatenate(boot);point=f1(counts.sum(axis=2)).mean(axis=1);out=[]
        for a,b in comparisons:
            i,j=names.index(a),names.index(b);v=sampled[:,i]-sampled[:,j]
            out.append({'cohort':tag,'left':a,'right':b,'delta_f1':float(point[i]-point[j]),
                        'lower':float(np.nanquantile(v,.025)),'upper':float(np.nanquantile(v,.975)),
                        'components':len(groups),'replicates':2000,'multiplicity_adjusted':False})
        return out
    per_seed=[];means=[]
    for variant in variants:
        p,m=summarize([variant],ids_for([variant]),'full_strict')
        per_seed.extend(p);means.extend(m)
    ci=bootstrap(list(ARMS),ids,[("exact_topo",a) for a in ARMS[:-1]],'full_strict')
    result={'status':'CORE_COMPLETE' if core else 'COMPLETE','display_name':'CoNC','strict_test_n':len(ids),
            'input_receipts':receipts,'per_seed':per_seed,'summary':means,'intervals':ci,
            'all_positive_reference':metrics([truth[s]['label'] for s in ids],np.ones(len(ids),int)),
            'full_access_groups':dict(Counter(r['access_group'] for r in meta)),
            'coverage':{a:{str(s):len(predictions[(a,s)]) for s in seeds(a)} for a in variants}}
    if not core:
        primary=['exact_topo','vd','deepwukong','deepdfa','cppcheck','clang_sa','frama_c_eva']
        common=ids_for(primary);result['common_ids']=common
        p,m=summarize(primary,common,'common');result['per_seed']+=p;result['summary']+=m
        result['intervals']+=bootstrap(primary,common,[('exact_topo',a) for a in primary[1:]],'common')
        groups=[]
        for group in GROUPS:
            cohort=[sid for sid in common if truth[sid]['access_group']==group]
            if cohort:
                p,m=summarize(primary,cohort,group);groups.extend(m);result['per_seed']+=p
            else:groups.append({'cohort':group,'n':0,'status':'NO_STRICT_TEST_SUPPORT'})
        result['rq3']=groups
        path=c.OUT/'trace/predictions.json';receipt=c.read(path.parent/'completion.json');assert c.sha(path)==receipt['sha256']
        receipts[str(path)]=c.sha(path);trace={r['sample_id']:r['D_trace'] for r in c.read(path)}
        assert set(trace)==set(ids)
        counts=[]
        for seed in SEEDS:
            predictions[('H',seed)]={};predictions[('static_control',seed)]={}
            for sid in ids:
                d=trace[sid];eva=predictions[('frama_c_eva',0)].get(sid,{}).get('prediction')
                n=predictions[('exact_topo',seed)][sid]['prediction']
                fixed=d if d in (0,1) else 1 if eva==1 else None
                predictions[('H',seed)][sid]={'prediction':n if fixed is None else fixed,'score':None}
                predictions[('static_control',seed)][sid]={'prediction':0 if fixed is None else fixed,'score':None}
            pos=[s for s in common if truth[s]['label']==1];neg=[s for s in common if truth[s]['label']==0]
            n=lambda s:predictions[('exact_topo',seed)][s]['prediction']
            e=lambda s:predictions[('frama_c_eva',0)][s]['prediction']
            h=lambda s:predictions[('H',seed)][s]['prediction']
            b=lambda s:predictions[('static_control',seed)][s]['prediction']
            counts.append({'seed':seed,'conc_only_positive':sum(n(s)==1 and e(s)==0 for s in pos),
                           'eva_only_positive':sum(n(s)==0 and e(s)==1 for s in pos),
                           'conc_only_false_alert':sum(n(s)==1 and e(s)==0 for s in neg),
                           'eva_only_false_alert':sum(n(s)==0 and e(s)==1 for s in neg),
                           'fallback_positive':sum(h(s)==1 and b(s)==0 for s in pos),
                           'fallback_false_alert':sum(h(s)==1 and b(s)==0 for s in neg)})
        hybrid=['exact_topo','frama_c_eva','static_control','H']
        p,m=summarize(hybrid,common,'rq4_common');result['per_seed']+=p;result['summary']+=m
        result['intervals']+=bootstrap(hybrid,common,[('H','static_control'),('H','exact_topo')],'rq4_common')
        result['rq4_counts']=counts;result['trace_summary']=receipt
    folder=c.OUT/'analysis';folder.mkdir(exist_ok=True)
    name='core_results.json' if core else 'results.json';result['analysis_code_sha256']=c.sha(__file__)
    c.save(folder/name,result)
    c.t.write_csv(folder/('core_per_seed.csv' if core else 'per_seed.csv'),result['per_seed'])
    c.t.write_csv(folder/('core_intervals.csv' if core else 'intervals.csv'),result['intervals'])
    print(json.dumps({'status':result['status'],'summaries':len(result['summary']),'common_n':len(result.get('common_ids',[])),
                     'intervals':result['intervals']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--core-only',action='store_true');args=parser.parse_args();main(args.core_only)
