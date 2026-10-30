"""Prediction-only analysis. Seeds are repeated fits, never bootstrap units."""
import argparse,math
from collections import defaultdict,Counter
from .common import *

def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def normalize(r):
    r=dict(r);r['label']=int(r['label']);r['prediction']=None if r.get('prediction') in ('',None) else int(r['prediction'])
    for key in ['score','probability']:r[key]=None if r.get(key) in ('',None) else float(r[key])
    r['status']='SCORED' if r['prediction'] is not None else 'UNKNOWN'
    return r

def metrics(rows):
    import numpy as np
    from sklearn.metrics import average_precision_score,roc_auc_score
    good=[r for r in rows if r['status']=='SCORED'];n=len(good)
    y=np.array([r['label'] for r in good]);p=np.array([r['prediction'] for r in good])
    tp=int(((y==1)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
    pos=int(y.sum());den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    result={'target':len(rows),'processed':n,'unknown':len(rows)-n,'positive':pos,'negative':n-pos,'groups':len({r['group'] for r in good}),
            'tp':tp,'tn':tn,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'specificity':tn/(tn+fp) if tn+fp else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'mcc':(tp*tn-fp*fn)/math.sqrt(den) if den else None,'all_positive_f1':2*pos/(n+pos) if n else None,
            'ap':None,'auroc':None,'rank_metric_reason':'NA: missing continuous scores or both classes'}
    if pos and n-pos and all(r.get('score') is not None for r in good):
        scores=[r['score'] for r in good];result.update(ap=float(average_precision_score(y,scores)),auroc=float(roc_auc_score(y,scores)),rank_metric_reason='raw continuous score')
    return result

def paired(rows,dataset):
    buckets=defaultdict(list)
    for r in rows:buckets[(r.get('pair_id') or r['group']) if dataset=='diagnostic' else r['group']].append(r)
    possible=eligible=correctpairs=totalpairs=full4=correct4=0;units=[];score_deltas=[]
    for group,items in buckets.items():
        if {r['label'] for r in items}!={0,1}:continue
        possible+=1;ok=[r for r in items if r['status']=='SCORED'];a=[r for r in ok if r['label']==0];b=[r for r in ok if r['label']==1]
        if a and b:
            eligible+=1;den=len(a)*len(b);num=sum(x['prediction']==0 and y['prediction']==1 for x in a for y in b)
            correctpairs+=num;totalpairs+=den
            score_deltas.extend(y['score']-x['score'] for x in a for y in b if x.get('score') is not None and y.get('score') is not None)
            units.append({'group':items[0]['group'],'numerator':num,'denominator':den})
        if dataset=='mit' and len(items)==4 and len(ok)==4:
            full4+=1;correct4+=all(r['prediction']==r['label'] for r in items)
    return {'target_opposite_label_groups':possible,'eligible_opposite_label_groups':eligible,'pair_group_coverage':eligible/possible if possible else None,
            'eligible_safe_bad_pairs':totalpairs,'both_correct_pairs':correctpairs,'pair_accuracy':correctpairs/totalpairs if totalpairs else None,
            'score_pairs':len(score_deltas),'bad_score_above_safe_rate':sum(d>0 for d in score_deltas)/len(score_deltas) if score_deltas else None,
            'score_tie_rate':sum(d==0 for d in score_deltas)/len(score_deltas) if score_deltas else None,'mean_bad_minus_safe_score':sum(score_deltas)/len(score_deltas) if score_deltas else None,
            'eligible_four_version_cases':full4,'all_four_correct_cases':correct4,'all_four_accuracy':correct4/full4 if full4 else None},units

def numeric_summary(rows):
    """Node-weighted conditional graph probes; retain sample counts and grouping scope."""
    bags=defaultdict(list)
    for r in rows:
        literal=str(r['literal']).lower()=='true'
        if not literal and int(r['operation']) not in (2,3,4):continue
        kind='literal' if literal else 'computation'
        for scope in ['all','dependency_depth','carry_or_borrow_length']:
            value='ALL' if scope=='all' else str(r.get(scope) if r.get(scope) not in ('',None) else 'UNKNOWN')
            bags[(str(r['layer']),kind,scope,value)].append(r)
    answer=[]
    for (layer,kind,scope,value),items in sorted(bags.items()):
        def mean(key):return sum(float(r[key]) for r in items)/len(items)
        drifts=[float(r['literal_max_abs_drift']) for r in items if r.get('literal_max_abs_drift') not in ('',None)]
        answer.append({'layer':layer,'node_kind':kind,'grouping':scope,'group_value':value,'nodes':len(items),
            'samples':len({r.get('sample_id','fixture') for r in items}),
            'integer_exact_rate':sum(str(r['integer_exact']).lower()=='true' for r in items)/len(items),
            'effective_bit_accuracy_mean':mean('effective_bit_accuracy'),'all_bit_accuracy_mean':mean('all_bit_accuracy'),
            'literal_drift_mean':sum(drifts)/len(drifts) if drifts else None,'literal_drift_max':max(drifts) if drifts else None,
            'interpretation':'node-weighted; conditional supplied-graph reference, not C execution truth'})
    return answer

def bootstrap_difference(a,b,seed,reps):
    """Same group draw for both methods and all three seeds; average seed effects."""
    import numpy as np
    seeds=sorted(set(a)&set(b));ids=set.intersection(*[{r['scene_id'] for r in a[s] if r['status']=='SCORED'} & {r['scene_id'] for r in b[s] if r['status']=='SCORED'} for s in seeds])
    lookup=[({r['scene_id']:r for r in a[s]},{r['scene_id']:r for r in b[s]}) for s in seeds]
    groups=sorted({lookup[0][0][i]['group'] for i in ids});gi={g:i for i,g in enumerate(groups)}
    if len(groups)<2:return {'n':len(ids),'groups':len(groups),'status':'NA_TOO_FEW_GROUPS'}
    counts=np.zeros((len(groups),2,len(seeds),4),dtype=np.int64)
    for j,(aa,bb) in enumerate(lookup):
        for i in sorted(ids):
            if (aa[i]['label'],aa[i]['group'])!=(bb[i]['label'],bb[i]['group']):raise ValueError('Paired label/group mismatch')
            for k,rr in enumerate([aa[i],bb[i]]):
                y,p=rr['label'],rr['prediction'];index=0 if y and p else 1 if not y and p else 2 if y and not p else 3
                counts[gi[rr['group']],k,j,index]+=1
    def f1(c):
        den=2*c[...,0]+c[...,1]+c[...,2]
        return np.divide(2*c[...,0],den,out=np.zeros_like(den,dtype=float),where=den!=0)
    point=f1(counts.sum(0));rng=np.random.default_rng(seed);deltas=[]
    for start in range(0,reps,100):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(100,reps-start))
        boot=f1(np.einsum('bg,gksc->bksc',weights,counts,optimize=True));deltas.extend((boot[:,0]-boot[:,1]).mean(1).tolist())
    lo,hi=np.quantile(deltas,[.025,.975]);result={'n':len(ids),'groups':len(groups),'status':'PASS','effect_mean_f1':float((point[0]-point[1]).mean()),'ci_low':float(lo),'ci_high':float(hi),'replicates':reps}
    for s,v in zip(seeds,point[0]-point[1]):result['seed_'+str(s)+'_f1_delta']=float(v)
    return result

def bootstrap_pairs(a,b,dataset,seed,reps):
    import numpy as np
    seeds=sorted(set(a)&set(b));ids=set.intersection(*[{r['scene_id'] for r in a[s] if r['status']=='SCORED'} & {r['scene_id'] for r in b[s] if r['status']=='SCORED'} for s in seeds])
    bags=[];allgroups=set()
    for s in seeds:
        both=[]
        for method in [a,b]:
            _,units=paired([r for r in method[s] if r['scene_id'] in ids],dataset);d=defaultdict(lambda:[0,0])
            for u in units:d[u['group']][0]+=u['numerator'];d[u['group']][1]+=u['denominator']
            both.append(d);allgroups.update(d)
        bags.append(both)
    groups=sorted(allgroups)
    if len(groups)<2:return {'status':'NA_TOO_FEW_PAIRED_GROUPS','groups':len(groups)}
    counts=np.array([[[bags[s][k].get(g,[0,0]) for s in range(len(seeds))] for k in range(2)] for g in groups])
    def rate(c):return np.divide(c[...,0],c[...,1],out=np.full(c[...,0].shape,np.nan),where=c[...,1]!=0)
    point=rate(counts.sum(0));rng=np.random.default_rng(seed);deltas=[]
    for start in range(0,reps,100):
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(100,reps-start))
        rates=rate(np.einsum('bg,gksc->bksc',weights,counts,optimize=True));deltas.extend(np.nanmean(rates[:,0]-rates[:,1],1).tolist())
    valid=np.array(deltas);valid=valid[np.isfinite(valid)];lo,hi=np.quantile(valid,[.025,.975])
    return {'status':'PASS','groups':len(groups),'effect_mean_pair_accuracy':float(np.nanmean(point[0]-point[1])),'ci_low':float(lo),'ci_high':float(hi),'replicates':reps,'finite_replicates':len(valid)}

def interaction(methods,dataset,endpoint):
    import numpy as np
    names=['full','no_h_to_q','no_dynamic_q_to_h','no_both_dynamic'];seeds=config()['seeds'];spec=config()['statistics']
    if any(set(methods.get(m,{}))!=set(seeds) for m in names):return {'status':'NA_MISSING_ARM'}
    ids=set.intersection(*[{r['scene_id'] for r in methods[m][s] if r['status']=='SCORED'} for m in names for s in seeds]);bags={};groups=set()
    for m in names:
        for s in seeds:
            rs=[r for r in methods[m][s] if r['scene_id'] in ids];bag=defaultdict(lambda:[0]*(4 if endpoint=='f1' else 2))
            if endpoint=='f1':
                for r in rs:
                    y,p=r['label'],r['prediction'];j=0 if y and p else 1 if not y and p else 2 if y and not p else 3;bag[r['group']][j]+=1
            else:
                for u in paired(rs,dataset)[1]:bag[u['group']][0]+=u['numerator'];bag[u['group']][1]+=u['denominator']
            bags[(m,s)]=bag;groups.update(bag)
    groups=sorted(groups)
    if len(groups)<2:return {'status':'NA_TOO_FEW_GROUPS'}
    shape=4 if endpoint=='f1' else 2;c=np.array([[[bags[(m,s)].get(g,[0]*shape) for s in seeds] for m in names] for g in groups]);contrast=np.array([1,-1,-1,1])
    def effect(x):
        numerator=2*x[...,0] if endpoint=='f1' else x[...,0];den=2*x[...,0]+x[...,1]+x[...,2] if endpoint=='f1' else x[...,1]
        rate=np.divide(numerator,den,out=np.zeros_like(den,dtype=float),where=den!=0)
        return (rate*contrast.reshape((1,)*max(0,rate.ndim-2)+(4,1))).sum(-2).mean(-1)
    point=float(effect(c.sum(0)));rng=np.random.default_rng(spec['bootstrap_seed']);effects=[]
    for start in range(0,spec['bootstrap_replicates'],100):
        w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=min(100,spec['bootstrap_replicates']-start));effects.extend(effect(np.einsum('bg,gmsc->bmsc',w,c,optimize=True)).tolist())
    lo,hi=np.quantile(effects,[.025,.975]);return {'status':'PASS','n':len(ids),'groups':len(groups),'endpoint':endpoint,'contrast':'Full - no_HQ - no_QH + no_both','effect':point,'ci_low':float(lo),'ci_high':float(hi),'replicates':spec['bootstrap_replicates'],'interpretation':'interaction on this metric scale; not proof of causal synergy'}

def load_predictions(out):
    sets={};inputs={}
    from .static_reuse import predictions as static_predictions
    static,static_inputs=static_predictions(out);sets.update(static)
    if static_inputs!=read(out/'static_reuse_identity.json'):raise ValueError('Static inputs changed after freeze')
    inputs.update({'workspace:'+p:h for p,h in static_inputs.items()})
    for p in sorted((out/'predictions').glob('*/completion.json')):
        c=read(p);req=c['request'];dest=p.parent
        for name,h in c['artifacts'].items():
            if sha(dest/name)!=h:raise ValueError('Prediction changed')
        key=(req['dataset'],req['variant'],int(req['seed']))
        if key in sets:raise ValueError('Multiple completed runs for the same frozen method/seed/dataset')
        sets[key]=[normalize(r) for r in csv_rows(dest/'scene_predictions.csv')];inputs[str(p.relative_to(out))]=sha(p)
    # Published native baseline keeps its own documented threshold and coverage.
    from .evaluate import records_for
    for seed in config()['seeds']:
        for dataset in ['source','mit','formai']:
            folder=out/'deepdfa/runs'/('seed'+str(seed));p=folder/(dataset+'_evaluation.json')
            if not p.exists():continue
            receipt=read(p)
            for name,h in receipt['artifacts'].items():
                if sha(folder/name)!=h:raise ValueError('DeepDFA predictions changed')
            records,failed,_=records_for(out,dataset);meta={r['scene_id']:r for r in records};meta.update(failed)
            with gzip.open(folder/(dataset+'_predictions.jsonl.gz'),'rt') as f:raw=[json.loads(s) for s in f]
            byid={r['scene_id']:r for r in raw};rows=[]
            for sid,m in sorted(meta.items()):
                r=byid.get(sid,{})
                rows.append(normalize({'scene_id':sid,'group':m.get('group') or m.get('family_id') or sid,'label':m['label'],'prediction':r.get('prediction'),
                                       'score':r.get('logit'),'probability':r.get('probability')}))
            sets[(dataset,'deepdfa',seed)]=rows;inputs[str(p.relative_to(out))]=sha(p)
    if (out/'tensor_rule_receipt.json').exists():
        receipt=read(out/'tensor_rule_receipt.json')
        if sha(out/'tensor_rule_slice_predictions.csv')!=receipt['prediction_sha256']:raise ValueError('Rule predictions changed')
        raw=csv_rows(out/'tensor_rule_slice_predictions.csv');grouped=defaultdict(list)
        for r in raw:grouped[(r['dataset'],r['scene_id'])].append(r)
        for dataset in ['source','mit','formai','diagnostic']:
            records,failed,_=records_for(out,dataset);meta={r['scene_id']:r for r in records};meta.update(failed);rows=[]
            for sid,m in sorted(meta.items()):
                sub=grouped.get((dataset,sid),[]);known=bool(sub) and all(r['prediction'] not in ('',None) for r in sub)
                rows.append(normalize({'scene_id':sid,'group':m.get('group') or m.get('family_id') or sid,'label':m['label'],'prediction':max(int(r['prediction']) for r in sub) if known else None,
                                       'pair_id':m.get('pair_id',''),'numeric_stratum':m.get('numeric_stratum',''),'template_stratum':m.get('template_stratum','')}))
            # Deterministic rule repeated only for paired seed alignment; reported once.
            sets[(dataset,'tensor_rule',0)]=rows
        inputs['tensor_rule_receipt.json']=sha(out/'tensor_rule_receipt.json')
    return sets,inputs

def run(out):
    import numpy as np
    require_freeze(out);sets,inputs=load_predictions(out);request={'freeze_sha256':sha(out/'freeze.json'),'inputs':inputs,'statistics':config()['statistics']}
    dest=out/'analysis'/digest(request);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completion.json').exists():
        prior=read(dest/'completion.json')
        for p,h in prior['artifacts'].items():
            if sha(dest/p)!=h:raise ValueError('Analysis changed')
        return prior
    rows=[];pairs=[];diffs=[];pairdiffs=[];members=[];diagnostic=[];cost=[];complement=[];interactions=[];equivalence=[];overlap=[]
    trainhash={r['input_sha256'] for r in csv_rows(out/'source_manifest.csv') if r['split']=='train'}
    ext=csv_rows(out/'external_information_ledger.csv');overlap_ids=defaultdict(set);overlap_groups=defaultdict(set)
    for r in ext:
        if r['input_sha256'] in trainhash:overlap_ids[r['dataset']].add(r['scene_id']);overlap_groups[r['dataset']].add(r['group'])
    for dataset in ['mit','formai']:overlap.append({'dataset':dataset,'scenes_with_exact_TRAIN_input':len(overlap_ids[dataset]),'base_groups_with_exact_TRAIN_input':len(overlap_groups[dataset]),'semantic_near_clones':'NOT_VERIFIED'})
    for dataset in ['source','mit','formai','diagnostic']:
        subset={k:v for k,v in sets.items() if k[0]==dataset}
        if not subset:continue
        neural=[v for k,v in subset.items() if k[2]!=0];common=set.intersection(*[{r['scene_id'] for r in v if r['status']=='SCORED'} for v in neural]) if neural else set()
        comparables=[v for k,v in subset.items() if k[1]!='tensor_rule'];common_all=set.intersection(*[{r['scene_id'] for r in v if r['status']=='SCORED'} for v in comparables]) if comparables else set()
        available=defaultdict(dict)
        for (_,method,seed),rs in subset.items():
            available[method][seed]=rs
            cohorts=[('own_coverage',rs),('common_neural', [r for r in rs if r['scene_id'] in common]),('common_neural_and_tools',[r for r in rs if r['scene_id'] in common_all])]
            if dataset in ['mit','formai']:cohorts.extend([('exclude_exact_scene',[r for r in rs if r['scene_id'] not in overlap_ids[dataset]]),('exclude_exact_whole_group',[r for r in rs if r['group'] not in overlap_groups[dataset]])])
            for cohort,chosen in cohorts:
                rows.append({'dataset':dataset,'method':method,'seed':seed,'cohort':cohort,**metrics(chosen)})
                if dataset in ('mit','diagnostic'):pairs.append({'dataset':dataset,'method':method,'seed':seed,'cohort':cohort,**paired(chosen,dataset)[0]})
            for r in rs:members.append({'dataset':dataset,'method':method,'seed':seed,'scene_id':r['scene_id'],'group':r['group'],'common_neural':r['scene_id'] in common,'status':r['status']})
            if dataset=='diagnostic':
                strata=defaultdict(list)
                for r in rs:strata[(r.get('numeric_stratum','UNKNOWN'),r.get('template_stratum','UNKNOWN'),r.get('kind','UNKNOWN'),r.get('operation','UNKNOWN'),r.get('depth','UNKNOWN'))].append(r)
                for key,items in strata.items():diagnostic.append({'method':method,'seed':seed,**dict(zip(['numeric','template','kind','operation','depth'],key)),**metrics(items),**paired(items,dataset)[0]})
                eq=defaultdict(list)
                for r in rs:
                    if r.get('equivalence_id') and r['status']=='SCORED':eq[r['equivalence_id']].append(r)
                for eid,items in eq.items():
                    if len(items)<2:continue
                    if len({r['label'] for r in items})!=1:raise ValueError('Invalid equivalence labels')
                    equivalence.append({'method':method,'seed':seed,'equivalence_id':eid,'versions':len(items),'same_decision':len({r['prediction'] for r in items})==1,'all_correct':all(r['prediction']==r['label'] for r in items),'probability_spread':max(r['probability'] for r in items)-min(r['probability'] for r in items) if all(r.get('probability') is not None for r in items) else None})
        interactions.append({'dataset':dataset,**interaction(available,dataset,'f1')})
        if dataset in ['mit','diagnostic']:interactions.append({'dataset':dataset,**interaction(available,dataset,'pair_accuracy')})
        if 'full' in available and set(available['full'])==set(config()['seeds']):
            for method in sorted(available):
                if method=='full':continue
                b=available[method]
                if set(b)=={0}:b={s:b[0] for s in config()['seeds']}
                if set(b)!=set(config()['seeds']):continue
                spec=config()['statistics'];info={'dataset':dataset,'reference':'full','comparator':method}
                diffs.append(info|bootstrap_difference(available['full'],b,spec['bootstrap_seed'],spec['bootstrap_replicates']))
                if dataset in ['mit','diagnostic']:pairdiffs.append(info|bootstrap_pairs(available['full'],b,dataset,spec['bootstrap_seed'],spec['bootstrap_replicates']))
                for seed in config()['seeds']:
                    aa={r['scene_id']:r for r in available['full'][seed]};bb={r['scene_id']:r for r in b[seed]};ids={i for i in aa.keys()&bb.keys() if aa[i]['status']==bb[i]['status']=='SCORED'}
                    only=[aa[i] for i in ids if aa[i]['prediction']==1 and bb[i]['prediction']==0]
                    programs=defaultdict(lambda:[0,0,0])
                    for i in ids:
                        k=aa[i].get('source_case') or aa[i]['group'];old=programs[k];programs[k]=[max(old[0],aa[i]['label']),max(old[1],aa[i]['prediction']),max(old[2],bb[i]['prediction'])]
                    complement.append(info|{'seed':seed,'common_scenes':len(ids),'additional_true_positive':sum(r['label']==1 for r in only),'additional_false_positive':sum(r['label']==0 for r in only),'additional_tp_groups':len({r['group'] for r in only if r['label']==1}),
                                            'common_candidate_program_or_base_groups':len(programs),'additional_positive_candidate_programs':sum(y==1 and a==1 and b==0 for y,a,b in programs.values()),'additional_negative_candidate_programs_flagged':sum(y==0 and a==1 and b==0 for y,a,b in programs.values()),'program_scope':'max of common evaluated candidates; not all vulnerabilities in a program'})
    summaries=[];grouped=defaultdict(list)
    for r in rows:grouped[(r['dataset'],r['method'],r['cohort'])].append(r)
    for key,items in sorted(grouped.items()):
        entry=dict(zip(['dataset','method','cohort'],key));entry.update(seeds=len(items),processed_min=min(r['processed'] for r in items),processed_max=max(r['processed'] for r in items))
        for metric in ['precision','recall','specificity','f1','ap','mcc','auroc','all_positive_f1']:
            vals=[r[metric] for r in items if r[metric] is not None];entry[metric+'_mean']=float(np.mean(vals)) if vals else None;entry[metric+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        summaries.append(entry)
    for p in sorted((out/'training').glob('*/*/completion.json')):
        c=read(p);cost.append({**c['identity'],'status':c['status'],'best_epoch':c['best_epoch'],'completed_epochs':c['completed_epochs'],'budget_exhausted':c['budget_exhausted'],'elapsed_s':c['elapsed_s'],'peak_gpu_bytes':c['peak_gpu_bytes'],'parameters':c['parameters'],'requires_grad_parameters':c['requires_grad_parameters']})
    numeric=[]
    for p in sorted((out/'predictions').glob('*/completion.json')):
        receipt=read(p);req=receipt['request']
        if req['dataset']=='diagnostic':
            numeric.extend({'method':req['variant'],'seed':req['seed'],**r} for r in numeric_summary(csv_rows(p.parent/'numeric_state_observations.csv')))
    tables={'numeric_state_summary.csv':numeric,'metrics_per_seed.csv':rows,'summary.csv':summaries,'pair_metrics.csv':pairs,'paired_f1_bootstrap.csv':diffs,'paired_accuracy_bootstrap.csv':pairdiffs,'membership.csv':members,'boundary_strata.csv':diagnostic,'training_cost.csv':cost,'complementarity.csv':complement,'coupling_interaction.csv':interactions,'equivalent_expression_consistency.csv':equivalence,'TRAIN_exact_overlap.csv':overlap}
    for name,data in tables.items():
        fields=list(dict.fromkeys(k for r in data for k in r));write_csv(dest/name,data,fields or ['status'])
    text=['# TIFS mechanism experiment results','', 'Prediction-only analysis; mean ± sample SD (ddof=1). Group bootstrap uses the same group draw for both methods and all seeds. UNKNOWN is excluded from detection metrics and retained in coverage.','',
          '| Dataset | Method | Cohort | N | F1 mean | SD | All-positive F1 |','|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        fmt=lambda x:'NA' if x is None else f'{x:.4f}'
        text.append(f"| {r['dataset']} | {r['method']} | {r['cohort']} | {r['processed_min']}–{r['processed_max']} | {fmt(r['f1_mean'])} | {fmt(r['f1_sd'])} | {fmt(r['all_positive_f1_mean'])} |")
    text+=['','The source partition is a retrospective strict group split. It is not a previously untouched test set. Source train/test class proportions differ after indivisible groups are isolated; report the all-positive reference and specificity alongside F1.',
           'Diagnostic cells refer to exact normalized slice-template overlap, not certified semantic template novelty. Generated programs are mechanism diagnostics, not real vulnerabilities. Conditional tensor rules are an internal control, not a sound C verifier.',
           'Real vulnerability–fix pairs require independent label and context verification; the feasibility ledger is not confirmatory detection evidence. DeepDFA diagnostic-bank coverage is NA because no matching native CPG has been built for that bank.',
           'Effect intervals are exploratory across multiple controls; do not interpret isolated positive intervals as multiplicity-adjusted significance. Budget-limited training is explicitly flagged in training_cost.csv.']
    (dest/'REPORT.md').write_text('\n'.join(text)+'\n')
    result={'status':'COMPLETE_PREDICTION_ANALYSIS','request':request,'directory':str(dest),'evaluated_method_seed_datasets':len(sets),'artifacts':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
    atomic_json(dest/'completion.json',result,immutable=True);atomic_json(out/'analysis_latest.json',{'directory':str(dest),'completion_sha256':sha(dest/'completion.json')});return result

def selftest():
    rows=[normalize({'scene_id':str(i),'group':str(i//2),'label':y,'prediction':p,'score':s}) for i,(y,p,s) in enumerate([(0,0,-2),(1,1,2),(0,1,1),(1,0,-1)])]
    m=metrics(rows);assert (m['tp'],m['tn'],m['fp'],m['fn'])==(1,1,1,1) and m['f1']==.5 and m['auroc']==.75
    p,_=paired(rows,'mit');assert p['pair_accuracy']==.5 and p['bad_score_above_safe_rate']==.5 and p['mean_bad_minus_safe_score']==1
    tie=[dict(r,score=0) for r in rows];assert paired(tie,'mit')[0]['score_tie_rate']==1
    assert paired([dict(r,score=None) for r in rows],'mit')[0]['score_tie_rate'] is None
    probes=[{'sample_id':'a','layer':1,'literal':False,'operation':3,'dependency_depth':2,'carry_or_borrow_length':3,'integer_exact':v,'effective_bit_accuracy':a,'all_bit_accuracy':a,'literal_max_abs_drift':None} for v,a in [(True,1),(False,.5)]]
    summary=numeric_summary(probes);assert len(summary)==3 and all(r['nodes']==2 and r['integer_exact_rate']==.5 and r['effective_bit_accuracy_mean']==.75 for r in summary)
    literal=dict(probes[0],literal=True,literal_max_abs_drift=.125);assert numeric_summary([literal])[0]['literal_drift_max']==.125
    b=bootstrap_difference({11:rows,29:rows,47:rows},{11:rows,29:rows,47:rows},1,2000);assert b['effect_mean_f1']==b['ci_low']==b['ci_high']==0
    same={m:{s:rows for s in config()['seeds']} for m in ['full','no_h_to_q','no_dynamic_q_to_h','no_both_dynamic']}
    for endpoint in ['f1','pair_accuracy']:
        z=interaction(same,'mit',endpoint);assert z['effect']==z['ci_low']==z['ci_high']==0
    assert metrics([normalize({'scene_id':'x','group':'x','label':1,'prediction':1})])['ap'] is None
    return {'status':'PASS','checks':['known_confusion_and_rank_scores','paired_denominator','paired_seed_cluster_bootstrap_identical_zero','hard_labels_no_rank_metrics','pair_score_order_ties_and_deltas','numeric_probe_grouped_known_answers']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=campaign_root());p.add_argument('--selftest',action='store_true');a=p.parse_args();print(json.dumps(selftest() if a.selftest else run(output_dir(a.out))))
