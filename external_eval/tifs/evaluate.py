"""Single committed score per sample/checkpoint/contract; interrupted inference resumes missing IDs."""
import argparse,fcntl,sqlite3,time
from collections import defaultdict
from .common import *
from .train import stage_dir

def records_for(out,name):
    if name=='source':
        with (out/'source_manifest.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['split']=='test']
        return [r|{'scene_id':r['sample_id'],'label':int(r['label']),'dataset':name,'processable':True} for r in rows],{}, {'manifest_sha256':sha(out/'source_manifest.csv')}
    if name in ['mit','formai']:
        from .datasets import external
        records,scenes,ids,identity=external(name)
        if (out/'external_input_audit.json').exists() and identity!=read(out/'external_input_audit.json')[name]['identity']:raise ValueError('External input identity changed after audit')
        return [r|{'processable':True} for r in records],{s:r for s,r in scenes.items() if s not in ids},identity
    if name=='diagnostic':
        with (out/'diagnostics/ledger.csv').open() as f:ledger=list(csv.DictReader(f))
        records=[];failed={}
        for r in ledger:
            if r['status']!='PASS':failed[r['sample_id']]=r;continue
            if sha(r['result_path'])!=r['result_sha256']:raise ValueError('Diagnostic result changed')
            result=read(r['result_path'])
            if sha(Path(r['result_path']).parent/'tensors.npz')!=result['tensor_sha256']:raise ValueError('Diagnostic tensors changed')
            for s in result['slices']:
                records.append({**s,'scene_id':r['sample_id'],'label':int(r['label']),'group':r['group'],'pair_id':r['pair_id'],
                                'numeric_stratum':r['numeric_stratum'],'template_stratum':r['template_stratum'],'equivalence_id':r['equivalence_id'],
                                'kind':r['kind'],'operation':r['operation'],'depth':int(r['depth']),'dataset':name,'processable':True})
        return records,failed,{'manifest_sha256':sha(out/'diagnostics/manifest.json'),'ledger_sha256':sha(out/'diagnostics/ledger.csv')}
    raise ValueError(name)

def score(out,variant,seed,dataset,device):
    out=output_dir(out);freeze=require_freeze(out);np,torch=setup(seed)
    from .models import make_model
    from .datasets import tensor
    from overflow51.training import load_own_checkpoint
    stage=1 if variant=='vd' else 2;trained=stage_dir(out,variant,seed,stage);completion=read(trained/'completion.json')
    if sha(trained/'best.pt')!=completion['best_sha256']:raise ValueError('Checkpoint changed')
    records,failed,input_identity=records_for(out,dataset)
    request={'freeze_sha256':sha(out/'freeze.json'),'variant':variant,'seed':seed,'dataset':dataset,'input_identity':input_identity,
             'checkpoint_sha256':completion['best_sha256'],'threshold':.5,'prediction_unit':'slice_then_scene_max','device':device,'torch':torch.__version__}
    run_id=digest(request);dest=out/'predictions'/run_id;dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(dest/'request.json',request,immutable=True)
    if (dest/'completion.json').exists():
        prior=read(dest/'completion.json')
        if prior['request']!=request:raise ValueError('Prediction request changed')
        for p,h in prior['artifacts'].items():
            if sha(dest/p)!=h:raise ValueError('Completed predictions changed')
        return prior
    db=sqlite3.connect(dest/'scores.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS score (sample_id TEXT PRIMARY KEY,label INTEGER,scene_id TEXT,logit0 REAL,logit1 REAL,risk REAL,status TEXT,reason TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS numeric_state (sample_id TEXT PRIMARY KEY, observations TEXT NOT NULL)')
    have={r[0] for r in db.execute('SELECT sample_id FROM score')}
    expected={r['sample_id'] for r in records}
    if not have<=expected:raise ValueError('Stale prediction membership')
    pending=[r for r in records if r['sample_id'] not in have]
    if pending:
        model=make_model(variant,stage,seed);ck=load_own_checkpoint(trained/'best.pt');model.load_state_dict(ck['model'],strict=True);del ck
        if dataset in ['mit','diagnostic'] and variant!='vd':
            from mit_numeric_ood import patch_numeric_range
            patch_numeric_range(model,torch)
        model=model.eval().to(device);data=SourceData() if dataset=='source' else None;started=time.monotonic()
        observed={};handles=[]
        if dataset=='diagnostic' and variant!='vd':
            from .observe import summarize
            for layer,mod in enumerate(model.backbone.q_layers):
                handles.append(mod.register_forward_hook(lambda m,a,r,layer=layer:observed.update({layer:r.detach().cpu().numpy()})))
        for at in range(0,len(pending),64):
            batch=pending[at:at+64];valid=[];inputs=[]
            for r in batch:
                x={k:data.a[k][int(r['row'])] for k in KEYS} if data else tensor(r)
                if variant=='vd' and int(np.asarray(x['numbers']).max())>999:
                    db.execute('INSERT INTO score VALUES(?,?,?,?,?,?,?,?)',(r['sample_id'],r['label'],r['scene_id'],None,None,None,'UNKNOWN','VD_NUMERIC_DOMAIN_0_999'))
                else:valid.append(r);inputs.append(x)
            if valid:
                x=stack_inputs(inputs,device)
                if variant=='vd':
                    from mit_baseline_frozen_evaluate import vd_batch
                    x=vd_batch(x,torch)
                with torch.inference_mode():logits=model(x)['logits'];probs=logits.softmax(-1)[:,1]
                if logits.shape!=(len(valid),2) or not logits.isfinite().all():raise FloatingPointError('Invalid prediction')
                for r,pair,p in zip(valid,logits.cpu().tolist(),probs.cpu().tolist()):
                    db.execute('INSERT INTO score VALUES(?,?,?,?,?,?,?,?)',(r['sample_id'],r['label'],r['scene_id'],*pair,p,'SCORED',''))
                if observed:
                    for i,r in enumerate(valid):
                        states=[s for layer,qs in sorted(observed.items()) for s in summarize(inputs[i],qs[i],layer)]
                        db.execute('INSERT INTO numeric_state VALUES(?,?)',(r['sample_id'],json.dumps(states)))
            db.commit()
            if at%1024==0:print(json.dumps({'phase':'inference','variant':variant,'seed':seed,'dataset':dataset,'newly_committed':at+len(batch),'pending_target':len(pending),'elapsed_s':time.monotonic()-started}),flush=True)
    db.row_factory=sqlite3.Row;scores={r['sample_id']:dict(r) for r in db.execute('SELECT * FROM score')}
    states=[{'sample_id':r['sample_id'],**s} for r in db.execute('SELECT * FROM numeric_state ORDER BY sample_id') for s in json.loads(r['observations'])];db.close()
    if set(scores)!=expected:raise ValueError('Incomplete predictions')
    slices=[{**r,**scores[r['sample_id']]} for r in records]
    byscene=defaultdict(list)
    for r in slices:byscene[r['scene_id']].append(r)
    scenes=[]
    for sid,items in sorted(byscene.items()):
        labels={r['label'] for r in items}
        if len(labels)!=1:raise ValueError('Mixed scene labels')
        r=items[0];ok=all(s['status']=='SCORED' for s in items)
        # Match the existing highest probability, stable sample-ID tie policy.
        chosen=max(sorted(items,key=lambda r:r['sample_id']),key=lambda r:r['risk'] if r['risk'] is not None else -1)
        scenes.append({'scene_id':sid,'label':r['label'],'group':r['group'],'status':'SCORED' if ok else 'UNKNOWN',
                       'score':chosen['logit1']-chosen['logit0'] if ok else None,'probability':chosen['risk'] if ok else None,
                       'prediction':int(chosen['risk']>=.5) if ok else None,'required_slices':len(items),
                       'pair_id':r.get('pair_id',''),'equivalence_id':r.get('equivalence_id',''),'numeric_stratum':r.get('numeric_stratum',''),
                       'template_stratum':r.get('template_stratum',''),'source_case':r.get('source_case',''),
                       'kind':r.get('kind',''),'operation':r.get('operation',''),'depth':r.get('depth','')})
    fields=list(scenes[0]) if scenes else ['scene_id','label','group','status','score','probability','prediction']
    for sid,r in sorted(failed.items()):
        row={k:r.get(k) for k in fields};row.update(scene_id=sid,label=int(r['label']),group=r.get('group') or r.get('family_id') or sid,status='UNKNOWN',score=None,probability=None,prediction=None)
        if 'pair_id' in row:row['pair_id']=r.get('pair_id','')
        scenes.append(row)
    write_csv(dest/'scene_predictions.csv',scenes,fields)
    write_csv(dest/'slice_predictions.csv',[{k:r[k] for k in ['sample_id','scene_id','label','group','status','logit0','logit1','risk','reason']} for r in slices])
    write_csv(dest/'numeric_state_observations.csv',states)
    result={'status':'COMPLETE','request':request,'run_id':run_id,'target_scenes':len(scenes),'scored_scenes':sum(r['status']=='SCORED' for r in scenes),
            'new_slice_forwards':len(pending)-sum(scores[r['sample_id']]['status']=='UNKNOWN' for r in pending),
            'artifacts':{p.name:sha(p) for p in [dest/'scene_predictions.csv',dest/'slice_predictions.csv',dest/'numeric_state_observations.csv']}}
    atomic_json(dest/'completion.json',result,immutable=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=campaign_root());p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--dataset',choices=['source','mit','formai','diagnostic'],required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();print(json.dumps(score(a.out,a.variant,a.seed,a.dataset,a.device)),flush=True)
if __name__=='__main__':main()
