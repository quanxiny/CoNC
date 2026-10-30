#!/usr/bin/env python3
"""TRAIN-keyed official four-channel vocabulary and unchanged DeepDFA model."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip,hashlib,json,math,os,platform,random,time
from pathlib import Path
import numpy as np
from native_model import make_model,HERE,ROOT
import torch
import dgl

BASE=ROOT/'external_eval/paper_baselines/deepdfa'
DATA=Path(os.environ.get('OVERFLOW_DATA_DIR', ROOT/'data'))/'baselines/deepdfa'
def packed_manifest(data):
    data=Path(data)
    return data/('manifest.json' if (data/'manifest.json').exists() else 'packing.json')
CHANNELS=['api','datatype','literal','operator']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def key(values,channel,atomic):
    values=values[:1] if channel=='datatype' else sorted(set(values))
    return json.dumps({channel:sorted(set(v if v in atomic else 'UNKNOWN' for v in values))},sort_keys=True)
def fit(graphs):
    nodes={}
    for g in graphs:
        if g.get('family_split',g.get('split'))!='train':continue
        for n in g['nodes']:
            if n['features'] is not None:
                identity=(str(g['source_case']),n['native_id'])
                prior=nodes.setdefault(identity,n['features'])
                if prior!=n['features']:raise ValueError('Node-key collision')
    vocab={'atomic':{},'composite':{},'training_unique_definitions':len(nodes)}
    for channel in CHANNELS:
        c=Counter(v for f in nodes.values() for v in (f[channel][:1] if channel=='datatype' else sorted(set(f[channel]))))
        atomic=[k for k,v in sorted(c.items(),key=lambda t:(-t[1],t[0]))[:1000]]
        c=Counter(key(f[channel],channel,set(atomic)) for f in nodes.values())
        vocab['atomic'][channel]=atomic
        vocab['composite'][channel]={k:i+2 for i,(k,v) in enumerate(sorted(c.items(),key=lambda t:(-t[1],t[0]))[:1000])}
    return vocab

def encode(graph,vocab):
    feats=[]
    for channel in CHANNELS:
        atomic=set(vocab['atomic'][channel]);mapping=vocab['composite'][channel]
        feats.append([0 if n['features'] is None else mapping.get(key(n['features'][channel],channel,atomic),1) for n in graph['nodes']])
    return {'meta':{k:v for k,v in graph.items() if k not in {'nodes','edges'}},'n':len(graph['nodes']),'edges':graph['edges'],'features':feats}
def tensorgraph(record):
    edges=record['edges'];src=[e[0] for e in edges];dst=[e[1] for e in edges]
    graph=dgl.add_self_loop(dgl.graph((src,dst),num_nodes=record['n']))
    for channel,values in zip(CHANNELS,record['features']):graph.ndata['_ABS_DATAFLOW_'+channel]=torch.tensor(values,dtype=torch.int64)
    return graph

def cases(root):
    for p in sorted(Path(root).glob('cases/*/*.json.gz')):
        with gzip.open(p,'rt') as f:x=json.load(f)
        yield p,x

def pack(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'packing.json').exists():raise ValueError('Pack already frozen; do not overwrite')
    graph_records=[];coverage=[];inputs={};counts=Counter()
    for path,case in cases(args.frontend):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        byid={g['sample_id']:g for g in case.get('features',{}).get('graphs',[])}
        for original in case['record']['slices']:
            graph=byid.get(original['sample_id']);ok=graph is not None and bool(graph['nodes'])
            counts['total']+=1;counts['covered']+=ok
            status='accepted' if ok else ('no_native_cfg' if graph else case['status'])
            meta={**original,'dataset':case['record']['dataset'],'coverage':bool(ok),'coverage_status':status}
            coverage.append(meta)
            if ok:
                graph={**graph,**meta};graph['family_split']=original['split'];graph_records.append(graph)
    sourcecounts=Counter(r['split'] for r in coverage if r['dataset']=='source')
    if dict(sourcecounts)!={'train':94179,'val':11940,'test':11577}:raise ValueError('Incomplete source frontend: '+str(sourcecounts))
    vocab=fit(graph_records);dump(out/'vocabulary.json',vocab)
    byscope=defaultdict(list)
    for g in graph_records:byscope[g['dataset']].append(encode(g,vocab))
    for scope,records in byscope.items():torch.save(records,out/(scope+'.pt'))
    with gzip.open(out/'coverage.jsonl.gz','wt') as f:
        for r in coverage:f.write(json.dumps(r,separators=(',',':'))+'\n')
    dump(out/'packing.json',{'status':'COMPLETE','counts':dict(counts),'source_counts':dict(sourcecounts),'frontend':str(Path(args.frontend).resolve()),
                            'input_case_sha256':inputs,'vocabulary_sha256':sha(out/'vocabulary.json'),'protocol_sha256':sha(BASE/'protocol.json'),
                            'artifacts':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'status':'PACK_COMPLETE','counts':dict(counts)}),flush=True)

def validate_packed(data):
    data=Path(data);info=json.loads(packed_manifest(data).read_text())
    if 'protocol_sha256' in info and info['protocol_sha256']!=sha(BASE/'protocol.json'):raise ValueError('Packed protocol mismatch')
    for name,digest in info.get('files', info.get('artifacts', {})).items():
        if sha(data/name)!=digest:raise ValueError('Packed artifact changed: '+name)
    return info

def capture_rng():return {'torch':torch.get_rng_state(),'python':random.getstate(),'numpy':np.random.get_state()}
def restore_rng(state):
    torch.set_rng_state(state['torch']);random.setstate(state['python']);np.random.set_state(state['numpy'])
def validate_completed(out,frozen):
    result=json.loads((out/'completed.json').read_text())
    if result['frozen']!=frozen or not (out/'best.pt').is_file() or sha(out/'best.pt')!=result['best_sha256']:raise ValueError('Completed checkpoint/config hash mismatch')
    checkpoint=torch.load(out/'best.pt',weights_only=False)
    if checkpoint['frozen']!=frozen:raise ValueError('Checkpoint internal contract mismatch')
    return result

def setup(threads):
    runtime=json.loads((BASE/'cpu_runtime.json').read_text())
    if runtime['dtype']!='float32' or runtime['device']!='cpu' or runtime['torch_version']!=torch.__version__ or threads!=runtime['threads']:raise ValueError('Frozen CPU runtime mismatch')
    # Configure the floating-point environment before starting worker pools.
    if not torch.set_flush_denormal(runtime['flush_denormal']):raise ValueError('CPU does not support the frozen denormal policy')
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    if runtime['flush_denormal'] and torch.tensor(1e-40,dtype=torch.float32).item()!=0:raise ValueError('CPU denormal policy did not activate')
    print(json.dumps({'status':'CPU_RUNTIME_VERIFIED','runtime_sha256':sha(BASE/'cpu_runtime.json'),'flush_denormal':runtime['flush_denormal'],'torch_version':torch.__version__,'machine':platform.machine(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}),flush=True)
def seedall(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def batches(indices,size,rng=None):
    indices=list(indices)
    if rng:rng.shuffle(indices)
    for start in range(0,len(indices),size):yield indices[start:start+size]
def metrics(labels,probs):
    y=np.asarray(labels);p=np.asarray(probs)>.5;tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum())
    return {'n':len(y),'exact_probability_0_5':int((np.asarray(probs)==.5).sum()),'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'accuracy':(tp+tn)/max(len(y),1)}
def score(model,graphs,labels,indices,batch_size):
    model.eval();probs=[];ys=[];loss=0
    with torch.no_grad():
        for ids in batches(indices,batch_size):
            logits=model(dgl.batch([graphs[i] for i in ids]),{}).reshape(-1);y=labels[ids]
            loss+=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum').item()
            probs.extend(torch.sigmoid(logits).tolist());ys.extend(y.tolist())
    return loss/max(len(ys),1),probs,ys

def train(args):
    setup(args.threads);seedall(args.seed);data=Path(args.data);out=Path(args.out)/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    validate_packed(data)
    protocol=json.loads((BASE/'protocol.json').read_text());recipe=protocol['training']
    frozen={'protocol_sha256':sha(BASE/'protocol.json'),'packing_sha256':sha(packed_manifest(data)),'vocabulary_sha256':sha(data/'vocabulary.json'),'cpu_runtime_sha256':sha(BASE/'cpu_runtime.json'),
            'training_sha256':sha(__file__),'native_model_sha256':sha(HERE/'native_model.py'),'seed':args.seed}
    if (out/'frozen.json').exists():
        if json.loads((out/'frozen.json').read_text())!=frozen:raise ValueError('Training cache identity changed')
    else:dump(out/'frozen.json',frozen)
    if (out/'completed.json').exists():
        validate_completed(out,frozen);print('TRAIN_COMPLETE_CACHED_VERIFIED',flush=True);return
    records=torch.load(data/'source.pt',weights_only=False);graphs=[tensorgraph(r) for r in records]
    labels=torch.tensor([r['meta']['label'] for r in records],dtype=torch.float32)
    trainids=[i for i,r in enumerate(records) if r['meta']['split']=='train'];devids=[i for i,r in enumerate(records) if r['meta']['split']=='val']
    positives=[i for i in trainids if labels[i]==1];negatives=[i for i in trainids if labels[i]==0]
    if len(negatives)<len(positives):raise ValueError('Official equal-negative sampling needs >= negatives')
    model=make_model();optimizer=torch.optim.Adam(model.parameters(),lr=recipe['lr'],weight_decay=recipe['weight_decay'])
    best=math.inf;start_epoch=0;history=[]
    latest=out/'latest.pt'
    if latest.exists():
        state=torch.load(latest,weights_only=False)
        if state['frozen']!=frozen:raise ValueError('Resume checkpoint contract mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);best=state['best'];start_epoch=state['epoch']+1;history=state['history'];restore_rng(state['rng'])
    start=time.monotonic()
    for epoch in range(start_epoch,recipe['epochs']):
        rng=random.Random(args.seed*100000+epoch);ids=positives+rng.sample(negatives,len(positives));model.train();sumloss=0
        for batch in batches(ids,recipe['batch_size'],rng):
            optimizer.zero_grad(set_to_none=True);logits=model(dgl.batch([graphs[i] for i in batch]),{}).reshape(-1)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels[batch]);loss.backward();optimizer.step();sumloss+=loss.item()*len(batch)
        devloss,prob,y=score(model,graphs,labels,devids,recipe['batch_size'])
        result={'epoch':epoch,'train_loss':sumloss/len(ids),'dev_loss':devloss,'dev':metrics(y,prob),'elapsed_s':time.monotonic()-start}
        history.append(result)
        if devloss<best:
            best=devloss;tmp=out/'best.tmp';torch.save({'model':model.state_dict(),'epoch':epoch,'dev_loss':best,'frozen':frozen},tmp);tmp.replace(out/'best.pt')
        tmp=out/'latest.tmp';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'history':history,'frozen':frozen,'rng':capture_rng()},tmp);tmp.replace(latest)
        dump(out/'history.json',history);print(json.dumps(result),flush=True)
    dump(out/'completed.json',{'status':'COMPLETE','seed':args.seed,'best_dev_loss':best,'best_sha256':sha(out/'best.pt'),'history':history,'frozen':frozen})

def evaluate(args):
    from evaluation import PredictionStore,score_missing,aggregate,digest
    setup(args.threads);data=Path(args.data);out=Path(args.out)/f'seed{args.seed}'
    validate_packed(data)
    frozen=json.loads((out/'frozen.json').read_text());validate_completed(out,frozen)
    if frozen['protocol_sha256']!=sha(BASE/'protocol.json') or frozen['packing_sha256']!=sha(packed_manifest(data)) or frozen['training_sha256']!=sha(__file__) or frozen['native_model_sha256']!=sha(HERE/'native_model.py') or frozen['vocabulary_sha256']!=sha(data/'vocabulary.json') or frozen['cpu_runtime_sha256']!=sha(BASE/'cpu_runtime.json'):raise ValueError('Evaluation training/config contract mismatch')
    checkpoint_digest=sha(out/'best.pt');checkpoint=torch.load(out/'best.pt',weights_only=False);model=make_model();model.load_state_dict(checkpoint['model']);model.eval()
    with gzip.open(data/'coverage.jsonl.gz','rt') as f:coverage=[json.loads(s) for s in f]
    def infer(records):
        with torch.no_grad():
            logits=model(dgl.batch([tensorgraph(r) for r in records]),{}).reshape(-1)
            return logits.tolist(),torch.sigmoid(logits).tolist()
    for scope in [name for name in ['source','mit','formai'] if (data/(name+'.pt')).exists()]:
        dest=out/(scope+'_predictions.jsonl.gz');slice_dest=out/(scope+'_slice_predictions.jsonl.gz');evaluation_path=out/(scope+'_evaluation.json')
        expected=[r for r in coverage if r['dataset']==scope and (scope!='source' or r['split'] in {'val','test'})]
        contract={'checkpoint_sha256':checkpoint_digest,'packing_sha256':sha(packed_manifest(data)),'protocol_sha256':sha(BASE/'protocol.json'),
                  'evaluation_contract_sha256':sha(BASE/'evaluation_contract.json'),'evaluation_code_sha256':sha(HERE/'evaluation.py'),
                  'training_code_sha256':sha(__file__),'scope':scope,'seed':args.seed,'expected_sha256':digest(expected)}
        store=PredictionStore(out/(scope+'_predictions.sqlite'),contract)
        try:
            if evaluation_path.exists():
                previous=json.loads(evaluation_path.read_text())
                if previous['contract']!=contract or any(not (out/name).exists() or sha(out/name)!=value for name,value in previous['artifacts'].items()):raise ValueError('Completed prediction provenance mismatch')
                if store.ids()!={r['sample_id'] for r in expected}:raise ValueError('Completed SQLite ledger is incomplete')
                print(scope+'_EVALUATION_CACHED_VERIFIED',flush=True);continue
            records=torch.load(data/(scope+'.pt'),weights_only=False)
            new=score_missing(store,records,expected,infer,256)
            slices=store.rows();scene_rows=aggregate(expected,slices)
            for destination,rows in [(slice_dest,slices),(dest,scene_rows)]:
                temporary=destination.with_suffix('.tmp')
                with gzip.open(temporary,'wt') as f:
                    for row in rows:f.write(json.dumps({**row,'seed':args.seed,'checkpoint_sha256':checkpoint_digest},separators=(',',':'))+'\n')
                temporary.replace(destination)
            counts={'total':len(scene_rows),'covered':sum(r['coverage'] for r in scene_rows),'unknown':sum(not r['coverage'] for r in scene_rows),
                    'required_slices':len(expected),'scored_slices':sum(r['status']=='scored' for r in slices),'newly_inferred_slices':new,'exact_0_5_slice_probabilities':sum(r.get('probability')==.5 for r in slices),
                    'exact_0_5_scene_probabilities':sum(r.get('probability')==.5 for r in scene_rows)}
            scored=[r for r in scene_rows if r['coverage']];summaries={}
            for split in sorted(set(r['split'] for r in scored)):
                subset=[r for r in scored if r['split']==split];summaries[split]=metrics([r['label'] for r in subset],[r['probability'] for r in subset])
            if scope!='source':
                common=[r for r in scored if r.get('in_common_native')];summaries['frozen_native_common_available']=metrics([r['label'] for r in common],[r['probability'] for r in common])
            dump(evaluation_path,{'scope':scope,'counts':counts,'metrics':summaries,'contract':contract,'checkpoint_sha256':checkpoint_digest,
                                  'prediction_sha256':sha(dest),'slice_prediction_sha256':sha(slice_dest),'artifacts':{p.name:sha(p) for p in [dest,slice_dest]},
                                  'selection':'source DEV minimum BCE; fixed sigmoid>.5','scene_coverage':'all required frozen slices scored; otherwise UNKNOWN',
                                  'max_over_slices':True,'rank_score':'actual raw logit of selected maximum-probability slice; ties choose first ascending sample_id',
                                  'threshold_operator':'>','threshold_value':.5,'exact_0_5_prediction':0,'official_threshold_reference':'code_gnn/models/base_module.py:364 preds > 0.5'})
            print(json.dumps({'scope':scope,'counts':counts,'metrics':summaries}),flush=True)
        finally:store.close()

def pilot(args):
    setup(args.threads);seedall(11)
    x=json.loads((BASE/'native_pilot/native_features.json').read_text());graphs=x['graphs'];v=fit(graphs)
    assert v==fit(list(reversed(graphs)))
    rec=[encode(g,v) for g in graphs if g['nodes']];dg=[tensorgraph(r) for r in rec];model=make_model();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.01)
    labels=torch.tensor([r['meta']['label'] for r in rec],dtype=torch.float32)
    begin=time.monotonic();times=[]
    for n in range(30):
        ids=[(n*256+i)%len(rec) for i in range(256)];tick=time.monotonic();opt.zero_grad(set_to_none=True)
        logits=model(dgl.batch([dg[i] for i in ids]),{}).reshape(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels[ids]);loss.backward();opt.step();times.append(time.monotonic()-tick)
        assert torch.isfinite(loss)
    no=[g for g in graphs if g['nodes'] and not g['definition_count']]
    assert len(no)==4 and all(all(all(vv==0 for vv in channel) for channel in encode(g,v)['features']) for g in no)
    result={'status':'PASS_NATIVE_TRAINING_COST_PILOT','real_train_graphs':len(dg),'threads':args.threads,'batch':256,'updates':30,
            'mean_update_s_excluding_warmup':float(np.mean(times[3:])),'max_update_s':max(times),'wall_s':time.monotonic()-begin,
            'keyed_vocab_order_invariance':True,'no_definition_graphs_retained_all_zero':len(no),'params':sum(p.numel() for p in model.parameters()),
            'official_model_unchanged':True,'pilot_only_no_checkpoint':True}
    dump(BASE/'training_cost_pilot.json',result);print(json.dumps(result),flush=True)

def smoke(args):
    """One forward/backward update on TRAIN graphs, without saving weights."""
    setup(args.threads);seedall(args.seed);data=Path(args.data);validate_packed(data)
    records=torch.load(data/'source.pt',weights_only=False)
    records=[r for r in records if r['meta']['split']=='train'][:8]
    if len(records)!=8:raise ValueError('Expected eight source TRAIN fixtures')
    model=make_model();graph=dgl.batch([tensorgraph(r) for r in records])
    labels=torch.tensor([r['meta']['label'] for r in records],dtype=torch.float32)
    logits=model(graph,{}).reshape(-1)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels);loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite smoke result')
    print(json.dumps({'status':'PASS','graphs':len(records),'loss':float(loss.detach()),'parameters':sum(p.numel() for p in model.parameters()),'weights_saved':False}))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pack','train','evaluate','pilot','smoke']);p.add_argument('--frontend',default=str(BASE/'native_full'))
    p.add_argument('--data',default=str(DATA));p.add_argument('--out',default=str(BASE/'runs'));p.add_argument('--seed',type=int,choices=[11,29,47],default=11);p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    if args.mode=='pack':
        args.out=args.data;pack(args)
    elif args.mode=='train':train(args)
    elif args.mode=='evaluate':evaluate(args)
    elif args.mode=='smoke':smoke(args)
    else:pilot(args)
if __name__=='__main__':main()
