"""Auditable source-only CoNC adapter training and fixed-split evaluation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from external_eval.conc_exact_topo_20261114.model import ExactFusion, CoNCExactTopo, exact_states, GRAPH_FIELDS

HERE = Path(__file__).resolve().parent
OUT = ROOT/'external_eval/tifs_runs/conc_exact_topo_20261114'
OLD = ROOT/'external_eval/tifs_runs/access_numeric_20261027'
CACHE = ROOT/'transformer_gru_v76_v2/external/overflowgen/joint_graph_study_v1'
SPLITS = {'train':'source_train', 'val':'source_dev', 'test':'source_test'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(1024*1024), b''):
            h.update(piece)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(path)


def save_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pt.tmp'); torch.save(value, temp); temp.replace(path)


def config():
    return read(HERE/'protocol.json')


def setup(seed=11, device='cpu'):
    torch.set_num_threads(config()['runtime']['cpu_threads'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(config()['runtime']['gpu_memory_fraction'], device)


def records(split):
    meta = OLD/'manifests'/f'{SPLITS[split]}.json'
    receipt = read(meta)
    path = OLD/receipt['records_path']
    assert sha(path) == receipt['records_sha256']
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == config()['counts'][split]
    assert len({r['sample_id'] for r in rows}) == len(rows)
    return rows, receipt


def prepare():
    setup(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'freeze.json').exists(), 'Preparation cannot overwrite a frozen experiment'
    cfg = config(); arrays = {}
    original_receipt = read(CACHE/'CACHE_RECEIPT.json')
    source_hashes = {}
    for key in GRAPH_FIELDS + ('labels', 'family_split'):
        p = CACHE/(key+'.npy'); expected = original_receipt['cache_hashes'][p.name]
        assert sha(p) == expected, key
        arrays[key] = np.load(p, mmap_mode='r', allow_pickle=False)
        source_hashes[key] = expected
    row_sets, totals, files = {}, {}, {}
    for split, dataset in SPLITS.items():
        rr, receipt = records(split)
        ids = [int(r['source_row']) for r in rr]
        row_sets[split] = set(ids)
        assert np.all(arrays['family_split'][ids] == {'train':0,'val':1,'test':2}[split])
        assert np.array_equal(arrays['labels'][ids], np.array([r['label'] for r in rr]))
        bundle = {k:torch.from_numpy(np.array(arrays[k][ids], copy=True)) for k in GRAPH_FIELDS}
        bundle['labels'] = torch.tensor([r['label'] for r in rr], dtype=torch.long)
        bundle['source_row'] = torch.tensor(ids, dtype=torch.long)
        values, known = [], []
        for at in range(0, len(ids), 4096):
            x = {k:v[at:at+4096] for k,v in bundle.items() if k in GRAPH_FIELDS}
            v, k = exact_states(x)
            values.append(v); known.append(k)
        bundle['exact_values'] = torch.cat(values)
        bundle['exact_known'] = torch.cat(known)
        save_torch(OUT/'data'/f'{split}.pt', bundle)
        # Persist only identifiers/labels for evaluation, never old rule features.
        kept = [{k:r[k] for k in ['sample_id','source_row','label','group','split']} for r in rr]
        save_json(OUT/'data'/f'{split}_rows.json', kept)
        n = bundle['ops'].shape[1]
        valid = torch.arange(n)[None] < bundle['node_count'][:,None]
        comp = valid & ((bundle['ops'] == 2) | (bundle['ops'] == 3) | (bundle['ops'] == 4))
        lit = valid & bundle['ops'].eq(1) & bundle['known'].bool()
        totals[split] = {'slices':len(ids), 'valid_nodes':int(valid.sum()), 'literal_nodes':int(lit.sum()),
                         'supported_computation_nodes':int(comp.sum()),
                         'resolved_computation_nodes':int((comp & bundle['exact_known']).sum()),
                         'unresolved_computation_nodes':int((comp & ~bundle['exact_known']).sum()),
                         'slices_with_resolved_computation':int((comp & bundle['exact_known']).any(1).sum())}
        for p in [OUT/'data'/f'{split}.pt', OUT/'data'/f'{split}_rows.json']:
            files[str(p.relative_to(OUT))] = sha(p)
        print(json.dumps({'phase':'prepare', 'split':split, **totals[split]}), flush=True)
    assert not row_sets['train'] & row_sets['val']
    assert not row_sets['train'] & row_sets['test']
    assert not row_sets['val'] & row_sets['test']
    assert len(set.union(*row_sets.values())) == 117696
    save_json(OUT/'preparation.json', {'status':'COMPLETE', 'source_array_hashes':source_hashes,
                                     'totals':totals, 'files':files,
                                     'historical_split_limit':'known family split, not new independent holdout'})


def load_graph(split, device='cpu'):
    p = OUT/'data'/f'{split}.pt'
    assert sha(p) == read(OUT/'preparation.json')['files'][str(p.relative_to(OUT))]
    return {k:v.to(device) for k,v in torch.load(p, weights_only=True, map_location='cpu').items()}


def load_context(split, seed, device='cpu'):
    """Read only the original context and original logits from verified caches."""
    rr, _ = records(split)
    path = OLD/'cache'/SPLITS[split]/f'seed{seed}'
    done = read(path/'completion.json')
    assert done['slices'] == len(rr)
    assert done['identity']['seed'] == seed
    assert done['identity']['manifest'] == sha(OLD/'manifests'/f'{SPLITS[split]}.json')
    assert done['identity']['freeze'] == sha(OLD/'freeze.json')
    expected_models = read(OLD/'assets.json')['models']
    assert expected_models == read(OUT/'base_models.json') if (OUT/'base_models.json').exists() else True
    contexts, logits, at = [], [], 0
    for chunk in done['chunks']:
        p = path/chunk['file']; assert sha(p) == chunk['sha256']
        assert chunk['at'] == at
        batch = torch.load(p, map_location='cpu', weights_only=True)
        expected = digest({'identity':done['identity'], 'records':rr[at:at+chunk['n']]})
        assert batch['sub_id'] == chunk['sub_id'] == expected
        contexts.append(batch['features']['pooled_h'])
        logits.append(batch['features']['base_logits'])
        at += chunk['n']
    assert at == len(rr)
    return torch.cat(contexts).to(device), torch.cat(logits).to(device)


def acceptance():
    setup()
    from external_eval.numeric_precision_20261114.exact_channel import evaluate_graph
    data = load_graph('val')
    # Verify batched integer implementation against the independent ripple-carry implementation.
    rng = np.random.default_rng(20261114)
    chosen = rng.choice(len(data['labels']), 512, replace=False)
    compared = 0
    for idx in chosen:
        graph = {k:v[int(idx)].numpy() for k,v in data.items() if k in GRAPH_FIELDS}
        states = evaluate_graph(graph)
        for i,s in enumerate(states):
            assert bool(data['exact_known'][idx,i]) == (s.value is not None)
            if s.value is not None:
                assert int(data['exact_values'][idx,i]) == s.value
            compared += 1
    # Boundary cases include missing operands and signed overflow.
    ops = torch.tensor([[0,1,1,3,4,2,3], [0,1,1,3,4,2,3]])
    bits = torch.zeros(2,7,32, dtype=torch.long)
    for row,a,b in [(0,2**31-1,1),(1,0,1)]:
        for i,value in [(1,a),(2,b)]:
            bits[row,i] = torch.tensor([(value>>j)&1 for j in range(32)])
    toy = {'ops':ops, 'parents':torch.tensor([[[0,0],[0,0],[0,0],[1,2],[1,2],[4,0],[5,0]]]*2),
           'bits':bits,'known':ops.eq(1),'node_count':torch.tensor([7,7])}
    v,k = exact_states(toy)
    assert not k[0,3] and k[1,3] and v[1,3]==1 and v[1,4]==-1 and v[1,5]==-1
    assert not k[:,6].any()
    ctx, base_logits = load_context('val',11)
    sample = {key:val[:8] for key,val in data.items() if key in GRAPH_FIELDS}
    head = ExactFusion(); control = ExactFusion('literal_only')
    assert sum(p.numel() for p in head.parameters()) == sum(p.numel() for p in control.parameters())
    out = head(sample,ctx[:8],base_logits[:8])
    assert torch.equal(out['logits'], base_logits[:8])
    loss = F.cross_entropy(out['logits'],data['labels'][:8]);loss.backward()
    assert head.fusion[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    # Actual original checkpoint parity, then frozen wrapper parity.
    from external_eval.boundary_optimization.common import load_model
    from external_eval.tifs.common import SourceData
    base, identity = load_model(11,'cuda:0'); base.eval().requires_grad_(False)
    source = SourceData()
    x,_ = source.batch(data['source_row'][:8].tolist(),'cuda:0')
    wrapper = CoNCExactTopo(base).to('cuda:0').eval()
    with torch.inference_mode():
        raw = base(x)['logits']; output = wrapper(x)
    assert torch.equal(raw, output['logits'])
    cache_difference = float((raw.cpu()-base_logits[:8]).abs().max())
    assert torch.equal(raw.argmax(-1).cpu(), base_logits[:8].argmax(-1))
    assert cache_difference <= 2e-5
    del wrapper, base; torch.cuda.empty_cache()
    save_json(OUT/'acceptance.json', {'status':'PASS','independent_graph_nodes_compared':compared,
                                    'model_parameter_count':sum(p.numel() for p in head.parameters()),
                                    'zero_correction_live_logits_exact':True,
                                    'historical_cache_logit_max_abs_difference':cache_difference,
                                    'unknown_overflow_and_chain_checks':True,
                                    'base_seed11_identity':identity})
    print(json.dumps(read(OUT/'acceptance.json')), flush=True)


def freeze():
    assert read(OUT/'acceptance.json')['status']=='PASS'
    assert not (OUT/'freeze.json').exists()
    base = read(OLD/'assets.json')['models']
    package = ROOT/'transformer_gru_v76_v2'
    for r in base.values():
        assert sha(package/r['checkpoint']) == r['checkpoint_sha256']
    save_json(OUT/'base_models.json',base)
    files = {str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
    files[str((HERE/'protocol.json').relative_to(ROOT))] = sha(HERE/'protocol.json')
    for name in ['preparation.json','acceptance.json','base_models.json']:
        p=OUT/name; files[str(p.relative_to(ROOT))]=sha(p)
    for split in SPLITS:
        for seed in config()['seeds']:
            p=OLD/'cache'/SPLITS[split]/f'seed{seed}'/'completion.json'
            files[str(p.relative_to(ROOT))]=sha(p)
    save_json(OUT/'freeze.json', {'status':'FROZEN_BEFORE_TRAINING','files':files,
                                'config':config(),'test_selection':False})


def verify_freeze():
    frozen = read(OUT/'freeze.json')
    for name,h in frozen['files'].items():
        assert sha(ROOT/name)==h, 'Frozen file changed: '+name
    return sha(OUT/'freeze.json')


def batch(data, context, baseline, ids, arm):
    x = {k:data[k][ids] for k in GRAPH_FIELDS}
    if arm=='exact_topo':
        values, known = data['exact_values'][ids],data['exact_known'][ids]
    else:
        values, known = exact_states(x,False)
    return x, context[ids], baseline[ids], values, known


@torch.no_grad()
def infer(head, data, context, baseline, size=1024):
    head.eval(); outputs=[]
    for start in range(0,len(data['labels']),size):
        args=batch(data,context,baseline,slice(start,start+size),head.arm)
        outputs.append(head(*args)['logits'].cpu())
    return torch.cat(outputs)


def metric(logits, labels):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y=np.asarray(labels,dtype=int); l=logits.detach().cpu().float()
    scores=(l[:,1]-l[:,0]).numpy(); prediction=(l.softmax(-1)[:,1]>=.5).numpy()
    tp=int(((y==1)&prediction).sum());tn=int(((y==0)&~prediction).sum())
    fp=int(((y==0)&prediction).sum());fn=int(((y==1)&~prediction).sum())
    divide=lambda a,b:float(a/b) if b else 0.
    return {'n':len(y),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'f1':divide(2*tp,2*tp+fp+fn),'accuracy':divide(tp+tn,len(y)),
            'precision':divide(tp,tp+fp),'recall':divide(tp,tp+fn),'specificity':divide(tn,tn+fp),
            'mcc':divide(tp*tn-fp*fn,((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5),
            'auroc':float(roc_auc_score(y,scores)),'ap':float(average_precision_score(y,scores)),
            'ce':float(F.cross_entropy(l,torch.from_numpy(y)))}


def train(seed,device):
    identity=verify_freeze();setup(seed,device)
    cfg=config()['training']
    datasets={s:load_graph(s,device) for s in ['train','val']}
    contexts={s:load_context(s,seed,device) for s in ['train','val']}
    for arm in config()['arms']:
        folder=OUT/'training'/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completion.json').exists():
            r=read(folder/'completion.json');assert r['freeze']==identity and sha(folder/'best.pt')==r['checkpoint_sha256']
            continue
        assert not (folder/'last.pt').exists(), 'Unfinished training needs audited resume'
        setup(seed,device);head=ExactFusion(arm).to(device)
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        shuffle=torch.Generator().manual_seed(seed+20261114)
        history=[];best=float('inf');best_epoch=0;stale=0;steps=0;start=time.monotonic()
        def validate(epoch):
            nonlocal best,best_epoch,stale
            logits=infer(head,datasets['val'],*contexts['val'])
            metrics=metric(logits,datasets['val']['labels'].cpu().numpy())
            selected=metrics['ce'] < best-cfg['min_delta']
            if selected:
                best=metrics['ce'];best_epoch=epoch;stale=0
                save_torch(folder/'best.pt',{'head':head.state_dict(),'seed':seed,'arm':arm,'epoch':epoch,'freeze':identity})
            else:
                stale+=1
            result={'seed':seed,'arm':arm,'epoch':epoch,'steps':steps,'dev':metrics,'selected':selected,'elapsed_s':time.monotonic()-start}
            history.append(result);save_json(folder/'history.json',history)
            print(json.dumps(result),flush=True)
        validate(0)
        for epoch in range(1,cfg['max_epochs']+1):
            head.train();order=torch.randperm(len(datasets['train']['labels']),generator=shuffle)
            for at in range(0,len(order),cfg['batch_size']):
                ids=order[at:at+cfg['batch_size']].to(device)
                args=batch(datasets['train'],*contexts['train'],ids,arm)
                optimizer.zero_grad(set_to_none=True)
                out=head(*args)
                loss=F.cross_entropy(out['logits'],datasets['train']['labels'][ids])
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),cfg['clip_norm'],error_if_nonfinite=True)
                optimizer.step();steps+=1
            validate(epoch)
            save_torch(folder/'last.pt',{'head':head.state_dict(),'optimizer':optimizer.state_dict(),
                                        'shuffle_rng':shuffle.get_state(),'torch_rng':torch.get_rng_state(),
                                        'cuda_rng':torch.cuda.get_rng_state(device),'epoch':epoch,'steps':steps,'freeze':identity})
            if stale>=cfg['patience']:break
        save_json(folder/'completion.json',{'status':'COMPLETE','seed':seed,'arm':arm,'freeze':identity,
                                           'best_epoch':best_epoch,'epochs':epoch,'steps':steps,'best_dev':history[best_epoch]['dev'],
                                           'checkpoint_sha256':sha(folder/'best.pt'),
                                           'trainable_parameters':sum(p.numel() for p in head.parameters()),
                                           'elapsed_s':time.monotonic()-start,'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        del head,optimizer;torch.cuda.empty_cache()


def select():
    identity=verify_freeze(); checkpoints={}
    for seed in config()['seeds']:
        for arm in config()['arms']:
            folder=OUT/'training'/f'{arm}_s{seed}';done=read(folder/'completion.json')
            assert done['freeze']==identity and done['status']=='COMPLETE'
            assert done['checkpoint_sha256']==sha(folder/'best.pt')
            checkpoints[f'{arm}_s{seed}']=done['checkpoint_sha256']
    save_json(OUT/'selection.json',{'status':'ALL_DEV_CHECKPOINTS_FROZEN','checkpoints':checkpoints,'test_selection':False})


def evaluate(seed,device):
    identity=verify_freeze();selection=read(OUT/'selection.json');setup(seed,device)
    data=load_graph('test',device);context,baseline=load_context('test',seed,device)
    rr=read(OUT/'data/test_rows.json');labels=data['labels'].cpu().numpy()
    with (ROOT/'paper/results/revision/comparison_common_ids.csv').open() as f:
        common={r['scene_id'] for r in csv.DictReader(f) if r['cohort']=='source_common11342'}
    mask=np.array([r['sample_id'] in common for r in rr]);assert mask.sum()==11342
    # Baseline class parity against the original paper's committed full TEST predictions.
    candidates=list((ROOT/'external_eval/paper_frozen/runs').glob(f'overflowgen_v70_s{seed}_*/predictions.csv'))
    assert len(candidates)==1
    with candidates[0].open() as f:old={r['sample_id']:r for r in csv.DictReader(f)}
    assert set(old)=={r['sample_id'] for r in rr}
    base_pred=(baseline.softmax(-1)[:,1]>=.5).cpu().numpy()
    assert all(int(old[r['sample_id']]['prediction'])==int(p) and int(old[r['sample_id']]['label'])==r['label'] for r,p in zip(rr,base_pred))
    bags={'original':baseline.cpu()}
    for arm in config()['arms']:
        p=OUT/'training'/f'{arm}_s{seed}'/'best.pt'
        assert sha(p)==selection['checkpoints'][f'{arm}_s{seed}']
        state=torch.load(p,map_location='cpu',weights_only=True)
        assert state['freeze']==identity
        head=ExactFusion(arm).to(device);head.load_state_dict(state['head'])
        bags[arm]=infer(head,data,context,baseline)
    metrics=[]
    for arm,logits in bags.items():
        for subset,keep in [('full',np.ones(len(rr),dtype=bool)),('common11342',mask)]:
            metrics.append({'seed':seed,'arm':arm,'subset':subset,**metric(logits[keep],labels[keep])})
        p=logits.softmax(-1)[:,1].numpy();score=(logits[:,1]-logits[:,0]).numpy()
        path=OUT/'predictions'/f'{arm}_s{seed}.csv';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['sample_id','group','label','risk','score','prediction','common11342'])
            writer.writeheader()
            for r,pr,sc,cm in zip(rr,p,score,mask):
                writer.writerow({'sample_id':r['sample_id'],'group':r['group'],'label':r['label'],
                                 'risk':float(pr),'score':float(sc),'prediction':int(pr>=.5),'common11342':bool(cm)})
        save_json(path.with_suffix('.json'),{'sha256':sha(path),'seed':seed,'arm':arm,'freeze':identity,'selection':sha(OUT/'selection.json')})
    save_json(OUT/'predictions'/f'metrics_s{seed}.json',metrics)
    print(json.dumps({'seed':seed,'metrics':metrics}),flush=True)


def analyze():
    verify_freeze();cfg=config();metrics=[];bags={}
    for seed in cfg['seeds']:
        metrics.extend(read(OUT/'predictions'/f'metrics_s{seed}.json'))
        for arm in ['original']+cfg['arms']:
            p=OUT/'predictions'/f'{arm}_s{seed}.csv'
            assert sha(p)==read(p.with_suffix('.json'))['sha256']
            with p.open() as f:bags[(seed,arm)]=list(csv.DictReader(f))
    summary=[]
    for subset in ['full','common11342']:
        for arm in ['original']+cfg['arms']:
            rs=[r for r in metrics if r['subset']==subset and r['arm']==arm]
            item={'subset':subset,'arm':arm,'n':rs[0]['n']}
            for key in ['f1','accuracy','precision','recall','specificity','mcc','auroc','ap','ce']:
                item[key+'_mean']=float(np.mean([r[key] for r in rs]))
                item[key+'_sd']=float(np.std([r[key] for r in rs],ddof=1))
            summary.append(item)
    intervals=[]
    for subset in ['full','common11342']:
        ref=bags[(11,'original')]
        keep=np.array([subset=='full' or r['common11342']=='True' for r in ref])
        groups=sorted({r['group'] for r,k in zip(ref,keep) if k});lookup={g:i for i,g in enumerate(groups)}
        counts={}
        for arm in ['original']+cfg['arms']:
            all_seeds=[]
            for seed in cfg['seeds']:
                rs=bags[(seed,arm)];assert [r['sample_id'] for r in rs]==[r['sample_id'] for r in ref]
                c=np.zeros((len(groups),4),dtype=np.int64)
                for r,k in zip(rs,keep):
                    if not k:continue
                    y,p=int(r['label']),int(r['prediction']);slot=0 if y and p else 1 if not y and not p else 2 if p else 3
                    c[lookup[r['group']],slot]+=1
                all_seeds.append(c)
            counts[arm]=np.stack(all_seeds)
        def f1(c):
            tp,tn,fp,fn=np.moveaxis(c,-1,0)
            return 2*tp/np.maximum(1,2*tp+fp+fn)
        rng=np.random.default_rng(cfg['evaluation']['bootstrap_seed']);deltas={b:[] for b in ['original','literal_only']}
        for _ in range(cfg['evaluation']['bootstrap_replicates']):
            draw=rng.integers(0,len(groups),len(groups))
            score={arm:f1(c[:,draw].sum(1)).mean() for arm,c in counts.items()}
            for other in deltas:deltas[other].append(score['exact_topo']-score[other])
        for other,ds in deltas.items():
            point=float(f1(counts['exact_topo'].sum(1)).mean()-f1(counts[other].sum(1)).mean())
            intervals.append({'subset':subset,'comparison':'exact_topo minus '+other,'families':len(groups),
                              'f1_difference':point,'ci95':np.quantile(ds,[.025,.975]).tolist(),
                              'replicates':len(ds),'unit':'paired family resampling, mean over seeds'})
    save_json(OUT/'results.json',{'status':'COMPLETE','per_seed':metrics,'summary':summary,'bootstrap':intervals,
                                'numerical_coverage':read(OUT/'preparation.json')['totals'],
                                'limitations':cfg['interpretation'],'selection':read(OUT/'selection.json')})
    print(json.dumps({'summary':summary,'bootstrap':intervals}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','acceptance','freeze','train','select','evaluate','analyze'])
    p.add_argument('--seeds',type=int,nargs='+',default=[11]);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.phase=='train':
        for seed in a.seeds:train(seed,a.device)
    elif a.phase=='evaluate':
        for seed in a.seeds:evaluate(seed,a.device)
    else:globals()[a.phase]()


if __name__=='__main__':main()
