"""Strict split audit, fresh context caching, matched adapter training and scoring."""
from __future__ import annotations
import argparse, csv, gzip, json, re, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from external_eval.conc_exact_topo_20261114 import campaign as old
from external_eval.conc_exact_topo_20261114.model import ExactFusion, exact_states, GRAPH_FIELDS
from external_eval.tifs import common as t

ROOT = old.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'external_eval/tifs_runs/conc_strict_20261114'
PRIOR = ROOT / 'external_eval/tifs_runs/tifs_mechanism_20261019'
read, sha, save, save_torch = old.read, old.sha, old.save_json, old.save_torch


def cfg(): return read(HERE / 'protocol.json')
def csvread(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def setup(seed=11, device='cpu'):
    old.setup(seed, device)
def identity():
    frozen = read(OUT / 'freeze.json')
    for path, expected in frozen['files'].items(): assert sha(ROOT / path) == expected, path
    return sha(OUT / 'freeze.json')


@torch.no_grad()
def four_round(x):
    values, known = exact_states(x, False)
    rows = torch.arange(values.shape[0], device=values.device)[:, None]
    position = torch.arange(values.shape[1], device=values.device)[None]
    valid = position < x['node_count'][:, None]
    pa, pb = x['parents'].long().unbind(-1)
    a, b = pa.clamp(0, values.shape[1]-1), pb.clamp(0, values.shape[1]-1)
    for _ in range(4):
        av, bv = values[rows, a], values[rows, b]
        ak = known[rows, a] & (pa > 0) & (pa < position)
        bk = known[rows, b] & (pb > 0) & (pb < position)
        copying = valid & x['ops'].eq(2) & ak
        arithmetic = valid & (x['ops'].eq(3) | x['ops'].eq(4)) & ak & bk
        candidate = torch.where(x['ops'].eq(3), av + bv, av - bv)
        arithmetic &= candidate.ge(-2**31) & candidate.lt(2**31)
        values = torch.where(copying, av, torch.where(arithmetic, candidate, values))
        known = known | copying | arithmetic
    return values, known


def access_type(line):
    line = line.strip()
    match = re.search(r'\b(memcpy|memmove|memset|strcpy|strncpy|strcat|strncat|sprintf|snprintf)\s*\(', line)
    if match: return 'call_' + match.group(1)
    if re.fullmatch(r'[A-Za-z_]\w*\s*\[[^\[\]]+\]\s*=\s*[^;]+;', line): return 'simple_array_write'
    if re.fullmatch(r'[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\[[^\[\]]+\]\s*;', line): return 'simple_array_read'
    if re.fullmatch(r'\*\s*(?:[A-Za-z_]\w*|\([^;]+\))\s*=\s*[^;]+;', line): return 'simple_pointer_write'
    return 'other_or_multiple'


def prepare():
    setup(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'freeze.json').exists()
    t.require_freeze(PRIOR)
    manifest = csvread(PRIOR / 'source_manifest.csv'); metadata = t.load_metadata()
    assert len(manifest) == len(metadata) == 117696
    overlap = {}
    for key in ('source_case', 'family_id', 'normalized_slice_sha256', 'input_sha256', 'group'):
        sets = {s:{r[key] for r in manifest if r['split'] == s} for s in ('train','val','test')}
        counts = {a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
        assert not any(counts.values()), (key, counts)
        overlap[key] = counts
    sets = {s:{metadata[int(r['row'])]['exact_slice_sha256'] for r in manifest if r['split']==s}
            for s in ('train','val','test')}
    overlap['exact_slice_sha256'] = {a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(overlap['exact_slice_sha256'].values())
    source = t.SourceData(); groups = {}
    text_path = ROOT / 'external_eval/paper_baselines/deepwukong/data/source_text.jsonl.gz'
    with gzip.open(text_path, 'rt') as f:
        for line in f:
            r = json.loads(line); lines = r['slice_text'].splitlines()
            groups[r['sample_id']] = access_type(lines[r['source_lines'].index(r['key_source_line'])])
    baselines = {}
    for seed in cfg()['seeds']:
        for variant in ['full', 'vd']:
            for stage in ([1,2] if variant=='full' else [1]):
                folder = PRIOR / 'training' / f'{variant}_s{seed}' / f'stage{stage}'
                done = read(folder / 'completion.json')
                assert done['status']=='COMPLETE' and not done['test_used_for_selection']
                assert done['identity']['freeze_sha256']==sha(PRIOR/'freeze.json')
                assert sha(folder/'best.pt')==done['best_sha256']
                baselines[f'{variant}_s{seed}_stage{stage}'] = {'path':str(folder/'best.pt'),'sha256':done['best_sha256'], 'best_epoch':done['best_epoch']}
    totals, files = {}, {}
    from external_eval.numeric_precision_20261114.exact_channel import evaluate_graph
    acceptance_nodes = 0
    for split in ('train','val','test'):
        rows = [r for r in manifest if r['split']==split]
        assert len(rows)==cfg()['counts'][split]
        indices = [int(r['row']) for r in rows]
        data = {k:torch.from_numpy(np.array(source.a[k][indices], copy=True)) for k in GRAPH_FIELDS}
        data['labels']=torch.tensor([int(r['label']) for r in rows]);data['source_row']=torch.tensor(indices)
        assert np.array_equal(data['labels'].numpy(), source.a['labels'][indices])
        for prefix, function in [('exact',exact_states),('four',four_round)]:
            vs, ks = [], []
            for at in range(0,len(indices),4096):
                v,k=function({key:value[at:at+4096] for key,value in data.items() if key in GRAPH_FIELDS})
                vs.append(v);ks.append(k)
            data[prefix+'_values']=torch.cat(vs);data[prefix+'_known']=torch.cat(ks)
        if split=='val':
            for idx in np.random.default_rng(20261114).choice(len(indices),128,replace=False):
                x={k:v[int(idx)].numpy() for k,v in data.items() if k in GRAPH_FIELDS}
                for prefix,rounds in [('exact',None),('four',4)]:
                    for node,s in enumerate(evaluate_graph(x,rounds)):
                        assert bool(data[prefix+'_known'][idx,node])==(s.value is not None)
                        if s.value is not None:assert int(data[prefix+'_values'][idx,node])==s.value
                        acceptance_nodes+=1
        kept=[{'sample_id':r['sample_id'],'source_row':int(r['row']),'label':int(r['label']),
               'group':r['group'],'split':split,'access_group':groups[r['sample_id']]} for r in rows]
        save_torch(OUT/'data'/f'{split}.pt',data);save(OUT/'data'/f'{split}_rows.json',kept)
        for path in [OUT/'data'/f'{split}.pt',OUT/'data'/f'{split}_rows.json']:files[str(path.relative_to(ROOT))]=sha(path)
        totals[split]={'n':len(rows),'positive':int(data['labels'].sum()),'groups':len({r['group'] for r in rows}),
                       'access_groups':dict(Counter(r['access_group'] for r in kept))}
    save(OUT/'preparation.json',{'status':'PASS','overlap':overlap,'counts':totals,'baselines':baselines,
                               'acceptance_nodes':acceptance_nodes,'source_text_sha256':sha(text_path),'files':files,
                               'strict_manifest_sha256':sha(PRIOR/'source_manifest.csv')})
    for p in [HERE/'campaign.py',HERE/'protocol.json',old.HERE/'model.py',OUT/'preparation.json',PRIOR/'freeze.json',PRIOR/'source_manifest.csv']:
        files[str(p.relative_to(ROOT))]=sha(p)
    for item in baselines.values():files[str(Path(item['path']).relative_to(ROOT))]=item['sha256']
    save(OUT/'freeze.json',{'status':'FROZEN_BEFORE_NEW_ADAPTER_TRAINING','files':files,'protocol':cfg()})
    print(json.dumps({'phase':'prepare','status':'PASS','counts':totals,'acceptance_nodes':acceptance_nodes}),flush=True)


def graph(split,device='cpu'):
    return {k:v.to(device) for k,v in torch.load(OUT/'data'/f'{split}.pt',map_location='cpu',weights_only=True).items()}


def load_base(seed,device):
    from external_eval.tifs.models import make_model
    from overflow51.training import load_own_checkpoint
    info=read(OUT/'preparation.json')['baselines'][f'full_s{seed}_stage2']
    assert sha(info['path'])==info['sha256']
    model=make_model('full',2,seed);model.load_state_dict(load_own_checkpoint(info['path'])['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)


def cache(seed,device):
    ident=identity();setup(seed,device);model=load_base(seed,device);source=t.SourceData();started=time.monotonic()
    for split in ('train','val','test'):
        path=OUT/'context'/f'{split}_s{seed}.pt'; receipt=path.with_suffix('.json')
        if receipt.exists():
            r=read(receipt);assert r['freeze']==ident and sha(path)==r['sha256'];continue
        rows=read(OUT/'data'/f'{split}_rows.json');contexts=[];logits=[];captured=[]
        handle=model.backbone.head.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
        with torch.inference_mode():
            for at in range(0,len(rows),cfg()['runtime']['cache_batch_size']):
                indices=[r['source_row'] for r in rows[at:at+cfg()['runtime']['cache_batch_size']]]
                x,y=source.batch(indices,device);captured.clear();out=model(x)
                assert len(captured)==1
                contexts.append(captured[0].half().cpu());logits.append(out['logits'].float().cpu())
                if at%8192==0:print(json.dumps({'phase':'cache','seed':seed,'split':split,'at':at,'n':len(rows)}),flush=True)
        handle.remove()
        payload={'context':torch.cat(contexts),'logits':torch.cat(logits),'source_row':torch.tensor([r['source_row'] for r in rows])}
        assert payload['context'].shape==(len(rows),896)
        save_torch(path,payload);save(receipt,{'freeze':ident,'sha256':sha(path),'n':len(rows),'seed':seed,'elapsed_s':time.monotonic()-started})
    print(json.dumps({'phase':'cache','seed':seed,'status':'COMPLETE'}),flush=True)


def context(split,seed,device):
    path=OUT/'context'/f'{split}_s{seed}.pt';assert sha(path)==read(path.with_suffix('.json'))['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert saved['source_row'].tolist()==[r['source_row'] for r in read(OUT/'data'/f'{split}_rows.json')]
    return saved['context'].to(device),saved['logits'].to(device)


def batch(data,ctx,logits,indices,arm):
    x={k:v[indices] for k,v in data.items() if k in GRAPH_FIELDS}
    if arm=='literal_only':v,k=exact_states(x,False)
    else:
        prefix='four' if arm=='four_round' else 'exact'
        v,k=data[prefix+'_values'][indices],data[prefix+'_known'][indices]
    return x,ctx[indices],logits[indices],v,k


@torch.no_grad()
def infer(model,data,ctx,logits,arm):
    model.eval(); result=[]
    for at in range(0,len(data['labels']),512):result.append(model(*batch(data,ctx,logits,slice(at,at+512),arm))['logits'].cpu())
    return torch.cat(result)


def train(seed,device):
    ident=identity();setup(seed,device);tc=cfg()['training']
    ds={s:graph(s,device) for s in ['train','val']};contexts={s:context(s,seed,device) for s in ds}
    for arm in cfg()['arms']:
        folder=OUT/'training'/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completion.json').exists():
            done=read(folder/'completion.json');assert done['freeze']==ident and sha(folder/'best.pt')==done['checkpoint_sha256'];continue
        assert not (folder/'last.pt').exists(),'Interrupted run requires explicit recovery review'
        setup(seed,device);model=ExactFusion('literal_only' if arm=='literal_only' else 'exact_topo').to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
        rng=torch.Generator().manual_seed(seed+20261114);best=float('inf');best_epoch=stale=steps=0;history=[];start=time.monotonic()
        def validate(epoch):
            nonlocal best,best_epoch,stale
            logits=infer(model,ds['val'],*contexts['val'],arm);metrics=old.metric(logits,ds['val']['labels'].cpu().numpy())
            selected=metrics['ce']<best-tc['min_delta']
            if selected:
                best=metrics['ce'];best_epoch=epoch;stale=0
                save_torch(folder/'best.pt',{'head':model.state_dict(),'freeze':ident,'epoch':epoch,'arm':arm,'seed':seed})
            else:stale+=1
            row={'seed':seed,'arm':arm,'epoch':epoch,'dev':metrics,'selected':selected,'steps':steps,'elapsed_s':time.monotonic()-start}
            history.append(row);save(folder/'history.json',history);print(json.dumps(row),flush=True)
        validate(0)
        for epoch in range(1,tc['max_epochs']+1):
            model.train();order=torch.randperm(len(ds['train']['labels']),generator=rng);seen=0
            for at in range(0,len(order),tc['batch_size']):
                indices=order[at:at+tc['batch_size']].to(device);optimizer.zero_grad(set_to_none=True)
                output=model(*batch(ds['train'],*contexts['train'],indices,arm))
                loss=F.cross_entropy(output['logits'],ds['train']['labels'][indices])+tc['delta_l2_weight']*output['delta'].square().mean()
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),tc['clip_norm'],error_if_nonfinite=True)
                optimizer.step();steps+=1;seen+=len(indices)
            assert seen==cfg()['counts']['train'];validate(epoch)
            save_torch(folder/'last.pt',{'head':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'steps':steps,'rng':rng.get_state(),'freeze':ident})
            if stale>=tc['patience']:break
        save(folder/'completion.json',{'status':'COMPLETE','freeze':ident,'seed':seed,'arm':arm,'best_epoch':best_epoch,
             'best_dev':history[best_epoch]['dev'],'epochs':epoch,'steps':steps,'elapsed_s':time.monotonic()-start,
             'checkpoint_sha256':sha(folder/'best.pt'),'trainable_parameters':sum(p.numel() for p in model.parameters()),
             'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        del model,optimizer;torch.cuda.empty_cache()


def select():
    ident=identity();selected={}
    for seed in cfg()['seeds']:
        for arm in cfg()['arms']:
            folder=OUT/'training'/f'{arm}_s{seed}';done=read(folder/'completion.json')
            assert done['status']=='COMPLETE' and done['freeze']==ident and sha(folder/'best.pt')==done['checkpoint_sha256']
            selected[f'{arm}_s{seed}']={'path':str(folder/'best.pt'),'sha256':done['checkpoint_sha256'],'epoch':done['best_epoch']}
    value={'status':'FROZEN_BEFORE_NEW_TEST_SCORING','freeze':ident,'checkpoints':selected,'test_selection':False}
    if (OUT/'selection.json').exists():assert read(OUT/'selection.json')==value
    else:save(OUT/'selection.json',value)


def evaluate(seed,device):
    ident=identity();selection=read(OUT/'selection.json');assert selection['freeze']==ident;setup(seed,device)
    data=graph('test',device);ctx,base=context('test',seed,device);rows=read(OUT/'data/test_rows.json')
    scores={'original':base.cpu()}
    for arm in cfg()['arms']:
        item=selection['checkpoints'][f'{arm}_s{seed}'];assert sha(item['path'])==item['sha256']
        model=ExactFusion('literal_only' if arm=='literal_only' else 'exact_topo').to(device)
        model.load_state_dict(torch.load(item['path'],map_location='cpu',weights_only=True)['head'])
        scores[arm]=infer(model,data,ctx,base,arm)
    for arm,logits in scores.items():
        path=OUT/'predictions'/f'{arm}_s{seed}.csv';assert not path.exists()
        prob=logits.softmax(-1)[:,1].numpy();delta=(logits[:,1]-logits[:,0]).numpy()
        t.write_csv(path,[{**r,'risk':float(p),'score':float(d),'prediction':int(p>=.5)} for r,p,d in zip(rows,prob,delta)])
        save(path.with_suffix('.json'),{'freeze':ident,'selection':sha(OUT/'selection.json'),'sha256':sha(path),
             'seed':seed,'arm':arm,'metrics':old.metric(logits,data['labels'].cpu().numpy())})
    print(json.dumps({'phase':'evaluate','seed':seed,'status':'COMPLETE'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cache','train','select','evaluate'])
    p.add_argument('--seeds',type=int,nargs='+',default=[11]);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.phase in ['prepare','select']:globals()[a.phase]()
    else:
        for seed in a.seeds:globals()[a.phase](seed,a.device)
