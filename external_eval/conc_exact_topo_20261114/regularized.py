"""DEV-informed confidence-anchored follow-up; original frozen experiment intact."""
from __future__ import annotations
import argparse
import csv
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import campaign as c
from .model import ExactFusion


def cfg():return c.read(c.HERE/'regularization_protocol.json')


def identity():
    return {'original_freeze':c.verify_freeze(), 'extension_code':c.sha(__file__),
            'extension_protocol':c.sha(c.HERE/'regularization_protocol.json')}


def train(seed,device):
    ident=identity();c.setup(seed,device);tc=cfg()['training']
    data={s:c.load_graph(s,device) for s in ['train','val']}
    contexts={s:c.load_context(s,seed,device) for s in ['train','val']}
    for arm in c.config()['arms']:
        folder=c.OUT/'training_regularized'/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completion.json').exists():
            done=c.read(folder/'completion.json');assert done['identity']==ident and done['checkpoint_sha256']==c.sha(folder/'best.pt')
            continue
        assert not (folder/'last.pt').exists()
        c.setup(seed,device);model=ExactFusion(arm).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
        rng=torch.Generator().manual_seed(seed+20261114)
        best=float('inf');best_epoch=0;stale=0;steps=0;history=[];start=time.monotonic()
        def validate(epoch,training=None):
            nonlocal best,best_epoch,stale
            predictions=c.infer(model,data['val'],*contexts['val'])
            metrics=c.metric(predictions,data['val']['labels'].cpu().numpy())
            selected=metrics['ce']<best-tc['min_delta']
            if selected:
                best=metrics['ce'];best_epoch=epoch;stale=0
                c.save_torch(folder/'best.pt',{'head':model.state_dict(),'seed':seed,'arm':arm,'epoch':epoch,'identity':ident})
            else:stale+=1
            row={'seed':seed,'arm':arm,'epoch':epoch,'steps':steps,'dev':metrics,'selected':selected,
                 'train':training,'elapsed_s':time.monotonic()-start}
            history.append(row);c.save_json(folder/'history.json',history);print(json.dumps(row),flush=True)
        validate(0)
        for epoch in range(1,tc['max_epochs']+1):
            model.train();order=torch.randperm(len(data['train']['labels']),generator=rng)
            sums=torch.zeros(3,device=device);seen=0
            for at in range(0,len(order),tc['batch_size']):
                ids=order[at:at+tc['batch_size']].to(device)
                optimizer.zero_grad(set_to_none=True)
                output=model(*c.batch(data['train'],*contexts['train'],ids,arm))
                ce=F.cross_entropy(output['logits'],data['train']['labels'][ids])
                penalty=output['delta'].square().mean()
                loss=ce+tc['delta_l2_weight']*penalty
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),tc['clip_norm'],error_if_nonfinite=True)
                optimizer.step();steps+=1;seen+=len(ids)
                sums+=torch.stack([ce.detach(),penalty.detach(),loss.detach()])*len(ids)
            assert seen==94179
            validate(epoch,dict(zip(['ce','delta_squared','objective'],(sums/seen).cpu().tolist())))
            c.save_torch(folder/'last.pt',{'head':model.state_dict(),'optimizer':optimizer.state_dict(),
                                          'epoch':epoch,'steps':steps,'rng':rng.get_state(),'identity':ident})
            if stale>=tc['patience']:break
        c.save_json(folder/'completion.json',{'status':'COMPLETE','seed':seed,'arm':arm,'identity':ident,
                                             'best_epoch':best_epoch,'best_dev':history[best_epoch]['dev'],
                                             'epochs':epoch,'steps':steps,'elapsed_s':time.monotonic()-start,
                                             'checkpoint_sha256':c.sha(folder/'best.pt'),
                                             'trainable_parameters':sum(p.numel() for p in model.parameters()),
                                             'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        del model,optimizer;torch.cuda.empty_cache()


def select():
    ident=identity();means={};all_done={}
    for recipe,dir_name in [('initial','training'),('regularized','training_regularized')]:
        scores=[]
        for seed in c.config()['seeds']:
            for arm in c.config()['arms']:
                folder=c.OUT/dir_name/f'{arm}_s{seed}';done=c.read(folder/'completion.json')
                assert done['status']=='COMPLETE' and done['checkpoint_sha256']==c.sha(folder/'best.pt')
                if recipe=='regularized':assert done['identity']==ident
                else:assert done['freeze']==ident['original_freeze']
                all_done[(recipe,arm,seed)]=(folder,done)
                if arm=='exact_topo':scores.append(done['best_dev']['ce'])
        means[recipe]=float(np.mean(scores))
    winner='regularized' if means['regularized']<means['initial'] else 'initial'
    points={}
    for arm in c.config()['arms']:
        for seed in c.config()['seeds']:
            folder,done=all_done[(winner,arm,seed)]
            points[f'{arm}_s{seed}']={'path':str(folder/'best.pt'),'sha256':done['checkpoint_sha256'],'epoch':done['best_epoch']}
    c.save_json(c.OUT/'selection.json',{'status':'ALL_DEV_CHECKPOINTS_FROZEN','recipe':winner,
                                      'mean_exact_topo_dev_ce':means,'checkpoints':points,'identity':ident,
                                      'test_selection':False,'follow_up':'DEV-informed confidence-anchored training'})
    print(json.dumps(c.read(c.OUT/'selection.json')),flush=True)


def evaluate(seed,device):
    ident=identity();selection=c.read(c.OUT/'selection.json');assert selection['identity']==ident
    c.setup(seed,device)
    data=c.load_graph('test',device);context,baseline=c.load_context('test',seed,device)
    rr=c.read(c.OUT/'data/test_rows.json');labels=data['labels'].cpu().numpy()
    with (c.ROOT/'paper/results/revision/comparison_common_ids.csv').open() as f:
        common={r['scene_id'] for r in csv.DictReader(f) if r['cohort']=='source_common11342'}
    mask=np.array([r['sample_id'] in common for r in rr]);assert mask.sum()==11342
    historical=list((c.ROOT/'external_eval/paper_frozen/runs').glob(f'overflowgen_v70_s{seed}_*/predictions.csv'))
    assert len(historical)==1
    with historical[0].open() as f:old={r['sample_id']:r for r in csv.DictReader(f)}
    assert set(old)=={r['sample_id'] for r in rr}
    pred=(baseline.softmax(-1)[:,1]>=.5).cpu().numpy()
    assert all(int(old[r['sample_id']]['prediction'])==int(p) and int(old[r['sample_id']]['label'])==r['label'] for r,p in zip(rr,pred))
    bags={'original':baseline.cpu()}
    for arm in c.config()['arms']:
        selected=selection['checkpoints'][f'{arm}_s{seed}'];assert c.sha(selected['path'])==selected['sha256']
        saved=torch.load(selected['path'],map_location='cpu',weights_only=True)
        model=ExactFusion(arm).to(device);model.load_state_dict(saved['head'])
        bags[arm]=c.infer(model,data,context,baseline)
    metrics=[]
    for arm,logits in bags.items():
        for subset,keep in [('full',np.ones(len(rr),dtype=bool)),('common11342',mask)]:
            metrics.append({'seed':seed,'arm':arm,'subset':subset,**c.metric(logits[keep],labels[keep])})
        probability=logits.softmax(-1)[:,1].numpy();score=(logits[:,1]-logits[:,0]).numpy()
        path=c.OUT/'predictions'/f'{arm}_s{seed}.csv';path.parent.mkdir(parents=True,exist_ok=True)
        assert not path.exists(), 'Do not silently replace committed TEST predictions'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['sample_id','group','label','risk','score','prediction','common11342']);w.writeheader()
            for r,p,s,m in zip(rr,probability,score,mask):
                w.writerow({'sample_id':r['sample_id'],'group':r['group'],'label':r['label'],
                            'risk':float(p),'score':float(s),'prediction':int(p>=.5),'common11342':bool(m)})
        c.save_json(path.with_suffix('.json'),{'sha256':c.sha(path),'seed':seed,'arm':arm,'identity':ident,'selection':c.sha(c.OUT/'selection.json')})
    c.save_json(c.OUT/'predictions'/f'metrics_s{seed}.json',metrics)
    print(json.dumps({'seed':seed,'metrics':metrics}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','select','evaluate']);p.add_argument('--seeds',type=int,nargs='+',default=[11]);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.phase=='select':select()
    else:
        for seed in a.seeds:globals()[a.phase](seed,a.device)
