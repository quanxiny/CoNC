"""Fresh strict-TRAIN Word2Vec and official DeepWukong GCN; fixed DEV recipe."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from . import campaign as c

BASE=c.ROOT/'external_eval/baselines/deepwukong'
sys.path.insert(0,str(BASE))
import adapter as a
import train_evaluate as runner
import numpy as np
import torch
from torch_geometric.data import Batch


def run(seed,device,evaluate=False):
    assert os.environ.get('PYTHONHASHSEED')=='0'
    a.runtime(seed,4);c.setup(seed,device);ident=c.identity()
    records=a.load_records('source');split={r['sample_id']:r for r in c.csvread(c.PRIOR/'source_manifest.csv')}
    for r in records:
        s=split[r['sample_id']];assert r['label']==int(s['label']) and r['row']==int(s['row'])
        r['split']=s['split'];r['group']=s['group']
    indices={s:[i for i,r in enumerate(records) if r['split']==s] for s in ['train','val','test']}
    assert {s:len(ix) for s,ix in indices.items()}==c.cfg()['counts']
    folder=c.OUT/'deepwukong'/f's{seed}';folder.mkdir(parents=True,exist_ok=True)
    recipe={'lr':.002,'optimizer':'native Adam','batch_size':64,'max_epochs':50,'patience':10,
            'scheduler_gamma':.95,'clip':5,'selection':'DEV mean-of-batch CE; sample mean also reported',
            'word2vec':'strict TRAIN only, per seed, upstream recipe','precision':'CUDA FP32 deterministic',
            'vendor':a.verify_vendor(),'freeze':ident,'script_sha256':c.sha(__file__),
            'source_input_receipt':c.sha(a.OUTPUT/'data/preparation.json')}
    if (folder/'protocol.json').exists():assert c.read(folder/'protocol.json')==recipe
    else:c.save(folder/'protocol.json',recipe)
    w2v,vocab=runner.embedding(records,folder,seed)
    model=a.make_model(seed,w2v,vocab)
    if not evaluate and not (folder/'acceptance.json').exists():
        small=[records[i] for i in indices['train'][:64]]
        fast=[a.encoded_data(r,vocab) for r in small]
        for r,g in zip(small,fast):
            native=a.official_data(r,vocab)
            assert torch.equal(native.x,g.x) and torch.equal(native.edge_index,g.edge_index)
        model.eval()
        with torch.no_grad():
            cpu=model(Batch.from_data_list(fast));model.to(device)
            gpu=model(Batch.from_data_list(fast).to(device)).cpu()
        error=float((cpu-gpu).abs().max());assert error<1e-4
        model.train();x=Batch.from_data_list(fast).to(device)
        loss=torch.nn.functional.cross_entropy(model(x),torch.tensor([r['label'] for r in small],device=device));loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        c.save(folder/'acceptance.json',{'status':'PASS','native_input_parity':True,'cpu_cuda_max_difference':error,'cuda_backward_finite':True,'eligible_weights':False})
        del model;c.setup(seed,device);model=a.make_model(seed,w2v,vocab)
    model=model.to(device)
    if evaluate:
        done=c.read(folder/'completion.json');assert done['checkpoint_sha256']==c.sha(folder/'best.pt')
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
        selected=indices['test'];model.eval();output=[]
        with torch.inference_mode():
            for at in range(0,len(selected),64):
                subset=[records[i] for i in selected[at:at+64]]
                logits=model(Batch.from_data_list([a.encoded_data(r,vocab) for r in subset]).to(device)).cpu()
                for r,score,risk in zip(subset,logits.tolist(),logits.softmax(-1)[:,1].tolist()):
                    output.append({'sample_id':r['sample_id'],'label':r['label'],'group':r['group'],'prediction':int(risk>=.5),'risk':risk,'score':score[1]-score[0]})
        path=folder/'predictions.csv';assert not path.exists();c.t.write_csv(path,output)
        c.save(folder/'evaluation.json',{'status':'COMPLETE','sha256':c.sha(path),'checkpoint':done['checkpoint_sha256'],'n':len(output)})
        return
    if (folder/'completion.json').exists():
        assert c.read(folder/'completion.json')['checkpoint_sha256']==c.sha(folder/'best.pt');return
    assert not (folder/'last.pt').exists(),'Interrupted run requires reviewed recovery'
    graphs=[a.encoded_data(r,vocab) for r in records]
    opt=model.configure_optimizers();optimizer,scheduler=opt['optimizer'],opt['lr_scheduler']
    best=float('inf');stale=0;history=[];steps=0;started=time.monotonic()
    for epoch in range(1,51):
        order=[indices['train'][i] for i in torch.randperm(len(indices['train'])).tolist()]
        model.train()
        for at in range(0,len(order),64):
            ix=order[at:at+64];x=Batch.from_data_list([graphs[i] for i in ix]).to(device)
            y=torch.tensor([records[i]['label'] for i in ix],device=device)
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x),y)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();steps+=1
        losses=[];weighted=0;model.eval()
        with torch.inference_mode():
            for at in range(0,len(indices['val']),64):
                ix=indices['val'][at:at+64];x=Batch.from_data_list([graphs[i] for i in ix]).to(device)
                y=torch.tensor([records[i]['label'] for i in ix],device=device)
                loss=float(torch.nn.functional.cross_entropy(model(x),y));losses.append(loss);weighted+=loss*len(ix)
        dev=float(np.mean(losses));selected=dev<best
        if selected:
            best=dev;stale=0;best_epoch=epoch
            c.save_torch(folder/'best.pt',{'model':model.state_dict(),'epoch':epoch,'protocol':recipe})
        else:stale+=1
        scheduler.step()
        row={'epoch':epoch,'dev_batch_ce':dev,'dev_sample_ce':weighted/len(indices['val']),'selected':selected,'steps':steps,'elapsed_s':time.monotonic()-started}
        history.append(row);c.save(folder/'history.json',history);print(json.dumps({'seed':seed,**row}),flush=True)
        c.save_torch(folder/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,
                                      'rng':runner.rng_state(torch),'cuda_rng':torch.cuda.get_rng_state(device),'best':best,'stale':stale,'protocol':recipe})
        if stale>=10:break
    c.save(folder/'completion.json',{'status':'COMPLETE','seed':seed,'best_epoch':best_epoch,'epochs':epoch,'steps':steps,'best_dev':best,
            'checkpoint_sha256':c.sha(folder/'best.pt'),'elapsed_s':time.monotonic()-started,'test_used_for_selection':False,
            'embedding_receipt':c.sha(folder/'embedding.json'),'protocol_sha256':c.sha(folder/'protocol.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--evaluate',action='store_true');args=p.parse_args()
    run(args.seed,args.device,args.evaluate)
