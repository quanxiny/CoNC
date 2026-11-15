#!/usr/bin/env python3
"""Train one strict-split fusion arm using frozen graph and context tensors."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from external_eval.conc_exact_topo_20261114.model import ExactFusion, GRAPH_FIELDS, exact_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(11,29,47), default=11)
    parser.add_argument('--arm', choices=('exact_topo','literal_only','four_round'), default='exact_topo')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-epochs', type=int, default=30)
    a = parser.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit('Output directory must be empty')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = ROOT/'data/current_assets/strict'
    ds = {}
    for split in ('train','val'):
        graph=torch.load(folder/f'data/{split}.pt',map_location='cpu',weights_only=True)
        cache=torch.load(folder/f'context/{split}_s{a.seed}.pt',map_location='cpu',weights_only=True)
        if not torch.equal(graph['source_row'],cache['source_row']): raise ValueError('Input row mismatch')
        ds[split]=({k:v.to(a.device) for k,v in graph.items()},cache['context'].to(a.device),cache['logits'].to(a.device))
    model=ExactFusion('literal_only' if a.arm=='literal_only' else 'exact_topo').to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.01)
    rng=torch.Generator().manual_seed(a.seed+20261114)
    best=float('inf'); stale=0; history=[]
    def batch(split,ix):
        graph,context,logits=ds[split]
        x={k:graph[k][ix] for k in GRAPH_FIELDS}
        if a.arm=='literal_only': values,known=exact_states(x,False)
        else:
            prefix='four' if a.arm=='four_round' else 'exact'
            values,known=graph[prefix+'_values'][ix],graph[prefix+'_known'][ix]
        return model(x,context[ix],logits[ix],values,known),graph['labels'][ix]
    for epoch in range(a.max_epochs+1):
        if epoch:
            model.train()
            order=torch.randperm(len(ds['train'][0]['labels']),generator=rng)
            for i in range(0,len(order),512):
                ix=order[i:i+512].to(a.device);opt.zero_grad(set_to_none=True)
                result,y=batch('train',ix)
                loss=F.cross_entropy(result['logits'],y)+0.05*result['delta'].square().mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True);opt.step()
        model.eval();pred=[]
        with torch.inference_mode():
            for i in range(0,len(ds['val'][0]['labels']),512):
                result,y=batch('val',slice(i,i+512));pred.append(result['logits'].cpu())
        ce=F.cross_entropy(torch.cat(pred),ds['val'][0]['labels'].cpu()).item()
        selected=ce < best-1e-5
        if selected:
            best,stale=ce,0
            torch.save({'head':model.state_dict(),'epoch':epoch,'arm':a.arm,'seed':a.seed},a.output_dir/'best.pt')
        else: stale+=1
        history.append({'epoch':epoch,'dev_cross_entropy':ce,'selected':selected})
        (a.output_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(history[-1]),flush=True)
        if stale>=5:break

if __name__=='__main__': main()
