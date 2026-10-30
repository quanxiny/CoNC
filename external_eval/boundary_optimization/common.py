from pathlib import Path
import json, hashlib, csv, os
from external_eval.tifs.common import ROOT, PACKAGE, CACHE, KEYS, setup, stack_inputs, atomic_json, write_csv, sha, digest, read, SourceData, canonical_input
HERE=Path(__file__).resolve().parent
OUT=ROOT/'external_eval/tifs_runs/boundary_optimization_20261020'
OLD=ROOT/'external_eval/tifs_runs/tifs_mechanism_20261019'

def config():return read(HERE/'config.json')

def load_model(seed,device='cpu',path=None):
    np,torch=setup(seed)
    from external_eval import paper_frozen_evaluate as p
    run=p.registry_run('v70',seed);identity=p.model_identity(run)
    model=p.build_model(run,torch,np,device)
    if path:
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'],strict=True)
    from external_eval.mit_numeric_ood import patch_numeric_range
    patch_numeric_range(model,torch)
    return model,identity

class Bank:
    def __init__(self,manifest):
        from external_eval.tifs.datasets import tensor
        self.specs=read(manifest)['specs'];self.scenes=[];self.tensor=tensor
        self.manifest=manifest;self.cached={}
        old=Path(manifest).parent
        for s in self.specs:
            rpath=Path(s['result_path']) if 'result_path' in s else old/'cases'/s['sample_id'].split(':')[-1]/'result.json'
            if rpath.exists():
                r=read(rpath)
                if r['status']=='PASS':
                    r.update({k:v for k,v in s.items() if k not in ['source','slices']})
                    self.scenes.append(r)
        self.byid={s['sample_id']:s for s in self.scenes}
    def batch(self,scenes,device):
        items=[];owners=[]
        for i,s in enumerate(scenes):
            for sl in s['slices']:
                key=(sl['tensor_cache'],sl['tensor_row'])
                if key not in self.cached:self.cached[key]=self.tensor(sl)
                items.append(self.cached[key]);owners.append(i)
        return stack_inputs(items,device),owners

def aggregate(logits,owners,n,torch):
    # Preserve max probability and first-slice tie behavior, including FP32 saturation.
    margin=logits[:,1]-logits[:,0];risk=logits.softmax(-1)[:,1]
    chosen=[]
    for i in range(n):
        ids=[j for j,o in enumerate(owners) if o==i]
        winner=ids[int(risk[ids].argmax())]
        chosen.append(margin[winner])
    return torch.stack(chosen)
