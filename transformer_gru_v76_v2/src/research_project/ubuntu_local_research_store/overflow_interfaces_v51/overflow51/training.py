"""Shared convergence policy, explicit auxiliary supervision and exact CPU resume."""
from dataclasses import dataclass, asdict
from pathlib import Path
from hashlib import sha256
import copy, json, os, random
import numpy as np
import torch
from torch.nn import functional as F
from .grouping import digest_json
from .metrics import risk_metrics

@dataclass(frozen=True)
class TrainConfig:
    seed:int=11
    max_epochs:int=100
    patience:int=10
    min_delta:float=1e-4
    lr:float=2e-5
    batch_size:int=64
    weight_decay:float=0.
    grad_clip:float=5.
    scheduler_gamma:float=.95
    positive_weight:float=1.
    binding_weight:float=0.
    path_weight:float=0.
    selection:str="val_loss"

@dataclass
class EarlyStop:
    best:float=float("inf")
    bad_epochs:int=0
    best_epoch:int=-1
    def update(self,value,epoch,config):
        if not np.isfinite(value):raise ValueError("nonfinite validation loss")
        improved=self.best_epoch<0 or value<self.best-config.min_delta
        if improved:self.best=float(value);self.bad_epochs=0;self.best_epoch=epoch
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=config.patience

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def objective(output,y,config,binding=None,path=None):
    y=y.float();p=output['risk'].clamp(1e-7,1-1e-7)
    if 'logits' in output:
        # Preserve original raw-logit CE; no probability clamp in author baselines.
        per_row=F.cross_entropy(output['logits'],y.long(),reduction='none')
        risk=(per_row*torch.where(y>0,config.positive_weight,1.)).mean()
    else:risk=(-config.positive_weight*y*p.log()-(1-y)*torch.log1p(-p)).mean()
    terms={"risk":risk}
    counts={"risk":len(y),"binding":0,"path":0}
    for name,target,weight in (("binding",binding,config.binding_weight),("path",path,config.path_weight)):
        if target is None:
            if weight:raise ValueError(f"{name} auxiliary enabled without independently annotated labels")
            continue
        valid=target>=0;counts[name]=int(valid.sum())
        if valid.any():
            pred=output[name+'_logits'][valid]
            terms[name]=F.cross_entropy(pred,target[valid].long()) if name=='binding' else F.binary_cross_entropy_with_logits(pred,target[valid].float())
    loss=terms['risk']+config.binding_weight*terms.get('binding',0)+config.path_weight*terms.get('path',0)
    return loss,{k:float(v.detach()) for k,v in terms.items()},counts

class Cursor:
    """Own generator, shuffled order and next-row cursor are all checkpointed."""
    def __init__(self,size,seed):
        self.size=size;self.generator=torch.Generator().manual_seed(seed);self.epoch=0
        self.order=torch.randperm(size,generator=self.generator);self.position=0
    def next(self,batch_size):
        if self.position==self.size:
            self.epoch+=1;self.position=0;self.order=torch.randperm(self.size,generator=self.generator)
        idx=self.order[self.position:self.position+batch_size];self.position+=len(idx);return idx
    def state_dict(self):return dict(size=self.size,epoch=self.epoch,order=self.order.clone(),position=self.position,rng=self.generator.get_state())
    def load_state_dict(self,state):
        if state['size']!=self.size:raise ValueError('dataset length changed')
        self.epoch=state['epoch'];self.order=state['order'].clone();self.position=state['position'];self.generator.set_state(state['rng'])

def snapshot(model,optimizer,scheduler,cursor,stop,config,manifest_hash,steps):
    return copy.deepcopy({"version":51,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),"cursor":cursor.state_dict(),"earlystop":asdict(stop),
        "config":asdict(config),"manifest_sha256":manifest_hash,"steps":steps,
        "parameter_signature":[(k,list(p.shape),str(p.dtype)) for k,p in model.named_parameters()],
        "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
               "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}})

def restore(state,model,optimizer,scheduler,cursor,stop,config,manifest_hash):
    signature=[(k,list(p.shape),str(p.dtype)) for k,p in model.named_parameters()]
    if state['version']!=51 or state['config']!=asdict(config) or state['manifest_sha256']!=manifest_hash or state['parameter_signature']!=signature:
        raise ValueError("checkpoint/config/data/architecture mismatch; restore refused before mutation")
    if state['cursor']['size']!=cursor.size:raise ValueError('dataset length mismatch before mutation')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    cursor.load_state_dict(state['cursor'])
    for key,val in state['earlystop'].items():setattr(stop,key,val)
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['steps']

def atomic_checkpoint(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(state,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)
    digest=sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix+'.sha256').write_text(digest+'\n')
    return digest

def load_own_checkpoint(path):
    path=Path(path);expected=path.with_suffix(path.suffix+'.sha256').read_text().strip()
    if sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('checkpoint checksum mismatch')
    # Only locally produced full RNG checkpoints, never external dataset/model pickles.
    return torch.load(path,map_location='cpu',weights_only=False)

@torch.no_grad()
def validate(model,batches,config):
    return evaluate_batches(model,batches)['loss']

@torch.no_grad()
def evaluate_batches(model,batches):
    previous=model.training;model.eval();total=0.;count=0
    targets=[];scores=[];coverage=[]
    try:
        for x,y in batches:
            out=model(x);p=out['risk'].clamp(1e-7,1-1e-7)
            # Unweighted validation loss is comparable across imbalance treatments.
            loss=F.cross_entropy(out['logits'],y.long(),reduction='sum') if 'logits' in out else F.binary_cross_entropy(p,y.float(),reduction='sum')
            total+=float(loss);count+=len(y)
            targets.extend(y.cpu().tolist());scores.extend(out['risk'].cpu().tolist());coverage.extend(out['accepted'].cpu().tolist())
    finally:model.train(previous)
    if not count:raise ValueError('empty validation set')
    return {'loss':total/count,**risk_metrics(targets,scores,coverage)}

def train_epoch(model,optimizer,scheduler,cursor,config,get_batch,*,deadline,clock,log_step):
    """Full epoch, not a fixed-step sufficiency proxy. A deadline is mandatory."""
    model.train()
    while True:
        if clock()>=deadline:return False
        idx=cursor.next(config.batch_size);x,y=get_batch(idx)
        optimizer.zero_grad(set_to_none=True);out=model(x);loss,terms,counts=objective(out,y,config)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip,error_if_nonfinite=True)
        optimizer.step();log_step(cursor.epoch,cursor.position,terms,counts,float(norm))
        if cursor.position==cursor.size:break
    scheduler.step();return True
