"""Minimal experimental interventions; original AllVisible/v70 forward is imported."""
from .common import imports
imports()
import torch
from torch import nn
from all_visible52 import AllVisible
from model70 import AdaptiveGraph70
from models import QLayer
from overflow51.baselines import RiskBridge, NativeVD

VARIANTS=('full','local_q','no_h_to_q','no_dynamic_q_to_h','symmetric_q','no_clamp','mlp_q','no_both_dynamic')

class ControlledQLayer(QLayer):
    def forward(self,h,q,parents,ops,bits,known):
        mode=self.intervention
        if mode=='no_h_to_q' or mode=='no_both_dynamic': h=torch.zeros_like(h)
        if mode=='symmetric_q':
            return .5*(super().forward(h,q,parents,ops,bits,known)+
                       super().forward(h,q,parents.flip(-1),ops,bits,known))
        if mode not in ('local_q','no_clamp','mlp_q'):return super().forward(h,q,parents,ops,bits,known)
        b,n,_=q.shape;ix=torch.arange(b,device=q.device)[:,None]
        a=q if mode=='local_q' else q[ix,parents[:,:,0]]
        c=q if mode=='local_q' else q[ix,parents[:,:,1]]
        op=self.op(ops)
        if mode=='mlp_q':
            pred=self.bit_mlp(torch.cat([a[...,:32],c[...,:32],op],-1)).sigmoid()
            proposal_bits,flags=pred[...,:32],pred[...,32:]
        else:
            seq=torch.cat([a[...,:32].unsqueeze(-1),c[...,:32].unsqueeze(-1),op.unsqueeze(2).expand(-1,-1,32,-1)],-1).reshape(b*n,32,18)
            rec,_=self.bit_rnn(seq);pred=self.bit_out(rec).reshape(b,n,32,4).sigmoid()
            proposal_bits,flags=pred[:,:,:,0],pred[:,:,-1,1:]
        value=self.value(torch.cat([q,a,c,op,self.hidden(h)],-1)).sigmoid()
        proposal=torch.cat([proposal_bits,value[:,:,32:33],flags],-1)
        w=self.route(h).softmax(-1)
        result=w[:,:,:1]*a+w[:,:,1:2]*c+w[:,:,2:]*proposal
        if mode=='no_clamp':return result
        literal=torch.cat([bits,torch.ones_like(known).unsqueeze(-1),torch.zeros_like(bits[...,:3])],-1)
        return torch.where(known.bool().unsqueeze(-1),literal,result)

def zero_output(module,args,result):return torch.zeros_like(result)

def initial_q(x):
    bits=x['bits'].float();known=x['known'].float()
    q=torch.cat([bits,known.unsqueeze(-1),torch.zeros_like(bits[...,:3])],-1)
    return torch.where(known.bool().unsqueeze(-1),q,torch.full_like(q,.5))

def fixed_refiner_q(module,args):
    h,q,x,valid,*rest=args
    return (h,initial_q(x),x,valid,*rest)

def install(backbone,variant,seed):
    if variant not in VARIANTS:raise ValueError(variant)
    if variant=='full':return
    for j,layer in enumerate(backbone.q_layers):
        layer.__class__=ControlledQLayer
        layer.intervention=variant
        if variant=='mlp_q':
            # 5,139 active operator parameters vs original GRU+head 5,124.
            # Keep common parameter initializations and global RNG identical.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+10000+j)
                layer.bit_mlp=nn.Sequential(nn.Linear(80,44),nn.GELU(),nn.Linear(44,35))
            del layer.bit_rnn;del layer.bit_out
    if variant in ('no_dynamic_q_to_h','no_both_dynamic'):
        for m in backbone.feedback:
            m.register_forward_hook(zero_output)
            m.requires_grad_(False)
        for m in backbone.edge_gates:
            m.register_forward_hook(zero_output) # original sigmoid => fixed gain 0.5
            m.requires_grad_(False)

def make_model(variant,stage,seed):
    if variant=='vd':return RiskBridge(NativeVD('family'))
    if stage==1:
        m=AllVisible();install(m,variant,seed);return RiskBridge(m)
    if stage!=2:raise ValueError(stage)
    m=AdaptiveGraph70('liquid_transfer_frozen');install(m.backbone,variant,seed)
    # Newly installed MLP must obey the original stage2 freeze policy too.
    for layer in m.backbone.q_layers:layer.requires_grad_(False)
    if variant in ('no_dynamic_q_to_h','no_both_dynamic'):
        m.refiner.register_forward_pre_hook(fixed_refiner_q)
    return m

def transfer_stage1(model,state):
    return model.transfer(state)
