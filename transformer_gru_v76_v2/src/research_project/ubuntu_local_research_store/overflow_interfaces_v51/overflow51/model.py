"""Dynamic public event graph + learned binding/path + frozen numerical evidence.

Learned-mass intervals are NOT formal verification bounds. NULL, unknown numeric
relations and incomplete frontend coverage widen uncertainty. No sample target,
teacher, chosen capacity, gold path or gold binding is accepted by forward.
"""
from dataclasses import dataclass, fields
import math
import torch
from torch import nn
from .numeric import ObjectVersion, Access, relation, integer_bits, validate_versions

@dataclass(frozen=True)
class Example:
    token_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    literals: tuple[int | None, ...]
    objects: tuple[ObjectVersion, ...]
    accesses: tuple[Access, ...]
    coverage_complete: bool
    node_text: tuple[tuple[int, ...], ...] = ()

@dataclass
class PublicBatch:
    tokens: torch.Tensor
    text_ids: torch.Tensor
    node_mask: torch.Tensor
    adjacency: torch.Tensor
    bits: torch.Tensor
    number_known: torch.Tensor
    object_nodes: torch.Tensor
    access_nodes: torch.Tensor
    candidates: torch.Tensor
    relations: torch.Tensor
    path_literal: torch.Tensor
    access_mask: torch.Tensor
    complete: torch.Tensor

    def to(self, device):
        return PublicBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

def collate(examples, width_bits=64):
    if not examples: raise ValueError("empty batch")
    b = len(examples); n = max(1, max(len(e.token_ids) for e in examples))
    o = max(1, max(len(e.objects) for e in examples)); a = max(1, max(len(e.accesses) for e in examples))
    t=max(1,max((len(row) for e in examples for row in e.node_text),default=1))
    x = PublicBatch(torch.zeros(b,n,dtype=torch.long),torch.zeros(b,n,t,dtype=torch.long), torch.zeros(b,n,dtype=torch.bool),
        torch.zeros(b,n,n), torch.zeros(b,n,width_bits), torch.zeros(b,n,1),
        torch.zeros(b,o,dtype=torch.long), torch.zeros(b,a,dtype=torch.long),
        torch.zeros(b,a,o,dtype=torch.bool), torch.full((b,a,o),-1,dtype=torch.int8),
        torch.full((b,a),-1,dtype=torch.int8), torch.zeros(b,a,dtype=torch.bool),
        torch.tensor([e.coverage_complete for e in examples],dtype=torch.bool))
    for i,e in enumerate(examples):
        if not e.token_ids or len(e.literals)!=len(e.token_ids): raise ValueError("nonempty aligned nodes required")
        validate_versions(e.objects)
        k=len(e.token_ids);x.tokens[i,:k]=torch.tensor(e.token_ids);x.node_mask[i,:k]=True
        if e.node_text and len(e.node_text)!=k:raise ValueError('text must align to graph nodes')
        for ni,row in enumerate(e.node_text):x.text_ids[i,ni,:len(row)]=torch.tensor(row)
        for src,dst in e.edges:
            if not 0<=src<k or not 0<=dst<k: raise ValueError("edge outside graph")
            x.adjacency[i,dst,src]=1
        for j,value in enumerate(e.literals):
            if value is not None:
                x.bits[i,j]=torch.tensor(integer_bits(value,width_bits));x.number_known[i,j]=1
        for j,obj in enumerate(e.objects):
            if not 0<=obj.node<k: raise ValueError("object node outside graph")
            x.object_nodes[i,j]=obj.node
        for j,acc in enumerate(e.accesses):
            if not 0<=acc.node<k: raise ValueError("access node outside graph")
            x.access_nodes[i,j]=acc.node;x.access_mask[i,j]=True
            if acc.path_literal is not None:x.path_literal[i,j]=int(acc.path_literal)
            for oi in acc.candidate_versions:
                if not 0<=oi<len(e.objects): raise ValueError("candidate outside object-version table")
                x.candidates[i,j,oi]=True;x.relations[i,j,oi]=relation(acc,e.objects[oi])
    return x

class CfC(nn.Module):
    def __init__(self,width):
        super().__init__();self.backbone=nn.Sequential(nn.Linear(width*2,width),nn.Tanh())
        self.a=nn.Linear(width,width);self.b=nn.Linear(width,width)
        self.time_a=nn.Linear(width,width);self.time_b=nn.Linear(width,width)
    def forward(self,x,h,delta=1.):
        z=self.backbone(torch.cat([x,h],-1));gate=torch.sigmoid(self.time_a(z)*delta+self.time_b(z))
        return (1-gate)*self.a(z).tanh()+gate*self.b(z).tanh()

@dataclass(frozen=True)
class ModelConfig:
    vocabulary: int=64
    width: int=128
    steps: int=4
    bits: int=64
    cell: str="cfc"
    arm: str="roles"  # roles / semantic / numeric / roles_free_head

class RoleModel(nn.Module):
    def __init__(self,cfg=ModelConfig()):
        super().__init__();self.cfg=cfg
        if cfg.cell not in {"cfc","gru"} or cfg.arm not in {"roles","semantic","numeric","roles_free_head"}:
            raise ValueError("unregistered model")
        if cfg.width<4 or cfg.steps<1:raise ValueError("positive model capacity required")
        w=cfg.width
        # All arms have identical core capacity; unused-input parameters are disclosed.
        self.word=nn.Embedding(cfg.vocabulary,w,padding_idx=0)
        self.number=nn.Linear(cfg.bits+1,w)
        self.message=nn.Linear(w,w);self.norm=nn.LayerNorm(w)
        self.update=CfC(w) if cfg.cell=="cfc" else nn.GRUCell(w,w)
        self.bind_q=nn.Linear(w,w);self.bind_k=nn.Linear(w,w)
        self.null=nn.Parameter(torch.zeros(w));self.path=nn.Linear(w,1)
        self.free_head=nn.Linear(w,1) if cfg.arm!="roles" else None

    def forward(self,x:PublicBatch,*,readout="soft"):
        if not isinstance(x,PublicBatch):raise TypeError("forward only accepts PublicBatch")
        if readout not in {"soft","hard"}:raise ValueError("unknown readout")
        if self.training and readout=="hard":raise ValueError("hard evaluation does not silently replace training")
        known=x.number_known; num=self.number(torch.cat([x.bits,known],-1))
        h=(self.word(x.tokens) if self.cfg.arm!="numeric" else torch.zeros_like(num))
        if self.cfg.arm!='numeric':
            tm=x.text_ids.ne(0).unsqueeze(-1)
            h=h+(self.word(x.text_ids)*tm).sum(2)/tm.sum(2).clamp_min(1)
        if self.cfg.arm!="semantic": h=h+num
        mask=x.node_mask.unsqueeze(-1);h=h*mask
        adj=x.adjacency*x.node_mask[:,None,:]*x.node_mask[:,:,None]
        for _ in range(self.cfg.steps):
            msg=self.message(torch.bmm(adj,h)/adj.sum(-1,keepdim=True).clamp_min(1))
            h=self.update(msg.reshape(-1,self.cfg.width),h.reshape(-1,self.cfg.width)).reshape_as(h)*mask
        h=self.norm(h)*mask
        ix=torch.arange(h.shape[0],device=h.device)[:,None]
        ah=h[ix,x.access_nodes];oh=h[ix,x.object_nodes]
        candidates=torch.cat([oh,self.null[None,None,:].expand(h.shape[0],1,-1)],1)
        scores=torch.einsum('baw,bow->bao',self.bind_q(ah),self.bind_k(candidates))/math.sqrt(self.cfg.width)
        allowed=torch.cat([x.candidates,torch.ones_like(x.candidates[:,:,:1])],-1)
        scores=scores.masked_fill(~allowed,-torch.inf);weights=scores.softmax(-1)
        if readout=="hard":weights=torch.nn.functional.one_hot(scores.argmax(-1),scores.shape[-1]).to(h.dtype)
        path_logits=self.path(ah).squeeze(-1);q=path_logits.sigmoid()
        q=torch.where(x.path_literal>=0,x.path_literal.to(q.dtype),q)*x.access_mask
        r=torch.cat([x.relations,torch.full_like(x.relations[:,:,:1],-1)],-1)
        lo=(weights*(r==1)).sum(-1)*q
        hi=(weights*(r!=0)).sum(-1)*q
        lower=lo.max(-1).values;upper=hi.sum(-1).clamp(max=1)
        complete=x.complete & x.access_mask.any(-1)
        upper=torch.where(complete,upper,torch.ones_like(upper))
        risk=(lower+upper)/2
        if self.free_head is not None:
            pooled=h.sum(1)/x.node_mask.sum(1,keepdim=True).clamp_min(1)
            risk=self.free_head(pooled).squeeze(-1).sigmoid()
        accepted=(lower>=.5)|(upper<.5)
        if self.cfg.arm in {"semantic","numeric","roles_free_head"}:accepted=torch.ones_like(accepted)
        return {"risk":risk,"lower":lower,"upper":upper,"accepted":accepted,
                "binding_logits":scores,"binding_probability":weights,"path_logits":path_logits,
                "path_probability":q,"hidden":h}

def rule_prediction(example):
    lower=[];upper=[]
    for access in example.accesses:
        states=[relation(access,example.objects[j]) for j in access.candidate_versions]
        if access.path_literal is False:lower.append(0.);upper.append(0.);continue
        if not states:states=[-1]
        lower.append(float(all(s==1 for s in states) and access.path_literal is True))
        upper.append(float(any(s!=0 for s in states)))
    lo=max(lower,default=0.);hi=min(1.,sum(upper)) if upper else 1.
    if not example.coverage_complete:hi=1.
    return {"lower":lo,"upper":hi,"risk":(lo+hi)/2,"accepted":lo>=.5 or hi<.5}
