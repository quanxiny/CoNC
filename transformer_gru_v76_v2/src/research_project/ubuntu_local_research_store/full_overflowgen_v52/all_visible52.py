"""All-visible numeric control from the existing v2 design, with pinned lineage.

Keeps lexical positions/type/operator tokens and exact raw integer bits. It is
an independently initialized comparison model, not a published pretrained model.
No new source context or computed relation answer is introduced.
"""
import sys
from pathlib import Path
from hashlib import sha256
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'overflow_interfaces_v51'))
from overflow51.baselines import OVERFLOW_ROOT,RiskBridge
from overflow51.full_data import FullData as BaseData

def classes():
    p=OVERFLOW_ROOT/'joint_graph_study_v1/implementation_family_isolated_v2/models.py'
    raw=p.read_bytes()
    if sha256(raw).hexdigest()!='cfeef58709837e62ef24ea9d882250d9272aaaad4b71f7f2c1d7f98e9ec89101':
        raise ValueError('legacy source hash changed')
    ns={};exec(compile(raw,str(p),'exec'),ns)
    # Local reviewed copy below is the existing v2 forward with integer validation.
    return ns['IndependentGraphModel']

class AllVisible(classes()):
    def __init__(self,width=448):
        super().__init__('numeric',width,4)
        self.word=nn.Embedding(50,width,padding_idx=0)
        self.token_pos=nn.Embedding(32,width);self.line_pos=nn.Embedding(12,width)
        layer=lambda: nn.TransformerEncoderLayer(width,8,4*width,.1,batch_first=True,norm_first=True)
        self.line_encoder=nn.TransformerEncoder(layer(),2,enable_nested_tensor=False)
        self.ordered_token_encoder=nn.TransformerEncoder(layer(),1,enable_nested_tensor=False)
        self.raw_bit_encoder=nn.Sequential(nn.Linear(33,128),nn.GELU(),nn.Linear(128,width))
    def line_inputs(self,x):
        numbers=x['numbers']
        if numbers.dtype!=torch.long:raise TypeError('exact integer values required before bit extraction')
        if ((numbers<0)|(numbers>999)).any():raise ValueError('outside audited domain')
        bits=((numbers.unsqueeze(-1)>>torch.arange(32,device=numbers.device))&1).float()
        numeric=self.raw_bit_encoder(torch.cat([bits,torch.ones_like(bits[...,:1])],-1))
        tokens=x['tokens'].long();valid=tokens.ne(0)
        emb=torch.where(x['marks'].bool().unsqueeze(-1),self.word(tokens),numeric)
        return emb+self.token_pos(torch.arange(tokens.shape[2],device=tokens.device)),valid
    def forward(self,**x):
        ops=x['ops'].long();B,N=ops.shape;dev=ops.device
        mask=torch.arange(N,device=dev)[None,:]>=x['node_count'][:,None];ix=torch.arange(B,device=dev)[:,None]
        h=self.op(ops)+self.cond(x['condition'].long())+self.unit(x['unit'].long())+self.target(x['target'].long())+self.pos(torch.arange(N,device=dev))+self.signedness_unknown
        emb,valid=self.line_inputs(x);_,L,T,D=emb.shape
        pad=~valid.reshape(B*L,T);safe=pad.clone();safe[safe.all(-1),0]=False
        encoded=self.ordered_token_encoder(emb.reshape(B*L,T,D),src_key_padding_mask=safe).reshape(B,L,T,D)
        line=(encoded*valid.unsqueeze(-1)).sum(2)/valid.sum(2).clamp_min(1).unsqueeze(-1)
        line=line+self.line_pos(torch.arange(L,device=dev))
        lm=torch.arange(L,device=dev)[None,:]>=x['line_count'][:,None]
        line=self.line_encoder(line,src_key_padding_mask=lm);adj=x['edges'].gt(0).float()
        line=line+torch.bmm(adj,line)/adj.sum(2).clamp_min(1).unsqueeze(-1);h=h+line[ix,x['node_line'].long()]
        known=x['known'].float();bits=x['bits'].float()
        q=torch.cat([bits,known.unsqueeze(-1),torch.zeros_like(bits[...,:3])],-1)
        q=torch.where(known.bool().unsqueeze(-1),q,torch.full_like(q,.5));h=h+self.q_init(q)
        parents=x['parents'].long();rows=x['node_line'].long()
        node_adj=adj[ix[:,:,None],rows[:,:,None],rows[:,None,:]];h=h+torch.bmm(node_adj,h)/node_adj.sum(2).clamp_min(1).unsqueeze(-1)
        for j,layer in enumerate(self.graph):
            a=h[ix,parents[:,:,0]];b=h[ix,parents[:,:,1]]
            message=self.parents_message[j](torch.cat([a,b],-1))
            q=self.q_layers[j](h,q,parents,ops,bits,known);qa=q[ix,parents[:,:,0]]
            message=message*self.edge_gates[j](torch.cat([q,qa],-1)).sigmoid();h=h+self.feedback[j](q)
            h=layer(h+message,src_key_padding_mask=mask)
        h=self.norm(h);vn=(~mask).float();target=x['target'].float()*vn
        pooled=(h*vn.unsqueeze(-1)).sum(1)/vn.sum(1).clamp_min(1).unsqueeze(-1)
        risk=(h*target.unsqueeze(-1)).sum(1)/target.sum(1).clamp_min(1).unsqueeze(-1)
        return self.head(torch.cat([pooled,risk],-1))

class FullData(BaseData):
    def __init__(self,track,arm):
        if arm!='numeric_all_visible':raise ValueError(arm)
        super().__init__(track,'numeric')
    def batch(self,rows,device='cpu'):
        x,y=super().batch(rows,device)
        if x['numbers'].is_floating_point():raise TypeError('cache must preserve integers')
        x['numbers']=x['numbers'].long()
        return x,y
def make_model(arm,track):
    if arm!='numeric_all_visible':raise ValueError(arm)
    return RiskBridge(AllVisible())
