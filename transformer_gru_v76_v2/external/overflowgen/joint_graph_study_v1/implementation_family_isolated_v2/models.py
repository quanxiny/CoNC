"""Independent randomly initialized semantic/numeric/coupled models.

No VD tensor/encoder reuse. Derived rule values and labels never enter forward.
q has 32 bit probabilities + known + carry + borrow + wrap probabilities;
width is the declared 32-bit representation, original C signedness is unknown.
All numerical operations are learned from slice labels, not executed calculators.
"""
import torch
from torch import nn
import torch.nn.functional as F
INPUT_KEYS=['tokens','numbers','marks','line_count','key','ops','parents','node_line','condition','unit','known','bits','target','node_count','edges']
class QLayer(nn.Module):
 def __init__(self,d):
  super().__init__();self.route=nn.Linear(d,3);self.op=nn.Embedding(20,16);self.hidden=nn.Linear(d,16)
  self.value=nn.Sequential(nn.Linear(36*3+32,512),nn.GELU(),nn.Linear(512,36))
  # Learned bit-sequence operator. Carry/borrow/wrap are output channels, never truth inputs.
  self.bit_rnn=nn.GRU(2+16,32,batch_first=True);self.bit_out=nn.Linear(32,4)
 def forward(self,h,q,parents,ops,bits,known):
  b,n,_=q.shape;ix=torch.arange(b,device=q.device)[:,None]
  a=q[ix,parents[:,:,0]];c=q[ix,parents[:,:,1]];op=self.op(ops)
  seq=torch.cat([a[:,:,:32].unsqueeze(-1),c[:,:,:32].unsqueeze(-1),op.unsqueeze(2).expand(-1,-1,32,-1)],-1).reshape(b*n,32,18)
  rec,_=self.bit_rnn(seq);pred=self.bit_out(rec).reshape(b,n,32,4).sigmoid()
  proposal=self.value(torch.cat([q,a,c,op,self.hidden(h)],-1)).sigmoid()
  proposal=torch.cat([pred[:,:,:,0],proposal[:,:,32:33],pred[:,:,-1,1:]],-1)
  w=self.route(h).softmax(-1);qnew=w[:,:,:1]*a+w[:,:,1:2]*c+w[:,:,2:]*proposal
  literal=torch.cat([bits,torch.ones_like(known).unsqueeze(-1),torch.zeros_like(bits[:,:,:3])],-1)
  return torch.where(known.bool().unsqueeze(-1),literal,qnew)
class IndependentGraphModel(nn.Module):
 def __init__(self,arm,width=512,layers=4):
  super().__init__();assert arm in ['semantic','numeric','joint'];self.arm=arm;self.width=width;self.depth=layers
  self.op=nn.Embedding(20,width);self.cond=nn.Embedding(4,width);self.unit=nn.Embedding(3,width);self.target=nn.Embedding(2,width);self.pos=nn.Embedding(64,width)
  self.signedness_unknown=nn.Parameter(torch.randn(width)*.02)
  if arm!='numeric':
   self.word=nn.Embedding(50,width,padding_idx=0);self.digit=nn.Embedding(12,32);self.char_rnn=nn.GRU(32,64,batch_first=True);self.char_out=nn.Linear(64,width)
   digits=torch.tensor([[0 if v<100 else v//100+1,0 if v<10 else v//10%10+1,v%10+1] for v in range(1000)])
   self.register_buffer('decimal_digits',digits);self.token_pos=nn.Embedding(32,width);self.line_pos=nn.Embedding(12,width)
   self.line_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,8,4*width,.1,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
  if arm!='semantic':
   self.q_init=nn.Linear(36,width);self.q_layers=nn.ModuleList([QLayer(width) for _ in range(layers)]);self.feedback=nn.ModuleList([nn.Linear(36,width) for _ in range(layers)])
   self.edge_gates=nn.ModuleList([nn.Linear(72,1) for _ in range(layers)])
  self.graph=nn.ModuleList([nn.TransformerEncoderLayer(width,8,4*width,.1,batch_first=True,norm_first=True) for _ in range(layers)])
  self.parents_message=nn.ModuleList([nn.Linear(width*2,width) for _ in range(layers)])
  self.norm=nn.LayerNorm(width);self.head=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,2))
 def forward(self,**x):
  ops=x['ops'].long();B,N=ops.shape;dev=ops.device;mask=torch.arange(N,device=dev)[None,:]>=x['node_count'][:,None];ix=torch.arange(B,device=dev)[:,None]
  h=self.op(ops)+self.cond(x['condition'].long())+self.unit(x['unit'].long())+self.target(x['target'].long())+self.pos(torch.arange(N,device=dev))+self.signedness_unknown
  if self.arm!='numeric':
   toks=x['tokens'].long();_,char=self.char_rnn(self.digit(self.decimal_digits));number=self.char_out(char[0])[x['numbers'].long()]
   emb=self.word(toks);emb=torch.where(x['marks'].bool().unsqueeze(-1),emb,number);emb=emb+self.token_pos(torch.arange(toks.shape[2],device=dev))
   tm=toks.ne(0).float();line=(emb*tm.unsqueeze(-1)).sum(2)/tm.sum(2).clamp_min(1).unsqueeze(-1);line=line+self.line_pos(torch.arange(line.shape[1],device=dev))
   lm=torch.arange(line.shape[1],device=dev)[None,:]>=x['line_count'][:,None];line=self.line_encoder(line,src_key_padding_mask=lm)
   # Existing REACHES/CONTROLS edges clipped to the same input slice, shared across arms.
   adj=x['edges'].gt(0).float();line=line+torch.bmm(adj,line)/adj.sum(2).clamp_min(1).unsqueeze(-1)
   h=h+line[ix,x['node_line'].long()]
  if self.arm!='semantic':
   known=x['known'].float();bits=x['bits'].float();q=torch.cat([bits,known.unsqueeze(-1),torch.zeros_like(bits[:,:,:3])],-1)
   # Unknown nodes explicitly carry uncertainty .5 bit probabilities, never all-zero value.
   q=torch.where(known.bool().unsqueeze(-1),q,torch.full_like(q,.5));h=h+self.q_init(q)
  parents=x['parents'].long()
  rows=x['node_line'].long();adj=x['edges'].gt(0).float();node_adj=adj[ix[:,:,None],rows[:,:,None],rows[:,None,:]]
  h=h+torch.bmm(node_adj,h)/node_adj.sum(2).clamp_min(1).unsqueeze(-1)
  for j,layer in enumerate(self.graph):
   a=h[ix,parents[:,:,0]];c=h[ix,parents[:,:,1]];message=self.parents_message[j](torch.cat([a,c],-1))
   if self.arm!='semantic':
    q=self.q_layers[j](h,q,parents,ops,bits,known);qa=q[ix,parents[:,:,0]]
    gate=self.edge_gates[j](torch.cat([q,qa],-1)).sigmoid();message=message*gate;h=h+self.feedback[j](q)
   h=layer(h+message,src_key_padding_mask=mask)
  h=self.norm(h);valid=(~mask).float();target=x['target'].float()*valid
  pooled=(h*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
  risk=(h*target.unsqueeze(-1)).sum(1)/target.sum(1).clamp_min(1).unsqueeze(-1)
  return self.head(torch.cat([pooled,risk],-1))
