"""Existing models on the supplied synthetic split, using the portable v76 recipe.

These matched-input comparisons are not native real-data baseline reproductions.
"""
import argparse,sys
sys.dont_write_bytecode=True
from portable_runtime import configure_imports
configure_imports()
import train_v76 as engine
from all_visible52 import AllVisible
from models import IndependentGraphModel
from model70 import AdaptiveGraph70,ARMS as V70_ARMS
from overflow51.baselines import RiskBridge,NativeVD,LegacyPureGraph
from overflow51.overflowgen import original_vd_batch

MODEL_SPECS={'all_visible_scratch':('absent',False,False),
 'all_visible_transfer':('absent',True,False),
 'independent_semantic':('absent',False,False),
 'independent_numeric':('absent',False,False),
 'independent_joint':('absent',False,False),
 'vd_family':('absent',False,False)}
MODEL_SPECS.update({'v70_'+arm:spec for arm,spec in V70_ARMS.items()})

class Baseline(RiskBridge):
 def transfer(self,state):
  if not all(k.startswith('model.') for k in state):raise ValueError('Unexpected source namespace')
  mapped={k[6:]:v for k,v in state.items()};self.model.load_state_dict(mapped,strict=True)
  return {k:k for k in state}
 def policy_receipt(self):return {'refiner_exists':False,'frozen_prefixes':[],'dropout':'Original module dropout retained','comparison_scope':'Selected synthetic inputs; v76 training recipe'}
class V70(AdaptiveGraph70):
 def policy_receipt(self):return {'refiner_exists':True,'arm':self.arm,'dropout':'Original module dropout retained','comparison_scope':'Selected synthetic inputs; v76 training recipe'}
class VD(Baseline):
 def __init__(self):super().__init__(NativeVD('family'))
 def forward(self,x):
  import numpy as np
  cache={key:x[key].detach().cpu().numpy() for key in ['tokens','numbers','marks','line_count','key']}
  inputs=original_vd_batch(cache,np.arange(len(x['tokens'])))
  inputs={key:value.to(x['tokens'].device) for key,value in inputs.items()}
  logits=self.model(**inputs);risk=logits.softmax(-1)[:,1]
  return {'logits':logits,'risk':risk,'accepted':engine.torch.ones_like(risk,dtype=engine.torch.bool)}
def make_model(name):
 if name=='vd_family':return VD()
 if name.startswith('v70_'):return V70(name[4:])
 if name.startswith('all_visible_'):return Baseline(AllVisible())
 if name.startswith('independent_'):return Baseline(LegacyPureGraph(name[12:]))
 raise ValueError(name)
def main():
 ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--model',choices=list(MODEL_SPECS),required=True)
 chosen,remaining=ap.parse_known_args();engine.ARMS={chosen.model:MODEL_SPECS[chosen.model]};engine.make_model=make_model
 sys.argv=[sys.argv[0],'--arm',chosen.model,*remaining];engine.main()
if __name__=='__main__':main()
