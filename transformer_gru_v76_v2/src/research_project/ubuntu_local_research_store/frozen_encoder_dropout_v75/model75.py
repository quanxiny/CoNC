"""A single frozen-encoder dropout policy change; no new model mechanism.

Keep every module's training flag unchanged. Set only the two fixed encoders'
Dropout.p and MultiheadAttention.dropout to zero. This includes attention
probability dropout, which changing nn.Dropout children alone would miss.
Avoid encoder.eval(): that can also select different Transformer kernels.
GRU/graph/head modes and all parameter names/values remain inherited.
"""
from common75 import *
from torch import nn
from model70 import AdaptiveGraph70
POLICIES=('legacy_dropout','fixed_encoder_no_dropout')
ENCODERS=('ordered_token_encoder','line_encoder')
class FrozenEncoder75(AdaptiveGraph70):
    def __init__(self, policy='fixed_encoder_no_dropout'):
        if policy not in POLICIES:raise ValueError(policy)
        super().__init__('liquid_transfer_frozen')
        self.dropout_policy75=policy
        self.dropout_sites75=[]
        for enc_name in ENCODERS:
            enc=getattr(self.backbone,enc_name)
            assert all(not p.requires_grad for p in enc.parameters())
            for name,m in enc.named_modules():
                attr='p' if isinstance(m,nn.Dropout) else 'dropout' if isinstance(m,nn.MultiheadAttention) else None
                if attr is not None:
                    original=float(getattr(m,attr));assert original==.1
                    if policy=='fixed_encoder_no_dropout':setattr(m,attr,0.)
                    self.dropout_sites75.append({'name':f'backbone.{enc_name}.{name}',
                        'attribute':attr,'control':original,'candidate':0.})
        assert len(self.dropout_sites75)==12
    def policy_receipt(self):
        return {'policy':self.dropout_policy75,'sites':self.dropout_sites75,
            'training_flags':{k:m.training for k,m in self.named_modules()},
            'no_eval_override':True,'rng_stream_policy':'Natural PyTorch stream; disabled masks consume fewer random draws. No common-mask causal claim.'}
