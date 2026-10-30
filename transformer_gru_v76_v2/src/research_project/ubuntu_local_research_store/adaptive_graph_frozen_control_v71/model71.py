"""Only the implemented update equation changes; no new network module."""
from common71 import V70
from model70 import AdaptiveGraph70,FROZEN_PREFIXES

ARMS71={'residual_transfer_frozen':('residual',True,True)}


class FrozenResidual71(AdaptiveGraph70):
    def __init__(self,arm='residual_transfer_frozen'):
        if arm not in ARMS71:raise ValueError(arm)
        # Exactly the v70 liquid-frozen construction/initialization/freeze set.
        super().__init__('liquid_transfer_frozen')
        self.refiner.mode='residual'
        self.arm=arm
