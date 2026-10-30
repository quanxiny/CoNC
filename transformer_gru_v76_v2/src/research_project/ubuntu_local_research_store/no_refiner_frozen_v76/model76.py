"""True no-refiner training: the sole child module is the original AllVisible."""
from common76 import torch
from model70 import AllVisible, FROZEN_PREFIXES

class NoRefiner76(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AllVisible()
        for name, parameter in self.backbone.named_parameters():
            if name.startswith(FROZEN_PREFIXES):
                parameter.requires_grad_(False)

    def transfer(self, state):
        target = self.backbone.state_dict()
        if not all(name.startswith('model.') for name in state):
            raise ValueError('Unexpected source namespace')
        mapped = {name[6:]: value for name, value in state.items()}
        if set(mapped) != set(target):
            raise ValueError('Source key mismatch')
        for name, value in mapped.items():
            if value.shape != target[name].shape or value.dtype != target[name].dtype:
                raise ValueError('Source shape/dtype mismatch: '+name)
        self.backbone.load_state_dict(mapped, strict=True)
        return {name: 'backbone.'+name[6:] for name in state}

    def forward(self, x):
        logits = self.backbone(**x)
        risk = logits.softmax(-1)[:, 1]
        return {'logits': logits, 'risk': risk,
                'accepted': torch.ones_like(risk, dtype=torch.bool)}

    def policy_receipt(self):
        return {'only_child': 'backbone', 'refiner_exists': False,
                'residual_scale_exists': False, 'dummy_parameters': False,
                'frozen_prefixes': list(FROZEN_PREFIXES),
                'dropout': 'All original probabilities and training flags retained',
                'no_detach_or_blanket_eval': True}
