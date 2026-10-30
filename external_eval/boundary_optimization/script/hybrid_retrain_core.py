"""Pure contracts for fixed-gate soft residual specialization.

Gold labels enter supervised CE only. Static gates are never pseudo-labels.
"""
import math
UNKNOWN = 'UNKNOWN'
ARMS = ('ordinary_continue', 'residual_weighted')


def state(v):
    if (type(v) is int and v in (0, 1)) or (type(v) is str and v == UNKNOWN):
        return v
    raise ValueError('gate state must be exact int0/1 or UNKNOWN')


def residual_mask(D_trace, Eva):
    d, e = state(D_trace), state(Eva)
    return d == UNKNOWN and e != 1


def validate_gate(record):
    d, e = state(record['D_trace']), state(record['Eva'])
    r = residual_mask(d, e)
    if type(record['R']) is not bool or record['R'] != r:
        raise ValueError('R differs from fixed D_unknown and Eva_not_positive rule')
    return {'D_trace': d, 'Eva': e, 'R': r}


def combine_decision(neural_risk, D_trace, Eva):
    d, e = state(D_trace), state(Eva)
    if type(d) is int:
        return {'prediction': d, 'source': 'D_trace', 'residual': False}
    if e == 1:
        return {'prediction': 1, 'source': 'Eva_positive', 'residual': False}
    if neural_risk is None:
        return {'prediction': UNKNOWN, 'source': 'UNRESOLVED_WITHOUT_N', 'residual': True}
    if isinstance(neural_risk, bool) or not isinstance(neural_risk, (int, float)) or not math.isfinite(neural_risk) or not 0 <= neural_risk <= 1:
        raise ValueError('neural risk must be finite probability or None')
    return {'prediction': int(neural_risk >= .5), 'source': 'N_residual', 'residual': True}


def supervised_ce(logits, labels, R, arm, mu):
    import torch
    from torch.nn import functional as F
    if arm not in ARMS:
        raise ValueError('unregistered training arm')
    if isinstance(mu, bool) or not isinstance(mu, (float, int)) or not math.isfinite(mu) or not 1 <= mu <= 2:
        raise ValueError('mu must be fixed1+fullTRAIN_residual_fraction')
    if logits.ndim != 2 or logits.shape[-1] != 2 or labels.shape != (logits.shape[0],):
        raise ValueError('binary query-level logits and labels required')
    residual = torch.as_tensor(R, device=logits.device)
    if residual.dtype != torch.bool or residual.shape != labels.shape:
        raise ValueError('R must be a bool mask matching query labels')
    ce = F.cross_entropy(logits, labels, reduction='none')
    weight = 1 + residual.to(ce.dtype)
    loss = ce.mean() if arm == 'ordinary_continue' else (weight * ce).mean() / mu
    return loss, {'CE': ce, 'weights': torch.ones_like(weight) if arm == 'ordinary_continue' else weight / mu,
                  'R': residual}


def selection_key(H_F1, NN_CE):
    if not math.isfinite(H_F1) or not 0 <= H_F1 <= 1 or not math.isfinite(NN_CE) or NN_CE < 0:
        raise ValueError('invalid original DEV selection metric')
    return (H_F1, -NN_CE)
