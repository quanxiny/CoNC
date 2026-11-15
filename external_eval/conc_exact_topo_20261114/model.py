"""Numerically exact state channel, frozen CoNC context, learned residual fusion."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

GRAPH_FIELDS = ('ops', 'parents', 'bits', 'known', 'node_count', 'condition', 'unit', 'target')


@torch.no_grad()
def exact_states(x, propagate=True):
    """Evaluate supported operations in dependency order, without floating arithmetic.

    Exactness is a property of the explicitly supplied signed32 expression DAG,
    not a claim that the original C extraction or type inference was correct.
    """
    ops = x['ops'].long(); batch, nodes = ops.shape
    device = ops.device
    valid = torch.arange(nodes, device=device)[None] < x['node_count'][:, None]
    rawbits = x['bits'].long()
    word = (rawbits << torch.arange(32, device=device)).sum(-1)
    literal = valid & ops.eq(1) & x['known'].bool() & ((rawbits == 0) | (rawbits == 1)).all(-1) & word.ge(0) & word.lt(2**31)
    values = torch.where(literal, word, torch.zeros_like(word))
    known = literal.clone()
    if not propagate:
        return values, known
    rows = torch.arange(batch, device=device)
    for node in range(nodes):
        a, b = x['parents'][:, node].long().unbind(-1)
        ac, bc = a.clamp(0, nodes-1), b.clamp(0, nodes-1)
        av, bv = values[rows, ac], values[rows, bc]
        ak = (a > 0) & (a < node) & known[rows, ac]
        bk = (b > 0) & (b < node) & known[rows, bc]
        op = ops[:, node]
        copying = valid[:, node] & op.eq(2) & ak
        arithmetic = valid[:, node] & ((op == 3) | (op == 4)) & ak & bk
        computed = torch.where(op.eq(3), av + bv, av - bv)
        arithmetic &= computed.ge(-(2**31)) & computed.lt(2**31)
        known[:, node] |= copying | arithmetic
        values[:, node] = torch.where(copying, av, torch.where(arithmetic, computed, values[:, node]))
    return values, known


class ExactFusion(nn.Module):
    """Matched parameterization in exact-topological and literal-only arms."""
    def __init__(self, arm='exact_topo'):
        super().__init__()
        if arm not in ('exact_topo', 'literal_only'):
            raise ValueError(arm)
        self.arm = arm
        self.node = nn.Sequential(nn.Linear(131, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.context = nn.Sequential(nn.LayerNorm(896), nn.Linear(896, 64), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(258, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x, context, base_logits, values=None, known=None):
        if values is None or known is None:
            values, known = exact_states(x, self.arm == 'exact_topo')
        ops = x['ops'].long(); batch, nodes = ops.shape
        valid = torch.arange(nodes, device=ops.device)[None] < x['node_count'][:, None]
        known = known & valid
        bits = ((values[..., None] >> torch.arange(32, device=ops.device)) & 1).float()
        bits = torch.where(known[..., None], bits, torch.full_like(bits, .5))
        q = torch.cat([bits, known[..., None].float()], -1)
        rows = torch.arange(batch, device=ops.device)[:, None]
        parent = x['parents'].long()
        pa, pb = parent[..., 0], parent[..., 1]
        position = torch.arange(nodes, device=ops.device)[None]
        ap = (pa > 0) & (pa < position) & valid
        bp = (pb > 0) & (pb < position) & valid
        pa, pb = pa.clamp(0, nodes-1), pb.clamp(0, nodes-1)
        unknown = q.new_full((33,), .5); unknown[-1] = 0
        qa = torch.where(ap[..., None], q[rows, pa], unknown)
        qb = torch.where(bp[..., None], q[rows, pb], unknown)
        pair_known = ap & bp & known[rows, pa] & known[rows, pb]
        va, vb = values[rows, pa], values[rows, pb]
        relation = torch.stack([pair_known, pair_known & (va < vb), pair_known & (va == vb), pair_known & (va > vb)], -1).float()
        meta = torch.cat([F.one_hot(ops, 20), F.one_hot(x['condition'].long(), 4),
                          F.one_hot(x['unit'].long(), 3), x['target'][..., None]], -1).float()
        node = self.node(torch.cat([q, qa, qb, relation, meta], -1))
        mask = valid[..., None]
        target = (x['target'].bool() & valid)[..., None]
        global_mean = (node * mask).sum(1) / mask.sum(1).clamp_min(1)
        target_mean = (node * target).sum(1) / target.sum(1).clamp_min(1)
        target_max = node.masked_fill(~target, -torch.inf).amax(1)
        target_max = torch.where(target.any(1), target_max, torch.zeros_like(target_max))
        fused = torch.cat([self.context(context.float()), global_mean, target_mean, target_max, base_logits.float()], -1)
        delta = self.fusion(fused).squeeze(-1)
        logits = torch.stack([base_logits[:, 0], base_logits[:, 1] + delta], -1)
        return {'logits': logits, 'exact_values': values, 'exact_known': known,
                'delta': delta, 'source_program_proof': False}


class CoNCExactTopo(nn.Module):
    """Live wrapper: preserve original forward, attach a new trainable branch."""
    def __init__(self, base, arm='exact_topo'):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.adapter = ExactFusion(arm)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x):
        captured = []
        handle = self.base.backbone.head.register_forward_pre_hook(lambda m, args: captured.append(args[0].detach()))
        try:
            with torch.no_grad():
                original = self.base(x)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError('Expected one CoNC pooled readout')
        # Match the explicitly approximate cached context used during training.
        context = captured[0].to(torch.float16).float()
        result = self.adapter(x, context, original['logits'])
        return {**result, 'risk': result['logits'].softmax(-1)[:, 1], 'base_logits': original['logits']}
