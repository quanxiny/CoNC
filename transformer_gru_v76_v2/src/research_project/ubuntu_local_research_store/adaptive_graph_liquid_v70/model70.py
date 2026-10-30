"""Adaptive liquid graph refinement on the complete frozen-lineage AllVisible.

This is an engineered state-dependent relaxation model, not a novelty claim or
an exact LTC/CfC reproduction. s denotes computational depth, never wall time
or program execution time. A matched residual graph update uses identical
parameters and inputs, without a learned relaxation time constant.
"""
import sys
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F

STORE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STORE / 'full_overflowgen_v52'))
sys.path.insert(0, str(STORE / 'overflow_interfaces_v51'))
from all_visible52 import AllVisible

ARMS = {'residual_scratch': ('residual', False, False),
        'liquid_scratch': ('liquid', False, False),
        'residual_transfer_joint': ('residual', True, False),
        'liquid_transfer_joint': ('liquid', True, False),
        'liquid_transfer_frozen': ('liquid', True, True)}
FROZEN_PREFIXES = ('word.', 'token_pos.', 'line_pos.', 'ordered_token_encoder.',
    'line_encoder.', 'raw_bit_encoder.', 'op.', 'cond.', 'unit.', 'target.', 'pos.',
    'signedness_unknown', 'q_init.', 'q_layers.', 'feedback.', 'edge_gates.')


def encode_original(m, x):
    """Exact AllVisible forward up to its normalized node states and final q.
    The gate must compare these logits with the untouched original class.
    """
    ops = x['ops'].long(); B, N = ops.shape; dev = ops.device
    mask = torch.arange(N, device=dev)[None, :] >= x['node_count'][:, None]
    ix = torch.arange(B, device=dev)[:, None]
    h = m.op(ops) + m.cond(x['condition'].long()) + m.unit(x['unit'].long()) + m.target(x['target'].long()) + m.pos(torch.arange(N, device=dev)) + m.signedness_unknown
    emb, valid = m.line_inputs(x); _, L, T, D = emb.shape
    pad = ~valid.reshape(B * L, T); safe = pad.clone(); safe[safe.all(-1), 0] = False
    encoded = m.ordered_token_encoder(emb.reshape(B * L, T, D), src_key_padding_mask=safe).reshape(B, L, T, D)
    line = (encoded * valid.unsqueeze(-1)).sum(2) / valid.sum(2).clamp_min(1).unsqueeze(-1)
    line = line + m.line_pos(torch.arange(L, device=dev))
    lm = torch.arange(L, device=dev)[None, :] >= x['line_count'][:, None]
    line = m.line_encoder(line, src_key_padding_mask=lm); adj = x['edges'].gt(0).float()
    line = line + torch.bmm(adj, line) / adj.sum(2).clamp_min(1).unsqueeze(-1)
    h = h + line[ix, x['node_line'].long()]
    known = x['known'].float(); bits = x['bits'].float()
    q = torch.cat([bits, known.unsqueeze(-1), torch.zeros_like(bits[..., :3])], -1)
    q = torch.where(known.bool().unsqueeze(-1), q, torch.full_like(q, .5)); h = h + m.q_init(q)
    parents = x['parents'].long(); rows = x['node_line'].long()
    node_adj = adj[ix[:, :, None], rows[:, :, None], rows[:, None, :]]
    h = h + torch.bmm(node_adj, h) / node_adj.sum(2).clamp_min(1).unsqueeze(-1)
    for j, layer in enumerate(m.graph):
        a = h[ix, parents[:, :, 0]]; b = h[ix, parents[:, :, 1]]
        message = m.parents_message[j](torch.cat([a, b], -1))
        q = m.q_layers[j](h, q, parents, ops, bits, known); qa = q[ix, parents[:, :, 0]]
        message = message * m.edge_gates[j](torch.cat([q, qa], -1)).sigmoid()
        h = h + m.feedback[j](q); h = layer(h + message, src_key_padding_mask=mask)
    return m.norm(h), q, ~mask


def readout(m, h, x, valid):
    vn = valid.float(); target = x['target'].float() * vn
    pooled = (h * vn.unsqueeze(-1)).sum(1) / vn.sum(1).clamp_min(1).unsqueeze(-1)
    risk = (h * target.unsqueeze(-1)).sum(1) / target.sum(1).clamp_min(1).unsqueeze(-1)
    return m.head(torch.cat([pooled, risk], -1))


class GraphRefiner(nn.Module):
    def __init__(self, mode, width=448, state=192, steps=6):
        super().__init__()
        if mode not in {'liquid', 'residual'}: raise ValueError(mode)
        self.mode = mode; self.steps = steps; self.state = state
        self.anchor = nn.Linear(width, state)
        self.meta = nn.Linear(48, state)
        self.query = nn.Linear(state, 32); self.key = nn.Linear(state, 32)
        self.value = nn.Linear(state, state)
        self.type_bias = nn.Parameter(torch.zeros(5))
        self.type_value = nn.Parameter(torch.randn(5, state) * .02)
        self.drive = nn.Sequential(nn.Linear(4 * state, 256), nn.GELU(), nn.Linear(256, state))
        self.rate = nn.Sequential(nn.Linear(4 * state, 256), nn.GELU(), nn.Linear(256, state))
        self.out = nn.Linear(state, width, bias=False)

    def relations(self, x, valid):
        B, N = valid.shape; dev = valid.device; rows = x['node_line'].long()
        ix = torch.arange(B, device=dev)[:, None, None]
        edge = x['edges'][ix, rows[:, :, None], rows[:, None, :]].long()
        types = torch.zeros(B, N, N, 5, device=dev)
        for slot in [0, 1]:
            types[..., slot].scatter_(2, x['parents'][:, :, slot].long().unsqueeze(-1), 1.)
        types[..., 2] = (edge & 1).ne(0); types[..., 3] = (edge & 2).ne(0)
        types[..., 4] = torch.eye(N, device=dev)[None]
        types = types * valid[:, :, None, None] * valid[:, None, :, None]
        allowed = types.sum(-1).gt(0)
        # Padded queries get a private self edge solely to keep softmax finite.
        safe = allowed | (torch.eye(N, device=dev, dtype=torch.bool)[None] & ~valid[:, :, None])
        count = types.sum(-1).clamp_min(1).unsqueeze(-1)
        return types / count, allowed, safe

    def forward(self, h, q, x, valid, steps=None):
        K = self.steps if steps is None else steps
        if K not in {1, 3, 6, 12}: raise ValueError('Unregistered computation depth')
        anchor = self.anchor(h); z0 = torch.tanh(anchor); z = z0 * valid.unsqueeze(-1)
        types, allowed, safe = self.relations(x, valid)
        typelogit = torch.einsum('bijt,t->bij', types, self.type_bias)
        typevalue = torch.einsum('bijt,td->bijd', types, self.type_value)
        degree = allowed.sum(-1).clamp_min(1)
        known = x['known'].float()
        known_neighbor = (allowed.float() * known[:, None, :]).sum(-1) / degree
        static = torch.cat([q, known.unsqueeze(-1), F.one_hot(x['condition'].long(), 4).float(),
            F.one_hot(x['unit'].long(), 3).float(), x['target'].float().unsqueeze(-1),
            (torch.log1p(degree.float()) / math.log(65)).unsqueeze(-1), known_neighbor.unsqueeze(-1)], -1)
        rate_means = []; gate_means = []; changes = []; dt = 1. / K
        for j in range(K):
            score = torch.bmm(self.query(z), self.key(z).transpose(1, 2)) / math.sqrt(32) + typelogit
            weights = score.masked_fill(~safe, -torch.inf).softmax(-1)
            message = torch.bmm(weights, self.value(z)) + (weights.unsqueeze(-1) * typevalue).sum(2)
            progress = torch.full_like(known.unsqueeze(-1), j / K)
            meta = self.meta(torch.cat([static, progress], -1))
            features = torch.cat([z, message, anchor, meta], -1)
            drive = self.drive(features); raw_rate = self.rate(features)
            if self.mode == 'liquid':
                rate = .05 + 3.95 * torch.sigmoid(raw_rate)
                alpha = -torch.expm1(-dt * rate)
                new = (1 - alpha) * z + alpha * torch.tanh(drive)
                rate_means.append(rate[valid].mean()); gate_means.append(alpha[valid].mean())
            else:
                # Identical active parameters; a conventional residual graph map.
                # The second MLP contributes to the drive, not a time constant.
                new = torch.tanh(z + dt * torch.tanh(drive + torch.tanh(raw_rate)))
            new = new * valid.unsqueeze(-1)
            changes.append((new - z)[valid].square().mean().sqrt()); z = new
        correction = torch.tanh(self.out(z - z0)) * valid.unsqueeze(-1)
        diagnostic = {'latent_abs_max': z[valid].abs().max(),
            'last_iteration_change_RMS': changes[-1], 'correction_RMS': correction[valid].square().mean().sqrt()}
        if rate_means:
            diagnostic.update(rate_mean=torch.stack(rate_means).mean(), update_gate_mean=torch.stack(gate_means).mean())
        return correction, diagnostic


class AdaptiveGraph70(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS: raise ValueError(arm)
        self.arm = arm
        self.backbone = AllVisible()
        self.refiner = GraphRefiner(ARMS[arm][0])
        self.residual_scale = nn.Parameter(torch.zeros(()))
        if ARMS[arm][2]:
            for name, p in self.backbone.named_parameters():
                if name.startswith(FROZEN_PREFIXES): p.requires_grad_(False)

    def forward(self, x, steps=None):
        h, q, valid = encode_original(self.backbone, x)
        correction, diagnostic = self.refiner(h, q, x, valid, steps)
        h = h + self.residual_scale.tanh() * correction
        logits = readout(self.backbone, h, x, valid)
        p = logits.softmax(-1)[:, 1]
        return {'risk': p, 'logits': logits, 'accepted': torch.ones_like(p, dtype=torch.bool),
                'diagnostic': diagnostic, 'residual_scale': self.residual_scale.tanh(),
                'source_program_proof': False}

    def transfer(self, state):
        if not all(k.startswith('model.') for k in state):
            raise ValueError('Expected original RiskBridge(AllVisible) checkpoint')
        mapped = {k[6:]: v for k, v in state.items()}
        expected = self.backbone.state_dict()
        if set(mapped) != set(expected): raise ValueError('Semantic module/key contract differs')
        for k, v in mapped.items():
            if v.shape != expected[k].shape or v.dtype != expected[k].dtype:
                raise ValueError('Incompatible transfer tensor ' + k)
        self.backbone.load_state_dict(mapped, strict=True)
        return {k: 'backbone.' + k[6:] for k in state}
