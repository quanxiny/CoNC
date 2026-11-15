"""Exact COPY/ADD/SUB channel for an explicit signed-integer expression DAG.

This is a numeric component, not a C interpreter or a vulnerability detector.
Unsupported operations, absent dependencies, and signed overflow stay unknown.
The neural latent states and their three historical auxiliary flags are not edited.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    value: int | None
    reason: str
    depth: int | None = None


def literal(value, width=32):
    if not -(1 << (width - 1)) <= value < (1 << (width - 1)):
        return State(None, 'outside_signed_domain')
    return State(int(value), 'exact_literal', 0)


def calculate(op, a, b=None, width=32):
    if op not in (2, 3, 4):
        return State(None, 'unsupported_operation')
    if a.value is None or (op != 2 and (b is None or b.value is None)):
        return State(None, 'unknown_operand')
    if not -(1 << (width - 1)) <= a.value < (1 << (width - 1)):
        return State(None, 'outside_signed_domain')
    if op == 2:
        return State(a.value, 'exact_copy', a.depth + 1)
    if not -(1 << (width - 1)) <= b.value < (1 << (width - 1)):
        return State(None, 'outside_signed_domain')
    # Full-adder recurrence, independent of the Python-integer scoring oracle.
    mask = (1 << width) - 1
    left, right = a.value & mask, b.value & mask
    subtract = op == 4
    if subtract:
        right ^= mask
    carry, word = int(subtract), 0
    for bit in range(width):
        x, y = (left >> bit) & 1, (right >> bit) & 1
        word |= (x ^ y ^ carry) << bit
        carry = (x & y) | (x & carry) | (y & carry)
    sign_a, sign_b, sign_result = a.value < 0, b.value < 0, bool(word & (1 << (width - 1)))
    overflow = ((sign_a != sign_b) if subtract else (sign_a == sign_b)) and sign_result != sign_a
    if overflow:
        return State(None, 'signed_overflow')
    value = word - (1 << width) if sign_result else word
    return State(value, 'exact_sub' if subtract else 'exact_add', max(a.depth, b.depth) + 1)


def evaluate_graph(graph, rounds=None):
    """Use only operation, ordered parents, literal bits/mask, and node count.

    rounds=None uses topological dependency order; an integer uses synchronous
    message-passing rounds. Node zero is a reserved parent sentinel.
    """
    n = int(graph['node_count'])
    states = [State(None, 'unresolved') for _ in range(n)]
    for i in range(n):
        if int(graph['ops'][i]) == 1 and bool(graph['known'][i]):
            bits = list(graph['bits'][i])
            if len(bits) != 32 or any(v not in (0, 1) for v in bits):
                states[i] = State(None, 'invalid_literal_bits')
            else:
                states[i] = literal(sum(int(v) << bit for bit, v in enumerate(bits)))

    def update(i, previous):
        op = int(graph['ops'][i])
        if op == 1:
            return states[i]
        if op not in (2, 3, 4):
            return State(None, 'unsupported_operation')
        a, b = map(int, graph['parents'][i])
        if not 0 < a < i or (op != 2 and not 0 < b < i):
            return State(None, 'missing_or_nontopological_parent')
        return calculate(op, previous[a], None if op == 2 else previous[b])

    if rounds is None:
        for i in range(n):
            states[i] = update(i, states)
    else:
        if rounds < 0:
            raise ValueError('rounds must be nonnegative')
        for _ in range(rounds):
            states = [update(i, states) for i in range(n)]
    return states


def numeric_features(states):
    """Separate 33-channel exact bits + exact-known input for future integration.

    These are *additional* channels, not replacements for a trained latent Q.
    Exact-known means exact under this supplied-graph arithmetic contract only.
    """
    return [[.5] * 32 + [0.] if s.value is None else
            [float((s.value >> bit) & 1) for bit in range(32)] + [1.] for s in states]
