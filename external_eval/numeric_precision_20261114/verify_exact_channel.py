"""Verify arithmetic against independent Python integers; no C code involved."""
import json
import random
from pathlib import Path
from exact_channel import State, calculate, evaluate_graph, literal, numeric_features


def verify():
    counts = {'exhaustive_signed8_operations': 0, 'random_signed32_operations': 0,
              'edge_signed32_operations': 0, 'graph_checks': 0}
    unknown_overflow = 0
    def check(op, a, b, width):
        nonlocal unknown_overflow
        result = calculate(op, literal(a, width), literal(b, width), width)
        expected = a + b if op == 3 else a - b
        if -(1 << (width - 1)) <= expected < (1 << (width - 1)):
            assert result.value == expected, (op, a, b, result, expected)
        else:
            assert result.value is None and result.reason == 'signed_overflow'
            unknown_overflow += 1
    for a in range(-128, 128):
        for b in range(-128, 128):
            for op in [3, 4]:
                check(op, a, b, 8)
                counts['exhaustive_signed8_operations'] += 1
    rng = random.Random(20261114)
    for _ in range(10000):
        a, b = [rng.randrange(-(2**31), 2**31) for _ in range(2)]
        for op in [3, 4]:
            check(op, a, b, 32)
            counts['random_signed32_operations'] += 1
    edges = sorted({-(2**31), 2**31-1, -1, 0, 1} |
                   {v for k in range(31) for v in [2**k-1, 2**k, -2**k]})
    for a in edges:
        for b in edges:
            for op in [3, 4]:
                check(op, a, b, 32)
                counts['edge_signed32_operations'] += 1
    # Twenty sequential additions require twenty synchronous rounds.
    graph = {'node_count': 23, 'ops': [0, 1, 1] + [3]*20,
             'known': [0, 1, 1] + [0]*20,
             'bits': [[0]*32, [0]*32, [1]+[0]*31] + [[0]*32 for _ in range(20)],
             'parents': [[0,0], [0,0], [0,0]] + [[1 if i==3 else i-1, 2] for i in range(3,23)]}
    topo, sync = evaluate_graph(graph), evaluate_graph(graph, 20)
    assert topo[-1].value == sync[-1].value == 20
    assert evaluate_graph(graph, 4)[-1].value is None
    assert numeric_features(topo)[-1][-1] == 1
    counts['graph_checks'] += 4
    graph['known'][1] = 0
    assert evaluate_graph(graph)[-1].value is None
    assert numeric_features(evaluate_graph(graph))[-1][-1] == 0
    counts['graph_checks'] += 2
    graph['known'][1] = 1
    graph['parents'][3] = [22, 2]
    assert evaluate_graph(graph)[-1].value is None
    counts['graph_checks'] += 1
    assert calculate(2, literal(-17)).value == -17
    assert calculate(3, State(None, 'unknown'), literal(0)).value is None
    assert calculate(7, literal(2), literal(3)).value is None
    counts['graph_checks'] += 3
    return {'status':'PASS', 'counts':counts, 'overflow_cases_correctly_unknown':unknown_overflow,
            'oracle':'independent Python integer + / - with explicit range check',
            'scope':'arithmetic component verification, not neural training or C semantic validation'}


if __name__ == '__main__':
    result = verify()
    Path(__file__).with_name('kernel_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
