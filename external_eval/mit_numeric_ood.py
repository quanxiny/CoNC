"""Frozen 0..32767 numeric adapter; neural weights and aggregation are unchanged."""
import types
MAX_LITERAL = 32767

def extended_literal_issues(source: str, included_lines: list[int]) -> list[dict]:
    selected = set(included_lines)
    issues = []
    from input_contract import LEXEME, CANONICAL_DECIMAL
    for match in LEXEME.finditer(source):
        if match.lastgroup != "number":
            continue
        token = match.group()
        line = source.count("\n", 0, match.start()) + 1
        if line not in selected:
            continue
        if not CANONICAL_DECIMAL.fullmatch(token):
            reason = "unsupported_literal_spelling"
        elif int(token) > MAX_LITERAL:
            reason = "number_outside_0_32767"
        else:
            continue
        issues.append({"line": line, "reason": reason, "literal": token})
    return issues

def encode_extended(public, edges, original):
    """Exact historical 15-field encoder with only the upper gate broadened."""
    import numpy as np
    lines, key = public["line_code"], public["key_line"]
    if not lines or not 0 <= key < len(lines):
        raise ValueError("INVALID_SLICE_KEY")
    if len(lines) > 12 or any(len(line) > 32 for line in lines):
        raise ValueError("SLICE_EXCEEDS_12_LINES_OR_32_TOKENS")
    graph = original["graph_build"](lines, key)
    if graph["stats"]["node_cap"] or len(graph["nodes"]) > 64:
        raise ValueError("GRAPH_EXCEEDS_64_NODES")
    x = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in {
        "tokens": ((12, 32), "uint16"), "numbers": ((12, 32), "int16"), "marks": ((12, 32), "uint8"),
        "line_count": ((), "uint8"), "key": ((), "uint8"), "ops": ((64,), "uint8"),
        "parents": ((64, 2), "int16"), "node_line": ((64,), "uint8"), "condition": ((64,), "uint8"),
        "unit": ((64,), "uint8"), "known": ((64,), "uint8"), "bits": ((64, 32), "uint8"),
        "target": ((64,), "uint8"), "node_count": ((), "uint8"), "edges": ((12, 12), "uint8")}.items()}
    x["line_count"][...] = len(lines)
    x["key"][...] = key
    x["marks"][...] = 1
    vocab = original["vocabulary"]
    for li, line in enumerate(lines):
        for ti, token in enumerate(line):
            numeric = token.lstrip("+-").isdigit()
            x["tokens"][li, ti] = vocab["<NUM>"] if numeric else vocab.get(token, 1)
            if numeric:
                value = int(token)
                if not 0 <= value <= MAX_LITERAL:
                    raise ValueError("NUMBER_OUTSIDE_0_32767")
                x["numbers"][li, ti] = value
                x["marks"][li, ti] = 0
    for ni, node in enumerate(graph["nodes"]):
        for key_out, key_in in [("ops", "op"), ("node_line", "line"), ("condition", "condition"),
                                ("unit", "unit"), ("known", "initial_known"), ("target", "target")]:
            x[key_out][ni] = node[key_in]
        x["parents"][ni, :len(node["parents"])] = node["parents"]
        if node["initial_known"]:
            value = node["literal"] & 0xffffffff
            x["bits"][ni] = [(value >> bit) & 1 for bit in range(32)]
    x["node_count"][...] = len(graph["nodes"])
    lookup = {int(value): index for index, value in enumerate(public["line_num"])}
    for a, b, kind in edges:
        if a in lookup and b in lookup:
            x["edges"][lookup[a], lookup[b]] |= 1 if kind == "REACHES" else 2
    return x, graph

def patch_numeric_range(model, torch):
    modules = [module for module in model.modules() if hasattr(module, "raw_bit_encoder") and hasattr(module, "line_inputs")]
    if len(modules) != 1:
        raise ValueError("expected one AllVisible numeric line encoder")
    module = modules[0]
    def line_inputs(self, x):
        numbers = x["numbers"]
        if numbers.dtype != torch.long:
            raise TypeError("exact integer values required before bit extraction")
        if ((numbers < 0) | (numbers > MAX_LITERAL)).any():
            raise ValueError("outside numeric OOD diagnostic domain")
        bits = ((numbers.unsqueeze(-1) >> torch.arange(32, device=numbers.device)) & 1).float()
        numeric = self.raw_bit_encoder(torch.cat([bits, torch.ones_like(bits[..., :1])], -1))
        tokens = x["tokens"].long()
        valid = tokens.ne(0)
        emb = torch.where(x["marks"].bool().unsqueeze(-1), self.word(tokens), numeric)
        return emb + self.token_pos(torch.arange(tokens.shape[2], device=tokens.device)), valid
    module.line_inputs = types.MethodType(line_inputs, module)
    return module
