"""Small storage adapter for the unmodified, pinned DeepWukong-GCN classifier.

The benchmark provides fixed backward slices, not the upstream bidirectional
whole-source XFG inventory. We retain its nodes and original Joern line edges.
Numeric candidate nodes, numeric states and handcrafted numeric features are
never read. Official statement tokenization, XFG conversion and model are used.
"""
from __future__ import annotations
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Tuple

BASE = Path(__file__).resolve().parent
EXTERNAL = BASE.parents[1]
WORKSPACE = EXTERNAL.parent
PACKAGE = WORKSPACE / "transformer_gru_v76_v2"
VENDOR = BASE / "vendor"
OUTPUT = EXTERNAL / "paper_baselines/deepwukong"
DATA = Path(os.environ.get("OVERFLOW_DATA_DIR", WORKSPACE / "data")) / "baselines/deepwukong"
def data_manifest():
    return DATA / "manifest.json"
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / "outputs/mpl_cache"))
sys.path.insert(0, str(BASE / "deps"))
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(EXTERNAL))
import networkx as nx
import numpy as np
import frozen_evaluate as frozen

sha = frozen.sha
read = lambda path: json.loads(Path(path).read_text())


def signed(path, value):
    frozen.atomic_json(path, {**value, "result_sha256": frozen.digest(value)})


def verify_vendor():
    manifest = read(VENDOR / "SOURCE_MANIFEST.json")
    if manifest["commit"] != "28449e57262dd24c0938108687e1019f0fc6ddd5":
        raise ValueError("unexpected upstream revision")
    for item in manifest["files"]:
        if sha(VENDOR / item["path"]) != item["sha256"]:
            raise ValueError("upstream source changed: " + item["path"])
    return {"commit": manifest["commit"], "manifest_sha256": sha(VENDOR / "SOURCE_MANIFEST.json"),
            "files": {r["path"]: r["sha256"] for r in manifest["files"]}}


def original_functions():
    """Compile exact upstream function ASTs; rebind only the API-list path.

    This avoids running upstream command-line mains and importing WordNinja
    when split_token=False. Function bodies and operator/keyword sets are intact.
    """
    verify_vendor()
    ns = {"nx": nx, "re": re, "List": List, "Set": Set, "Tuple": Tuple, "Dict": Dict,
          "join": os.path.join, "exists": os.path.exists}
    groups = {
        "src/data_generator.py": ["extract_line_number", "read_csv", "extract_nodes_with_location_info", "build_PDG", "build_XFG"],
        "src/preprocess/symbolizer.py": ["operators3", "operators2", "operators1", "keywords", "main_set", "main_args", "clean_gadget", "tokenize_code_line"],
    }
    for name, wanted in groups.items():
        selected, seen = [], set()
        for node in ast.parse((VENDOR / name).read_text()).body:
            names = ([node.name] if isinstance(node, ast.FunctionDef) else
                     [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else [])
            found = set(names) & set(wanted)
            if found and not found <= seen:
                selected.append(node)
                seen.update(found)
        if seen != set(wanted):
            raise ValueError("upstream AST definitions changed")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(VENDOR / name), "exec"), ns)
    api = EXTERNAL / "vendor/legacy_joern/data/sensiAPI.txt"
    asset = read(EXTERNAL / "vendor/source_assets.json")
    expected = next(a["sha256"] for a in asset["assets"] if a["path"] == "data/sensiAPI.txt")
    if asset["commit"] != "28449e57262dd24c0938108687e1019f0fc6ddd5" or sha(api) != expected:
        raise ValueError("upstream API list differs")
    ns["keywords"] = ns["keywords"].union(api.read_text().split(","))
    return ns, api


def native_graph(lines, tokens, edges, label=0):
    graph = nx.DiGraph(label=int(label))
    for line, words in zip(lines, tokens):
        graph.add_node(int(line), code_sym_token=list(words))
    for a, b in np.argwhere(np.asarray(edges) > 0):
        graph.add_edge(int(lines[a]), int(lines[b]), **{"c/d": "d" if int(edges[a][b]) & 1 else "c"})
    return graph


def official_data(record, vocab):
    from src.datas.graphs import XFG
    graph = native_graph(record["source_lines"], record["token_lines"], record["edges"], record["label"])
    result = XFG(xfg=graph).to_torch(vocab, 16)
    if result.edge_index.numel() == 0:
        # Upstream produces rank1 for []; preserve the empty set, normalize shape.
        result.edge_index = result.edge_index.reshape(2, 0)
    return result


def encoded_data(record, vocab):
    """Fast storage-equivalent conversion, accepted against official XFG.to_torch."""
    import torch
    from torch_geometric.data import Data
    x = torch.full((len(record["token_lines"]), 16), vocab.get_pad_id(), dtype=torch.long)
    for index, words in enumerate(record["token_lines"]):
        ids = vocab.convert_tokens_to_ids(words)[:16]
        if not ids:
            raise ValueError("empty statement would violate official packed-LSTM contract")
        x[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    edge = np.argwhere(np.asarray(record["edges"]) > 0).T.copy()
    return Data(x=x, edge_index=torch.as_tensor(edge, dtype=torch.long))


def load_records(name):
    manifest = read(data_manifest())
    path = DATA / (name + '.jsonl.gz')
    if sha(path) != manifest['files'][path.name]:
        raise ValueError('Prepared baseline input identity mismatch')
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def model_config(seed, w2v_path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(VENDOR / "configs/dwk.yaml")
    cfg.seed = seed
    cfg.num_workers = 0
    cfg.gnn.w2v_path = str(w2v_path)
    return cfg


def make_model(seed, w2v_path, vocab):
    from src.models.vd import DeepWuKong
    return DeepWuKong(model_config(seed, w2v_path), vocab, vocab.get_vocab_size(), vocab.get_pad_id())


def runtime(seed, threads=4):
    import random
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch
