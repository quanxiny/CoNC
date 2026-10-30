"""Load the unchanged official DeepDFA GGNN with its actual DGL operators.

Only the Lightning logging/metric base and registry decorator are bridged.
No graph convolution, pooling, embeddings, classifier, or model forward is
copied or replaced. This is a training-harness adaptation, not the native CLI.
"""
from __future__ import annotations
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / 'deps'))
os.environ.setdefault('DGLBACKEND', 'pytorch')
import torch
from torch import nn

SOURCE = ROOT / 'transformer_gru_v76_v2/third_party/DeepDFA/DDFA/code_gnn/models/flow_gnn/ggnn.py'
EXPECTED_SHA256 = '39361cb5ae76b6b37ce83e12d507650dfbe8573f0530cd2308786473ee1a7ea4'


class _HarnessBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def save_hyperparameters(self):
        values = dict(inspect.currentframe().f_back.f_locals)
        values.pop('self', None)
        values.pop('__class__', None)
        values.update(values.pop('kwargs', {}))
        self.hparams = types.SimpleNamespace(**values)


def official_class():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Official DeepDFA model source hash changed')
    # Reversible process-local import shims remove unrelated CLI dependencies.
    base = types.ModuleType('code_gnn.models.base_module')
    base.BaseModule = _HarnessBase
    registry = types.ModuleType('pytorch_lightning.utilities.cli')
    registry.MODEL_REGISTRY = lambda cls: cls
    shims = {'code_gnn.models.base_module': base,
             'pytorch_lightning.utilities.cli': registry}
    previous = {name: sys.modules.get(name) for name in shims}
    try:
        sys.modules.update(shims)
        spec = importlib.util.spec_from_file_location('_official_deepdfa_ggnn', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module.FlowGNNGGNNModule


def make_model():
    return official_class()(feat='_ABS_DATAFLOW_datatype_all_limitall_1000_limitsubkeys_1000',
                            input_dim=1002, hidden_dim=32, n_steps=5,
                            num_output_layers=3, concat_all_absdf=True,
                            label_style='graph', encoder_mode=False)
