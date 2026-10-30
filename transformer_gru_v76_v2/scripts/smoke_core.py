"""CPU architecture checks on constructed inputs; optional asset integrity checks."""
import argparse
import gc
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from portable_runtime import configure_imports, validate_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-assets', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    configure_imports()
    import torch
    from check_comparison_assets import fixture
    from model70 import AdaptiveGraph70, encode_original, readout
    from model71 import FrozenResidual71
    from model75 import FrozenEncoder75
    from model76 import NoRefiner76
    from all_visible52 import AllVisible
    from overflow51.baselines import RiskBridge, LegacyPureGraph, NativeVD
    from graph_input import build
    torch.set_num_threads(2)
    torch.manual_seed(11)
    x = fixture(torch)
    checks = []
    factories = [
        ('CoNC', lambda: AdaptiveGraph70('liquid_transfer_frozen')),
        ('residual_refiner', FrozenResidual71),
        ('no_refiner', NoRefiner76),
        ('source_backbone', lambda: RiskBridge(AllVisible())),
        ('lexical_context', lambda: RiskBridge(LegacyPureGraph('semantic'))),
        ('numerical_structure', lambda: RiskBridge(LegacyPureGraph('numeric'))),
        ('joint_representation', lambda: RiskBridge(LegacyPureGraph('joint'))),
        ('fixed_encoder_no_dropout', FrozenEncoder75),
    ]
    for name, factory in factories:
        model = factory().eval()
        with torch.inference_mode():
            logits = model(x)['logits']
            assert logits.shape == (2, 2) and torch.isfinite(logits).all(), name
            if name == 'CoNC':
                reference = model.backbone(**x)
                assert torch.equal(logits, reference), 'Zero-initialized refiner must preserve the backbone'
        checks.append({'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
                       'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'status': 'PASS'})
        del model, logits
        gc.collect()
    graph = build([['int', 'VAR1', '=', '3', ';'], ['char', 'VAR2', '[', '10', ']', ';'],
                   ['VAR2', '[', 'VAR1', ']', '=', '0', ';']], 2)
    assert graph['nodes']
    result = {'status': 'PASS', 'device': 'cpu', 'torch_version': torch.__version__, 'models': checks,
              'finite_token_frontend': 'PASS', 'optimizer_steps': 0, 'dataset_rows_used': 0,
              'full_training': 'NOT_RUN', 'historical_metric_reproduction': 'NOT_RUN'}
    if args.with_assets:
        result['assets'] = validate_contract()
        from common70 import SelectedData
        data = SelectedData()
        result['selected_rows'] = {'train': len(data.rows('train')), 'dev': len(data.rows('val'))}
        from overflow51.overflowgen import original_vd_batch
        model = NativeVD('family').eval()
        import numpy as np
        vd = original_vd_batch(data.a, data.rows('val')[:2])
        with torch.inference_mode():
            output = model(**vd)
        assert output.shape == (2, 2) and torch.isfinite(output).all()
        result['VDTransformer'] = 'PASS'
        result['dataset_rows_used'] = 2
        result['TEST_rows_used'] = 0
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
