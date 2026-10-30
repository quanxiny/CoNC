"""Frozen CoNC/VDTransformer inference on released MIT and boundary inputs."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'transformer_gru_v76_v2'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE/'scripts'))
from portable_runtime import configure_imports, sha, check_output
from external_eval.portable_inputs import load_external
from external_eval.frozen_evaluate import tensor_batch
from external_eval.mit_numeric_ood import patch_numeric_range


def metrics(rows):
    known = [r for r in rows if r['prediction'] != 'UNKNOWN']
    tp = sum(r['label'] == 1 and r['prediction'] == 1 for r in known)
    tn = sum(r['label'] == 0 and r['prediction'] == 0 for r in known)
    fp = sum(r['label'] == 0 and r['prediction'] == 1 for r in known)
    fn = sum(r['label'] == 1 and r['prediction'] == 0 for r in known)
    div = lambda a, b: a/b if b else None
    result = {'target_scenes': len(rows), 'known_scenes': len(known), 'unknown_scenes': len(rows)-len(known),
              'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision_known': div(tp,tp+fp),
              'recall_known': div(tp,tp+fn), 'F1_known': div(2*tp,2*tp+fp+fn), 'specificity_known': div(tn,tn+fp)}
    if known and len({r['label'] for r in known}) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y = [r['label'] for r in known]; margins = [r['margin'] for r in known]
        result['AP_known'] = float(average_precision_score(y, margins))
        result['AUROC_known'] = float(roc_auc_score(y, margins))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['mit', 'boundary'], default='mit')
    ap.add_argument('--model', choices=['conc', 'full', 'vd'], default='conc', help='conc is CoNC; full is its historical alias')
    ap.add_argument('--seed', type=int, choices=[11,29,47], default=11)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=32)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--limit-scenes', type=int, help='First N registered scenes (default: 2)')
    g.add_argument('--all-scenes', action='store_true', help='Evaluate the complete target population')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    requested_model = args.model
    if args.model == 'conc': args.model = 'full'
    if args.batch_size < 1 or (args.limit_scenes is not None and args.limit_scenes < 1):
        ap.error('batch-size and limit-scenes must be positive')
    data = load_external(args.dataset)
    scenes = data['scenes'] if args.all_scenes else data['scenes'][:args.limit_scenes or 2]
    selected = {r['scene_id'] for r in scenes}
    grouped = {sid: [] for sid in selected}
    for r in data['slices']:
        if r['scene_id'] in selected: grouped[r['scene_id']].append(r)
    configure_imports()
    import numpy as np
    import torch
    from model70 import AdaptiveGraph70
    from overflow51.baselines import RiskBridge, NativeVD
    from overflow51.training import load_own_checkpoint
    from external_eval.mit_baseline_frozen_evaluate import vd_batch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): ap.error('CUDA unavailable; use --device cpu')
    if args.model == 'full':
        model = AdaptiveGraph70('liquid_transfer_frozen'); patch_numeric_range(model, torch); upper = 32767
        default = PACKAGE/f'assets/comparisons/overflowgen/adaptive_graph_liquid_v70/runs/liquid_transfer_frozen_s{args.seed}/best.pt'
    else:
        model = RiskBridge(NativeVD('family')); upper = 999
        default = PACKAGE/f'assets/comparisons/overflowgen/full_overflowgen_v52/runs/family_vd_s{args.seed}_lr2e-05/best.pt'
    checkpoint = args.checkpoint or default
    state = load_own_checkpoint(checkpoint); model.load_state_dict(state['model'], strict=True); del state
    model.to(args.device).eval()
    output = check_output(args.output_dir)
    if any(output.is_relative_to((ROOT/n).resolve()) for n in ['data','weights','external_eval','scripts','manifests','results']):
        ap.error('Choose a separate output directory')
    output.mkdir(parents=True, exist_ok=False)
    unsupported, valid_records, numeric_caches = {}, [], {}
    for scene in scenes:
        sid = scene['scene_id']; records = grouped[sid]; expected = scene['slice_ids']; reason = None
        if not scene['processable'] or not expected:
            reason = scene.get('unprocessed_reason') or 'incomplete_neural_input'
        elif args.dataset == 'mit' and args.model == 'vd' and not scene['strict_eligible']:
            reason = 'outside_VDTransformer_0_999_cohort'
        elif {r['slice_id'] for r in records} != set(expected):
            reason = 'incomplete_slice_inventory'
        else:
            for r in records:
                path = r['tensor_cache']
                if path not in numeric_caches:
                    with np.load(path, allow_pickle=False) as a: numeric_caches[path] = a['numbers']
                numbers = numeric_caches[path][r['tensor_row']]
                if ((numbers < 0) | (numbers > upper)).any():
                    reason = f'outside_model_numerical_domain_0_{upper}'; break
        if reason: unsupported[sid] = reason
        else: valid_records.extend(records)
    del numeric_caches
    scores, slice_rows = {}, []
    with torch.inference_mode():
        for start in range(0, len(valid_records), args.batch_size):
            records = valid_records[start:start+args.batch_size]
            x = tensor_batch(records, args.device, np, torch)
            result = model(vd_batch(x,torch) if args.model == 'vd' else x)
            if not torch.isfinite(result['logits']).all(): raise FloatingPointError('Nonfinite logits')
            for record, risk, pair in zip(records,result['risk'].cpu().tolist(),result['logits'].cpu().tolist()):
                row = {'scene_id':record['scene_id'],'slice_id':record['slice_id'],'risk':risk,'logit0':pair[0],'logit1':pair[1]}
                scores[record['slice_id']] = row; slice_rows.append(row)
    scene_rows = []
    for scene in scenes:
        sid = scene['scene_id']
        row = {'scene_id':sid,'label':int(scene['label']),'group':scene['group'],'prediction':'UNKNOWN',
               'risk':'','margin':'','max_slice_id':'','slice_count':len(grouped[sid]),'reason':unsupported.get(sid,'')}
        if sid not in unsupported:
            # Exported record order is the original aggregation order. Python max
            # retains the first slice when FP32 class-1 probabilities are equal.
            best = max((scores[r['slice_id']] for r in grouped[sid]),key=lambda r:r['risk'])
            row.update(prediction=int(best['risk'] >= .5),risk=best['risk'],margin=best['logit1']-best['logit0'],max_slice_id=best['slice_id'])
        scene_rows.append(row)
    with (output/'scene_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scene_rows[0]));w.writeheader();w.writerows(scene_rows)
    with gzip.open(output/'slice_predictions.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scene_id','slice_id','risk','logit0','logit1']);w.writeheader();w.writerows(slice_rows)
    report = {'dataset':args.dataset,'model':requested_model,'model_id':args.model,'seed':args.seed,'partial_evaluation':not args.all_scenes,
              'registered_target_scenes':data['manifest']['scene_count'],'checkpoint_sha256':sha(checkpoint),
              'input_manifest_sha256':data['identity']['manifest_sha256'],'numerical_domain':[0,upper],'threshold':.5,
              'aggregation':'max FP32 class-1 probability; first slice on ties; incomplete scene UNKNOWN',
              'optimizer_steps':0,**metrics(scene_rows)}
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__': main()
