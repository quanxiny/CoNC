#!/usr/bin/env python3
"""Run a strict-split baseline or backbone intervention on encoded inputs."""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
VARIANTS=('vd','local_q','no_h_to_q','no_dynamic_q_to_h','symmetric_q','no_clamp','mlp_q','no_both_dynamic')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,default='vd')
    parser.add_argument('--seed',type=int,choices=(11,29,47),default=11)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--limit',type=int,default=32)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    import torch
    from external_eval.tifs.common import SourceData
    from external_eval.tifs.models import make_model
    torch.set_num_threads(4)
    source=SourceData()
    model=make_model(args.variant,1 if args.variant=='vd' else 2,args.seed)
    ck=torch.load(ROOT/f'data/current_baseline_assets/{args.variant}_s{args.seed}.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ck['model'],strict=True)
    model=model.to(args.device).eval()
    with gzip.open(ROOT/f'data/current/strict/{args.variant}_s{args.seed}.csv.gz','rt',newline='') as f:
        refs={r['scene_id']:r for r in csv.DictReader(f) if r['status']=='SCORED'}
    with gzip.open(ROOT/'data/current/strict/exact_topo_s11.csv.gz','rt',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['sample_id'] in refs][:args.limit]
    count=mismatch=0;error=0.0
    with torch.inference_mode():
        for at in range(0,len(rows),args.batch_size):
            part=rows[at:at+args.batch_size]
            x,_=source.batch([int(r['source_row']) for r in part],args.device,vd=args.variant=='vd')
            if args.variant=='vd' and x['number_ids'].max()>999:
                raise ValueError('VDTransformer supports only number IDs up to 999')
            probs=model(x)['logits'].softmax(-1)[:,1].cpu().tolist()
            for row,prob in zip(part,probs):
                ref=refs[row['sample_id']];count+=1
                mismatch+=(prob>=.5)!=bool(int(ref['prediction']))
                error=max(error,abs(prob-float(ref['probability'])))
    print(json.dumps({'variant':args.variant,'seed':args.seed,'rows':count,'class_mismatches':mismatch,'max_probability_difference':error},indent=2))
    if mismatch:raise SystemExit('Predicted classes differ from the saved reference')

if __name__=='__main__':main()
