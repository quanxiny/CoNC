"""Run the released strict-split mechanism experiments."""
import argparse
import json
from .common import DEFAULT_OUT, ROOT, config, output_dir

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['smoke', 'prepare', 'train', 'score'])
    p.add_argument('--out', type=__import__('pathlib').Path, default=DEFAULT_OUT / 'mechanism_release')
    p.add_argument('--variant', choices=['conc'] + config()['variants'] + ['vd'], default='conc',
                   help='conc selects the strict-study CoNC configuration (historical ID: full)')
    p.add_argument('--seed', type=int, choices=[11, 29, 47], default=11)
    p.add_argument('--stage', type=int, choices=[1, 2], default=1)
    p.add_argument('--dataset', choices=['source', 'mit'], default='source')
    p.add_argument('--device', default='cpu')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.variant == 'conc': a.variant = 'full'
    out = output_dir(a.out)
    if a.mode == 'smoke':
        from .validate import run
        result = run(out, a.device)
    elif a.mode == 'prepare':
        from .prepare_source import prepare
        result = prepare(out, ROOT / 'data/mechanism/source_manifest.csv')
    elif a.mode == 'train':
        from .train import run
        result = run(out, a.variant, a.seed, a.stage, a.device, a.resume)
    else:
        from .evaluate import score
        result = score(out, a.variant, a.seed, a.dataset, a.device)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
