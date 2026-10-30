"""Initialize the released strict-split source experiment without historical run files."""
import argparse
import shutil
from collections import Counter
from .common import *


def prepare(out, manifest):
    out = output_dir(out)
    manifest = Path(manifest).resolve()
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream))
    if Counter(row['split'] for row in rows) != {'train': 94157, 'val': 11770, 'test': 11769}:
        raise ValueError('Expected the frozen retrospective strict split')
    if {int(row['row']) for row in rows} != set(range(117696)):
        raise ValueError('Source row identities are incomplete')
    groups = {}
    for row in rows:
        if groups.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError('A connected group crosses split boundaries')
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'source_manifest.csv'
    if target.exists() and sha(target) != sha(manifest):
        raise ValueError('Existing experiment has different split membership')
    if not target.exists():
        shutil.copyfile(manifest, target)
    paths = [HERE / name for name in ['models.py', 'train.py', 'common.py', 'prepare_source.py', 'evaluate.py', 'config.json']]
    paths += list((STORE / 'overflow_interfaces_v51/overflow51').glob('*.py'))
    paths += [STORE / 'full_overflowgen_v52/all_visible52.py', STORE / 'adaptive_graph_liquid_v70/model70.py',
              CACHE / 'implementation_family_isolated_v2/models.py', CACHE / 'implementation_family_isolated_v2/graph_input.py']
    dependency_paths = [CACHE / (key + '.npy') for key in KEYS + ['labels']]
    dependency_paths += [METADATA, PACKAGE / 'external/overflowgen/safe_numeric_artifacts/word_index.json',
                         PACKAGE / 'external/overflowgen/safe_numeric_artifacts/dice_Q.npy']
    freeze = {'schema': 'source-only-release-v1', 'config': config(),
              'code': {str(path.relative_to(ROOT)): sha(path) for path in paths},
              'dependencies': {str(path.relative_to(ROOT)): sha(path) for path in dependency_paths},
              'artifacts': {'source_manifest.csv': sha(target)},
              'selection': 'source DEV native cross entropy; no target data used',
              'split': 'retrospective strict connected-component split'}
    atomic_json(out / 'freeze.json', freeze, immutable=True)
    return {'directory': str(out), 'counts': dict(Counter(row['split'] for row in rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT / 'mechanism_release')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/mechanism/source_manifest.csv')
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.manifest)))


if __name__ == '__main__':
    main()
