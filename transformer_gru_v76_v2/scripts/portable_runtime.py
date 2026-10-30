"""Package-relative source imports and checksummed experiment resources."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'configs/MIGRATION_CONTRACT_v2.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def resolve(name):
    path = ROOT / name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Resource path must stay inside the package: ' + str(name))
    return path


def configure_imports():
    os.environ['OVERFLOWGEN_ROOT'] = str(ROOT / 'external/overflowgen')
    os.environ['RESEARCH_ROOT'] = str(ROOT / 'src/research_project')
    store = ROOT / 'src/research_project/ubuntu_local_research_store'
    for group in ['overflow_interfaces_v51', 'full_overflowgen_v52', 'adaptive_graph_liquid_v70',
                  'adaptive_graph_frozen_control_v71', 'frozen_encoder_dropout_v75', 'no_refiner_frozen_v76']:
        sys.path.insert(0, str(store / group))
    sys.path.insert(0, str(ROOT / 'external/overflowgen/joint_graph_study_v1/implementation_family_isolated_v2'))


def validate_contract():
    contract = read(CONTRACT)
    for name, expected in contract['sha256'].items():
        path = resolve(name)
        if not path.is_file():
            raise FileNotFoundError('Required experiment asset is missing: ' + str(path) + '\nExtract the data/initialization archive into the repository root.')
        if sha(path) != expected:
            raise ValueError('Experiment asset checksum mismatch: ' + name)
    return {'status': 'PASS', 'contract_sha256': sha(CONTRACT), 'verified_files': len(contract['sha256'])}


def portable_protocol():
    contract = read(CONTRACT)
    override = os.environ.get('OVERFLOW_TRANSFER_SOURCES')
    source_file = Path(override).resolve() if override else ROOT / 'configs/portable_sources.json'
    sources = read(source_file)['transfer_sources']
    expanded = {}
    for seed, source in sources.items():
        item = dict(source)
        for key in ['checkpoint', 'DEV_receipt', 'rng']:
            if key not in item:
                continue
            item[key] = str((source_file.parent / item[key]).resolve() if override else resolve(item[key]))
        expanded[seed] = item
    return {'transfer_sources': expanded, 'resources': contract['resources'],
            'initialization_rng': {seed: {'file': source['rng'], 'sha256': source['rng_sha256']}
                                   for seed, source in expanded.items() if 'rng' in source}}


def validate_training_readiness():
    return validate_contract()


def check_output(path):
    out = Path(path).resolve()
    for name in ['assets', 'datasets', 'src', 'external', 'configs', 'scripts', 'third_party', 'docs', 'licenses']:
        if out.is_relative_to((ROOT / name).resolve()):
            raise ValueError('Choose an output directory outside supplied experiment resources')
    return out
