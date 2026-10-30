"""Frozen final-H boundary evidence using unchanged enumerator/address/native APIs."""
from pathlib import Path
import argparse
import collections
import copy
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[3]
S = Path(__file__).parent
sys.path[:0] = [str(ROOT), str(S), str(ROOT / 'external_eval')]
from external_eval.boundary_optimization.common import read, sha, atomic_json, write_csv
from external_eval import static_baselines as native
import enumerate_external_addresses as enumeration
import execute_external_address as execution
import execute_address_diagnostic as address
from external_address_adapter import enumerate_scene, aggregate_scene
from address_trace_readout import TraceReader

D = ROOT / 'external_eval/tifs_runs/hybrid_evidence_20261021/boundary'
PARENT = ROOT / 'external_eval/tifs_runs/boundary_external_20261021'
OLD = ROOT / 'external_eval/tifs_runs/boundary_hybrid_20261021'


def fixtures():
    checks = []
    source = 'void fixture(void) {\n char a[4];\n a[3] = 0;\n}\n'
    def reg(code, entry='fixture'):
        return {'unit_id': 'independent_fixture', 'entry': entry, 'group': 'fixture',
                'source_sha256': hashlib.sha256(code.encode()).hexdigest()}
    value = enumerate_scene(source, reg(source))
    assert value['inventory_complete'] and len(value['obligations']) == 1
    assert value['obligations'][0]['identity_valid']
    assert value['obligations'][0]['source_text'] == 'a[3]'
    checks.append('declared_fixture_entry_exact_span')
    value = enumerate_scene(source, reg(source, 'main'))
    assert not value['inventory_complete'] and value['context_blockers'] == ['ENTRY_NOT_UNIQUE']
    checks.append('wrong_entry_is_unknown')
    source2 = 'void fixture(void) {\n char a[4];\n a[0] = a[1];\n}\n'
    value = enumerate_scene(source2, reg(source2))
    assert len(value['obligations']) == 2 and not value['inventory_complete']
    assert all(not item['identity_valid'] for item in value['obligations'])
    checks.append('same_line_ambiguity_retained')
    source3 = '#include <string.h>\nvoid fixture(void) {\n char a[4], b[4];\n memcpy(a,b,4);\n}\n'
    value = enumerate_scene(source3, reg(source3))
    assert value['obligations'] and not value['inventory_complete']
    assert all(not item['identity_valid'] for item in value['obligations'])
    checks.append('memory_call_roles_not_newly_bound')
    result = {'status': 'PASS', 'checks': checks, 'adapter_sha256': sha(S / 'external_address_adapter.py'),
              'no_native_or_model_calls': True}
    atomic_json(D / 'entry_fixtures.json', result, immutable=True)


def prepare():
    D.mkdir(parents=True, exist_ok=True)
    if (D / 'prepare_receipt.json').exists():
        receipt = read(D / 'prepare_receipt.json')
        for p, h in receipt['files'].items(): assert sha(ROOT / p) == h, p
        return
    fixtures()
    old = read(OLD / 'units.json')
    units = [{k: a[k] for k in ['dataset', 'unit_id', 'entry', 'group', 'pair_id',
                              'source_path', 'source_sha256', 'diagnostic_split', 'variant']}
             for a in old['units'] if a['dataset'] == 'boundary']
    assert len(units) == len({a['unit_id'] for a in units}) == 448
    assert len({a['pair_id'] for a in units}) == 224 and len({a['group'] for a in units}) == 8
    assert {a['entry'] for a in units} == {'fixture'}
    for a in units: assert sha(a['source_path']) == a['source_sha256']
    units.sort(key=lambda a: a['unit_id'])
    atomic_json(D / 'scenes.json', {'units': units, 'registered_programs': 448, 'pairs': 224,
                'construction_groups': 8, 'labels_scores_used_for_selection': False,
                'scope': 'previously exposed controlled diagnostics; development split retained'}, immutable=True)
    small = []
    for group in sorted({a['group'] for a in units}):
        choices = sorted((a for a in units if a['group'] == group),
                         key=lambda a: hashlib.sha256(('final-boundary-entry-v1|' + a['unit_id']).encode()).hexdigest())
        small.append(choices[0])
    atomic_json(D / 'enumeration_preflight_manifest.json', {'units': small,
                'selection': 'one smallest fixed SHA per construction group, independent of label/score'}, immutable=True)
    cfg = native.signed(native.CONFIG)
    derived = copy.deepcopy(cfg)
    flags = derived['frama_c_eva']['flags']
    assert flags == ['-eva', '-machdep', 'x86_64', '-main', 'main', '-eva-slevel', '100']
    flags[flags.index('-main') + 1] = 'fixture'
    # No other native option, report selector, budget or model changes.
    derived['declared_entry_adapter'] = 'fixture from registered units, no wrapper/source edits'
    if (D / 'Eva_config.json').exists(): assert native.signed(D / 'Eva_config.json') == derived
    else: native.sign(D / 'Eva_config.json', derived)
    enum_dir = D / 'enumeration'; enum_dir.mkdir(exist_ok=True)
    deps = [Path(__file__), S / 'enumerate_external_addresses.py', S / 'external_address_adapter.py',
            S / 'query_access_linkage.py', D / 'scenes.json', D / 'enumeration_preflight_manifest.json',
            D / 'entry_fixtures.json', D / 'Eva_config.json', OLD / 'units.json',
            native.CONFIG, Path(native.__file__), native.WORK / 'tool_registry.json']
    files = {str(p.relative_to(ROOT)): sha(p) for p in deps}
    atomic_json(enum_dir / 'freeze.json', {'status': 'FROZEN_FINAL_BOUNDARY_ENUMERATION', 'files': files,
                'workers': 8, 'wall_per_unit': 60, 'python': enumeration.BASE,
                'python_sha256': sha(enumeration.BASE), 'new_adapter_semantics': False}, immutable=True)
    atomic_json(D / 'prepare_receipt.json', {'status': 'PASS', 'files': files,
                'source_programs': 448, 'source_execution': False, 'new_training_or_forward': False}, immutable=True)


def enumerate_all():
    prepare(); enumeration.D = D
    f = read(D / 'enumeration/freeze.json')
    for p, h in f['files'].items(): assert sha(ROOT / p) == h, p
    path = D / 'enumeration/all_receipt.json'
    if path.exists():
        receipt = read(path)
        for n, h in receipt['outputs'].items(): assert sha(D / 'enumeration' / n) == h
        return
    entries = []; started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(enumeration.one, a, f): a for a in read(D / 'scenes.json')['units']}
        for future in as_completed(futures):
            a = futures[future]; folder, result = future.result()
            entries.append({'unit_id': a['unit_id'], 'completion': str(folder / 'completion.json'),
                            'sha256': sha(folder / 'completion.json')})
    entries.sort(key=lambda a: a['unit_id'])
    atomic_json(D / 'enumeration/all_index.json', entries, immutable=True)
    atomic_json(path, {'status': 'COMPLETE_LABEL_BLIND_ENUMERATION', 'programs': len(entries),
                'wall_seconds': time.monotonic()-started,
                'outputs': {'all_index.json': sha(D / 'enumeration/all_index.json')}}, immutable=True)


def freeze_address():
    enumerate_all()
    stage = D / 'address_stage'; stage.mkdir(exist_ok=True)
    if (stage / 'freeze.json').exists():
        execution.D = D; execution.STAGE = stage; execution.validate_freeze(); return
    scenes = {a['unit_id']: a for a in read(D / 'scenes.json')['units']}
    jobs = []; unbound = failures = 0
    for a in read(D / 'enumeration/all_index.json'):
        assert sha(a['completion']) == a['sha256']
        c = read(a['completion']); owner = scenes[a['unit_id']]
        failures += bool(c['failure'])
        for o in c['result']['obligations']:
            if not o['identity_valid']: unbound += 1; continue
            jobs.append({'dataset': 'boundary', 'unit_id': o['id'], 'scene_id': owner['unit_id'],
                         'entry': owner['entry'], 'source_path': owner['source_path'],
                         'source_sha256': owner['source_sha256'], 'group': owner['group'],
                         'query_registration': o['query_registration'], 'source_span': o['source_span'],
                         'source_access_id': o['source_access_id'], 'role': o['role']})
    jobs.sort(key=lambda a: a['unit_id'])
    atomic_json(stage / 'jobs.json', {'units': jobs, 'unbound_obligations_retained_as_UNKNOWN': unbound,
                'failed_enumeration_scenes_retained_as_UNKNOWN': failures}, immutable=True)
    atomic_json(stage / 'library_references.json', {}, immutable=True)
    old = read(PARENT / 'address_stage/freeze.json')
    files = {p: h for p, h in old['files'].items()
             if p.endswith('.py') and not p.endswith(('freeze_external_address.py', 'execute_external_address.py'))}
    for path in [Path(__file__), S / 'execute_external_address.py', D / 'scenes.json',
                 D / 'enumeration/freeze.json', D / 'enumeration/all_receipt.json',
                 D / 'enumeration/all_index.json', stage / 'jobs.json', stage / 'library_references.json']:
        files[str(path.relative_to(ROOT))] = sha(path)
    files.update({str(Path(a['source_path']).relative_to(ROOT)): a['source_sha256'] for a in scenes.values()})
    f = {**old, 'status': 'FROZEN_CURRENT_H_BOUNDARY_ADDRESS', 'files': files,
         'stage_programs': len(jobs), 'external_scenes': 448, 'unbound_obligations': unbound,
         'new_Eva_neural_training': 0,
         'worker_environment_sha256': hashlib.sha256(json.dumps(address.environment(), sort_keys=True).encode()).hexdigest()}
    for p, h in files.items(): assert sha(ROOT / p) == h, p
    atomic_json(stage / 'freeze.json', f, immutable=True)


def run_static(scope='all'):
    freeze_address(); native.verify_tools()
    cfg = native.signed(D / 'Eva_config.json')
    identity = {'config_sha256': sha(D / 'Eva_config.json'),
                'tool_registry_sha256': sha(native.WORK / 'tool_registry.json'),
                'execution_script_sha256': sha(native.__file__), 'entry_adapter_sha256': sha(__file__)}
    units = read(D / ('enumeration_preflight_manifest.json' if scope=='preflight' else 'scenes.json'))['units']
    records = []; started = time.monotonic()
    def one(a):
        folder = D / 'eva' / hashlib.sha256(a['unit_id'].encode()).hexdigest()
        receipt, reused = native.run_unit('frama_c_eva', {'scene_id': a['unit_id'],
            'source_path': a['source_path'], 'source_sha256': a['source_sha256']}, folder, cfg, identity)
        return {'unit_id': a['unit_id'], 'prediction': receipt['prediction'] if receipt['prediction'] is not None else 'UNKNOWN',
                'failure': receipt['failure_reason'], 'receipt': str(folder / 'receipt.json'),
                'receipt_sha256': sha(folder / 'receipt.json'), 'cached_this_invocation': reused}
    # Native outcomes are per-scene; unlike OverflowGen no query-line attribution.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(one, units): records.append(result)
    records.sort(key=lambda a: a['unit_id'])
    target = D / ('Eva_preflight_predictions.json' if scope=='preflight' else 'Eva_predictions.json')
    if not target.exists():
        atomic_json(target, {'units': records, 'native_policy': 'oldEva non-RTE selectors, entry fixture'}, immutable=True)
    else:
        old = read(target)['units']
        assert [{k:v for k,v in r.items() if k!='cached_this_invocation'} for r in old] == [
               {k:v for k,v in r.items() if k!='cached_this_invocation'} for r in records]
    atomic_json(D / ('Eva_preflight_receipt.json' if scope=='preflight' else 'Eva_receipt.json'),
                {'status': 'COMPLETE', 'programs': len(units),
                'predictions_sha256': sha(target), 'new_NATIVE_not_RTE': True}, immutable=True)
    execution.D = D; execution.STAGE = D / 'address_stage'
    prior = sys.argv; sys.argv = [str(S / 'execute_external_address.py'), '--scope', scope]
    try: execution.main()
    finally: sys.argv = prior
    if scope=='all': aggregate_all()
    print(json.dumps({'status': 'BOUNDARY_STATIC_COMPLETE', 'wall_seconds_this_resume': time.monotonic()-started}))


def aggregate_all():
    stage = D / 'address_stage'; f = read(stage / 'freeze.json')
    reader = TraceReader(stage / 'freeze.json'); assert reader.failure is None
    jobs = {a['unit_id']: a for a in read(stage / 'jobs.json')['units']}
    idx = {a['unit_id']: a for a in read(stage / 'all_unit_index.json')}
    confirmed = {}
    for sid, a in jobs.items():
        entry = idx[sid]
        assert sha(entry['completion']) == entry['sha256']
        reg = {k:a[k] for k in ['unit_id', 'source_path', 'source_sha256', 'entry', 'query_registration', 'source_access_id']}
        reg['query_span'] = {k:a['source_span'][k] for k in ['start_byte', 'end_byte']}
        confirmed[sid] = reader.read(entry['completion'], reg, entry['sha256'])
    eva = {a['unit_id']: a for a in read(D / 'Eva_predictions.json')['units']}
    outcomes = []
    for a in read(D / 'enumeration/all_index.json'):
        c = read(a['completion']); inventory = c['result']; observations = []
        for o in inventory['obligations']:
            dr = confirmed.get(o['id'], {'D_trace':'UNKNOWN'})
            cc = read(idx[o['id']]['completion']) if o['id'] in idx else None
            rr = cc['result'] if cc else {}
            observations.append({'id': o['id'], 'identity_valid': o['identity_valid'],
                'task_scope_valid': o['task_scope_valid'], 'D_pos': 1 if dr['D_trace']==1 else 'UNKNOWN',
                'D_trace': dr['D_trace'], 'complete_trace': bool(dr['D_trace']==0),
                'query_visited': bool(dr.get('query_visit_count',0)>0), 'failure': cc['failure'] if cc else None})
        scene = aggregate_scene(observations, bool(inventory['inventory_complete'] and not c['failure']))
        outcomes.append({'unit_id': a['unit_id'], 'D_trace': scene['D_trace'], 'D_pos': scene['D_pos'],
                         'Eva': eva[a['unit_id']]['prediction'], 'inventory_complete': inventory['inventory_complete'],
                         'enumeration_failure': c['failure'], 'D_reasons': scene['reasons'],
                         'Eva_failure': eva[a['unit_id']]['failure'], 'full_C_program_safety': 'UNKNOWN'})
    atomic_json(D / 'confirmed_obligation_readouts.json', confirmed, immutable=True)
    atomic_json(D / 'scene_evidence.json', {'units': outcomes}, immutable=True)
    atomic_json(D / 'static_receipt.json', {'status': 'COMPLETE', 'programs': len(outcomes),
                'D_counts': dict(collections.Counter(str(a['D_trace']) for a in outcomes)),
                'Eva_counts': dict(collections.Counter(str(a['Eva']) for a in outcomes)),
                'files': {str(p.relative_to(ROOT)): sha(p) for p in [D / 'scene_evidence.json',
                  D / 'confirmed_obligation_readouts.json', D / 'Eva_receipt.json', stage / 'all_execution_receipt.json']},
                'labels_or_scores_used': False, 'full_C_program_safety': 'UNKNOWN'}, immutable=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', choices=['prepare','enumerate','freeze','preflight','run'], required=True)
    args = ap.parse_args()
    {'prepare':prepare, 'enumerate':enumerate_all, 'freeze':freeze_address,
     'preflight':lambda:run_static('preflight'), 'run':run_static}[args.phase]()
