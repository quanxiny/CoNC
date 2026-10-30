"""Freeze external obligations and unchanged numerical/trace producers."""
from pathlib import Path
import hashlib, json, sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from external_eval.boundary_optimization.common import read, sha, atomic_json
from execute_address_diagnostic import environment
D = ROOT / 'external_eval/tifs_runs/boundary_external_20261021'
B = ROOT / 'external_eval/tifs_runs/boundary_binding_20261021'
S = Path(__file__).parent


def main():
    stage = D / 'address_stage'; stage.mkdir(exist_ok=True)
    if (stage / 'freeze.json').exists():
        f = read(stage / 'freeze.json')
        for n, h in f['files'].items(): assert sha(ROOT / n) == h, n
        print('VERIFIED_EXTERNAL_FREEZE'); return
    scenes = {a['unit_id']: a for a in read(D / 'scenes.json')['units']}
    enum = read(D / 'enumeration/all_receipt.json'); assert enum['status'] == 'COMPLETE_LABEL_BLIND_ENUMERATION'
    jobs = []; unbound = 0
    for a in read(D / 'enumeration/all_index.json'):
        assert sha(a['completion']) == a['sha256']; c = read(a['completion']); r = c['result']; owner = scenes[a['unit_id']]
        for o in r['obligations']:
            if not o['identity_valid']:
                unbound += 1; continue
            jobs.append({'dataset': owner['dataset'], 'unit_id': o['id'], 'scene_id': owner['unit_id'],
                'source_path': owner['source_path'], 'source_sha256': owner['source_sha256'], 'entry': owner['entry'],
                'group': owner['group'], 'query_registration': o['query_registration'],
                'enumeration_completion': a['completion'], 'enumeration_completion_sha256': a['sha256'],
                'source_span': o['source_span'], 'role': o['role'], 'previously_exposed': True})
    jobs.sort(key=lambda a: a['unit_id']); assert len(jobs) == 1836 and unbound == 1320
    atomic_json(stage / 'jobs.json', {'units': jobs, 'unbound_obligations_retained_as_UNKNOWN': unbound,
        'failed_enumeration_scenes_retained_as_UNKNOWN': enum['process_failures']}, immutable=True)
    old = read(B / 'address_pilot/attempt_4a768be154f1/freeze.json')
    for name in ['scene_aggregation_6fa55ac3fe88.json', 'enumeration_6fa55ac3fe88.json']:
        r = read(D / 'root_acceptance' / name)
        assert r['status'] == 'PASS' and r['adapter_sha256'] == sha(S / 'external_address_adapter.py')
    protocol = read(D / 'protocol_draft.json')
    protocol.update(status='FROZEN_AFTER_ROOT32_AGGREGATION_AND14_ENUMERATION_PASS', classification_batch_authorized=True,
        obligations=3156, address_jobs=1836, unbound_obligations=1320,
        enumeration_process_failure='1 SIGSEGV retainedUNKNOWN, no replacement/retry or scenario deletion',
        FormAI_scope='previously frozen neural-processable167 program cohort; not fullFormAI-v2 or whole4096sample',
        FormAI_prediction_before_address_results='complete_inventory0/167; sceneD_trace0 impossible under frozen aggregation; no printf skipping',
        MIT_missing_neural_scope='105 registered scenes outside1059 main; standaloneD coverage only, never fair-main gain',
        analysis_interface='analyze_external_address.py --stage external_eval/tifs_runs/boundary_external_20261021',
        labels_or_scores_used_for_enumeration_or_policy=False)
    atomic_json(D / 'protocol.json', protocol, immutable=True)
    source_hashes = {a['source_sha256'] for a in jobs}
    registry = read(B / 'address_full_4a768be154f1/source_library_registry.json')
    refs = {k: v for k, v in registry.items() if k in source_hashes}
    for ref in refs.values():
        for n, h in ref['files'].items(): assert sha(n) == h, n
    atomic_json(stage / 'library_references.json', refs, immutable=True)
    # Keep the accepted dependency/runtime identity, without unrelated old C files.
    files = {n: h for n, h in old['files'].items() if '/sources/' not in n}
    paths = [D / n for n in ['identity.json', 'scenes.json', 'input_receipt.json', 'protocol.json',
             'enumeration_preflight_manifest.json', 'enumeration/freeze.json', 'enumeration/all_receipt.json',
             'enumeration/all_index.json', 'root_acceptance/scene_aggregation_6fa55ac3fe88.json',
             'root_acceptance/enumeration_6fa55ac3fe88.json']]
    paths += [stage / 'jobs.json', stage / 'library_references.json', Path(__file__), S / 'execute_external_address.py',
              S / 'external_address_adapter.py', S / 'address_trace_readout.py', S / 'execute_address_full.py']
    for p in paths: files[str(p.relative_to(ROOT))] = sha(p)
    for a in scenes.values():
        p = Path(a['source_path']); assert sha(p) == a['source_sha256']; files[str(p.relative_to(ROOT))] = a['source_sha256']
    f = {**old, 'status': 'FROZEN_EXTERNAL_RETROSPECTIVE_ADDRESS', 'files': files,
        'stage_programs': 1836, 'external_scenes': 1331, 'unbound_obligations': 1320,
        'worker_environment_sha256': hashlib.sha256(json.dumps(environment(), sort_keys=True).encode()).hexdigest(),
        'aggregation_sha256': sha(S / 'external_address_adapter.py'), 'trace_sha256': sha(S / 'address_trace_readout.py'),
        'limits': {'workers': 8, 'wall_seconds': 60, 'cpu_seconds': 50, 'address_space_bytes': 3 * 1024**3}}
    for n, h in f['files'].items(): assert sha(ROOT / n) == h, n
    atomic_json(stage / 'freeze.json', f, immutable=True)
    print(json.dumps({'stage': str(stage), 'jobs': len(jobs), 'unbound': unbound, 'freeze_sha256': sha(stage / 'freeze.json'), 'initial_source_library_refs': len(refs)}))


if __name__ == '__main__': main()
