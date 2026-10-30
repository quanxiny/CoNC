"""External obligations: unchanged isolated address worker and trace reader."""
from pathlib import Path
import argparse, hashlib, json, os, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from external_eval.boundary_optimization.common import read, sha, atomic_json, write_csv
import execute_address_diagnostic as worker
from execute_address_full import certificate
from execute_binding_pilot import descendants, pid_stamp
from address_trace_readout import TraceReader
from external_address_adapter import aggregate_scene
D = ROOT / 'external_eval/tifs_runs/boundary_external_20261021'
STAGE = D / 'address_stage'


def validate_freeze():
    f = read(STAGE / 'freeze.json')
    for n, h in f['files'].items(): assert sha(ROOT / n) == h, n
    for n, h in f['runtime_files'].items(): assert sha(n) == h, n
    assert hashlib.sha256(json.dumps(worker.environment(), sort_keys=True).encode()).hexdigest() == f['worker_environment_sha256']
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--scope', choices=['preflight', 'all'], required=True)
    scope = ap.parse_args().scope; f = validate_freeze()
    jobs = read(STAGE / 'jobs.json')['units']
    if scope == 'preflight':
        scenes = {a['unit_id'] for a in read(D / 'enumeration_preflight_manifest.json')['units']}
        jobs = [a for a in jobs if a['scene_id'] in scenes]
    receipt = STAGE / (scope + '_execution_receipt.json')
    if receipt.exists():
        r = read(receipt); assert r['freeze_sha256'] == sha(STAGE / 'freeze.json')
        for n, h in r['outputs'].items(): assert sha(STAGE / n) == h, n
        for a in read(STAGE / (scope + '_unit_index.json')):
            assert sha(a['completion']) == a['sha256']; c = read(a['completion'])
            worker.validate(Path(a['completion']).parent, c['request'])
        print('VERIFIED_CACHED_EXTERNAL_ADDRESS'); return
    start = time.monotonic(); entries = []; rows = []
    registry = dict(read(STAGE / 'library_references.json'))
    registry_lock = threading.Lock(); locks = {a['source_sha256']: threading.Lock() for a in jobs}
    atomic_json(STAGE / (scope + '_controller_start.json'), {'pid': os.getpid(), 'jobs': len(jobs),
        'scope': scope, 'stage': str(STAGE), 'freeze_sha256': sha(STAGE / 'freeze.json'), 'labels_or_model_scores_loaded': False})
    print(json.dumps({'pid': os.getpid(), 'scope': scope, 'jobs': len(jobs)}), flush=True)
    def run(a):
        with locks[a['source_sha256']]:
            folder = STAGE / 'units' / hashlib.sha256(a['unit_id'].encode()).hexdigest()
            if (folder / 'completion.json').exists():
                ref = read(folder / 'completion.json')['request'].get('library_binding_reference')
            else:
                with registry_lock: ref = registry.get(a['source_sha256'])
            folder, c = worker.one(a, STAGE, f, None, ref)
            new = certificate(a, folder, c)
            if new:
                with registry_lock: registry[a['source_sha256']] = new
            return folder, c
    def stop(signum, frame):
        for pid, stamp in reversed(list(descendants(os.getpid()).items())):
            if pid != os.getpid() and pid_stamp(pid) == stamp:
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run, a): a for a in jobs}
        for future in as_completed(futures):
            a = futures[future]; folder, c = future.result(); r = c['result']
            entries.append({'unit_id': a['unit_id'], 'scene_id': a['scene_id'], 'completion': str(folder / 'completion.json'), 'sha256': sha(folder / 'completion.json')})
            rows.append({'dataset': a['dataset'], 'scene_id': a['scene_id'], 'unit_id': a['unit_id'],
                'D_pos': r['potential_verdict'], 'failure': c['failure'], 'exit_code': c['exit_code'],
                'reasons': ';'.join(r.get('unknown_reasons', [])), 'CPP_created': r.get('CPP_created', False),
                'elapsed_seconds': c['elapsed_seconds']})
            if len(entries) % 32 == 0 or len(entries) == len(jobs):
                progress = {'status': 'RUNNING' if len(entries) < len(jobs) else 'COMPLETE',
                    'pid': os.getpid(), 'done': len(entries), 'jobs': len(jobs), 'wall_seconds': time.monotonic() - start,
                    'potential_count': sum(a['D_pos'] == 1 for a in rows), 'process_failures': sum(bool(a['failure']) for a in rows),
                    'CPP_created': sum(a['CPP_created'] for a in rows)}
                atomic_json(STAGE / (scope + '_progress.json'), progress); print(json.dumps(progress), flush=True)
    entries.sort(key=lambda a: a['unit_id']); rows.sort(key=lambda a: a['unit_id'])
    names = [scope + s for s in ['_unit_index.json', '_summary.csv', '_library_registry.json']]
    atomic_json(STAGE / names[0], entries, immutable=True); write_csv(STAGE / names[1], rows)
    atomic_json(STAGE / names[2], registry, immutable=True)
    # Validate each trace identity without reading labels or baseline scores.
    reader = TraceReader(STAGE / 'freeze.json'); results = {}; byjob = {a['unit_id']: a for a in jobs}
    for a in entries:
        c = read(a['completion']); r = c['result']; reg = dict(byjob[a['unit_id']])
        reg['query_span'] = r.get('query_linkage', {}).get('unique_access', {}).get('source_span', {})
        reg['source_access_id'] = r.get('query_linkage', {}).get('source_access_id')
        t = reader.read(a['completion'], reg, a['sha256'])
        assert r['potential_verdict'] != 1 or t['D_trace'] == 1, (a['unit_id'], t)
        results[a['unit_id']] = t
    readout_name = scope + '_readouts.json'
    atomic_json(STAGE / readout_name, results, immutable=True); names.append(readout_name)
    atomic_json(receipt, {'status': 'COMPLETE_EXTERNAL_FROZEN_ADDRESS', 'freeze_sha256': sha(STAGE / 'freeze.json'),
        'jobs': len(entries), 'wall_seconds': time.monotonic() - start,
        'potential_count': sum(a['D_pos'] == 1 for a in rows), 'process_failures': sum(bool(a['failure']) for a in rows),
        'CPP_created': sum(a['CPP_created'] for a in rows), 'new_Eva_neural_training': 0,
        'labels_or_model_scores_loaded': False, 'outputs': {n: sha(STAGE / n) for n in names}}, immutable=True)
    print('EXTERNAL_ADDRESS_COMPLETE', flush=True)


if __name__ == '__main__': main()
