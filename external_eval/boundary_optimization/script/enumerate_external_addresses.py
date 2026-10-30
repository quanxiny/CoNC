"""Isolated label-blind source obligation enumeration; no numerical inference."""
from pathlib import Path
import argparse, collections, hashlib, json, os, resource, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from external_eval.boundary_optimization.common import read, sha, atomic_json, write_csv
D = ROOT / 'external_eval/tifs_runs/boundary_external_20261021'
S = Path(__file__).parent
BASE = __import__('sys').executable


def worker(request_path, folder):
    q = read(request_path)
    resource.setrlimit(resource.RLIMIT_CPU, (50, 51))
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from external_address_adapter import enumerate_scene
    a = q['unit']
    raw = Path(a['source_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == a['source_sha256']
    r = enumerate_scene(raw.decode(), a)
    atomic_json(Path(folder) / 'result.json', r, immutable=True)


def freeze():
    folder = D / 'enumeration'; folder.mkdir(exist_ok=True)
    p = folder / 'freeze.json'
    if p.exists():
        f = read(p)
        for n, h in f['files'].items():
            assert sha(ROOT / n) == h, n
        return f
    files = [Path(__file__), S / 'external_address_adapter.py', S / 'query_access_linkage.py',
             D / 'scenes.json', D / 'enumeration_preflight_manifest.json', D / 'input_receipt.json']
    for name in ['scene_aggregation_6fa55ac3fe88.json', 'enumeration_6fa55ac3fe88.json']:
        p0 = D / 'root_acceptance' / name
        a = read(p0)
        assert a['status'] == 'PASS' and a['adapter_sha256'] == sha(S / 'external_address_adapter.py')
        files.append(p0)
    f = {'status': 'FROZEN_LABEL_BLIND_ENUMERATION', 'files': {str(p.relative_to(ROOT)): sha(p) for p in files},
         'python': BASE, 'python_sha256': sha(BASE), 'workers': 8, 'wall_per_unit': 60,
         'address_inference_or_CPP': False, 'labels_scores_or_official_vulnerability_lines_used': False}
    atomic_json(p, f, immutable=True)
    return f


def one(a, f):
    folder = D / 'enumeration/units' / hashlib.sha256(a['unit_id'].encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    q = {'unit': a, 'freeze_sha256': sha(D / 'enumeration/freeze.json')}
    cp = folder / 'completion.json'
    if cp.exists():
        c = read(cp)
        assert c['request'] == q
        for n, h in c['artifacts'].items():
            assert sha(folder / n) == h, n
        return folder, c
    assert not any(folder.iterdir()), 'partial enumeration is preserved; explicit recovery required'
    atomic_json(folder / 'request.json', q, immutable=True)
    start = time.monotonic(); failure = None
    with (folder / 'stdout.txt').open('w') as out, (folder / 'stderr.txt').open('w') as err:
        proc = subprocess.Popen([BASE, str(Path(__file__)), '--worker', str(folder / 'request.json'), str(folder)],
                                cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); code = proc.wait(); failure = 'WALL_TIMEOUT'
    if code != 0:
        failure = failure or ('EXIT_CODE:' + str(code))
    if failure:
        r = {'unit_id': a['unit_id'], 'source_sha256': a['source_sha256'], 'obligations': [],
             'inventory_complete': False, 'context_blockers': [failure], 'classification': 'UNKNOWN',
             'full_C_program_safety': 'UNKNOWN'}
        atomic_json(folder / 'result.json', r, immutable=True)
    else:
        r = read(folder / 'result.json')
    c = {'request': q, 'exit_code': code, 'failure': failure, 'elapsed_seconds': time.monotonic() - start,
         'result': r, 'artifacts': {n: sha(folder / n) for n in ['request.json', 'result.json', 'stdout.txt', 'stderr.txt']}}
    atomic_json(cp, c, immutable=True)
    return folder, c


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3]); return
    ap = argparse.ArgumentParser(); ap.add_argument('--scope', choices=['preflight', 'all'], required=True)
    args = ap.parse_args(); f = freeze(); assert sha(BASE) == f['python_sha256']
    manifest = read(D / ('enumeration_preflight_manifest.json' if args.scope == 'preflight' else 'scenes.json'))['units']
    receipt = D / 'enumeration' / (args.scope + '_receipt.json')
    if receipt.exists():
        r = read(receipt); assert r['freeze_sha256'] == sha(D / 'enumeration/freeze.json')
        for n, h in r['outputs'].items(): assert sha(D / 'enumeration' / n) == h
        for a in read(D / 'enumeration' / (args.scope + '_index.json')):
            assert sha(a['completion']) == a['sha256']
        print('VERIFIED_CACHED_EXTERNAL_ENUMERATION'); return
    start = time.monotonic(); rows = []; index = []; reasons = collections.Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, a, f): a for a in manifest}
        for future in as_completed(futures):
            a = futures[future]; folder, c = future.result(); r = c['result']; oo = r['obligations']
            index.append({'unit_id': a['unit_id'], 'completion': str(folder / 'completion.json'), 'sha256': sha(folder / 'completion.json')})
            rows.append({'dataset': a['dataset'], 'unit_id': a['unit_id'], 'main_cohort': a['main_cohort'],
                'group': a['group'], 'obligations': len(oo), 'bound': sum(o['identity_valid'] for o in oo),
                'unbound': sum(not o['identity_valid'] for o in oo), 'inventory_complete': r['inventory_complete'],
                'excluded_syntax': len(r.get('excluded_syntax', [])), 'context_blockers': ';'.join(r['context_blockers']),
                'failure': c['failure'], 'source_sha256': a['source_sha256'], 'elapsed_seconds': c['elapsed_seconds']})
            for reason in r['context_blockers']: reasons[a['dataset'], reason] += 1
    index.sort(key=lambda a: a['unit_id']); rows.sort(key=lambda a: (a['dataset'], a['unit_id']))
    names = [args.scope + s for s in ['_index.json', '_summary.csv', '_context_reasons.csv']]
    atomic_json(D / 'enumeration' / names[0], index, immutable=True)
    write_csv(D / 'enumeration' / names[1], rows)
    write_csv(D / 'enumeration' / names[2], [{'dataset': k[0], 'reason': k[1], 'scenes': v} for k, v in sorted(reasons.items())])
    counts = {ds: {'scenes': sum(a['dataset'] == ds for a in rows),
                  'obligations': sum(a['obligations'] for a in rows if a['dataset'] == ds),
                  'bound': sum(a['bound'] for a in rows if a['dataset'] == ds),
                  'complete_inventory_scenes': sum(a['inventory_complete'] for a in rows if a['dataset'] == ds)}
              for ds in ['mit', 'formai']}
    r = {'status': 'COMPLETE_LABEL_BLIND_ENUMERATION', 'scope': args.scope,
         'freeze_sha256': sha(D / 'enumeration/freeze.json'), 'counts': counts,
         'process_failures': sum(bool(a['failure']) for a in rows), 'wall_seconds': time.monotonic() - start,
         'no_numerical_inference_or_CPP': True, 'outputs': {n: sha(D / 'enumeration' / n) for n in names}}
    atomic_json(receipt, r, immutable=True); print(json.dumps(r))


if __name__ == '__main__':
    main()
