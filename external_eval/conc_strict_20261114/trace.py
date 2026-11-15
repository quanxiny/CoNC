"""Reapply the pinned conditional static trace readout to strict benchmark queries.

The workers receive source/query identities only, never labels or model scores.
No corpus C program is compiled to an executable or executed.
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import campaign as c

S = c.ROOT / 'external_eval/boundary_optimization/script'
sys.path.insert(0, str(S))
import execute_address_diagnostic as worker
from address_trace_readout import TraceReader


def main():
    c.identity()
    original_path = c.ROOT / 'external_eval/tifs_runs/boundary_binding_20261021/address_full_4a768be154f1/freeze.json'
    original = c.read(original_path)
    stage = c.OUT / 'trace'; stage.mkdir(exist_ok=True)
    source_path = c.OUT / 'static/sources.json'
    query_path = c.OUT / 'static/queries.json'
    freeze = {'core_sha256': original['core_sha256'], 'factory_sha256': original['factory_sha256'],
              'expression_budget': original['expression_budget'], 'limits': {**original['limits'], 'workers': 8},
              'files': original['files'], 'schema': 'strict-source-pinned-address-query',
              'source_original_freeze_sha256': c.sha(original_path), 'controller_sha256': c.sha(__file__),
              'sources_sha256': c.sha(source_path), 'query_registry_sha256': c.sha(query_path),
              'strict_freeze_sha256': c.sha(c.OUT / 'freeze.json'), 'labels_or_scores_in_worker': False}
    if (stage / 'freeze.json').exists(): assert c.read(stage / 'freeze.json') == freeze
    else: c.save(stage / 'freeze.json', freeze)
    reader = TraceReader(stage / 'freeze.json'); assert reader.failure is None, reader.failure
    sources = {r['source_case']: r for r in c.read(source_path)}
    entries = []
    for q in c.read(query_path):
        s = sources[q['source_case']]
        entries.append({'unit_id': q['sample_id'], 'source_path': s['source_path'], 'source_sha256': s['source_sha256'],
                        'entry': 'main', 'query_registration': {'family_id': q['group'], 'query_line': q['query_line'],
                        'row': q['source_row'], 'scene_id': q['sample_id'], 'source_case': q['source_case']}})
    assert len(entries) == 11769 and len({r['unit_id'] for r in entries}) == 11769
    manifest = {'units': entries, 'no_labels_or_model_scores': True}
    if (stage / 'manifest.json').exists(): assert c.read(stage / 'manifest.json') == manifest
    else: c.save(stage / 'manifest.json', manifest)
    started = time.monotonic(); rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(worker.one, entry, stage, freeze, None, None): entry for entry in entries}
        for i, future in enumerate(as_completed(futures), 1):
            entry = futures[future]; folder, receipt = future.result(); result = receipt['result']
            registration = dict(entry)
            registration['query_span'] = result.get('query_linkage', {}).get('unique_access', {}).get('source_span', {})
            registration['source_access_id'] = result.get('query_linkage', {}).get('source_access_id')
            output = reader.read(folder / 'completion.json', registration, c.sha(folder / 'completion.json'))
            if result.get('potential_verdict') == 1: assert output['D_trace'] == 1
            rows.append({'sample_id': entry['unit_id'], 'D_trace': output['D_trace'], 'D_pos': output['D_pos'],
                         'reasons': output['reasons'], 'worker_failure': receipt['failure'],
                         'completion': str(folder / 'completion.json'), 'completion_sha256': c.sha(folder / 'completion.json')})
            if i % 500 == 0 or i == len(entries):
                print(json.dumps({'completed': i, 'total': len(entries), 'seconds': time.monotonic() - started}), flush=True)
    rows.sort(key=lambda x: x['sample_id']); c.save(stage / 'predictions.json', rows)
    c.save(stage / 'completion.json', {'status': 'COMPLETE', 'n': len(rows), 'sha256': c.sha(stage / 'predictions.json'),
           'freeze_sha256': c.sha(stage / 'freeze.json'), 'D_trace_0': sum(r['D_trace'] == 0 for r in rows),
           'D_trace_1': sum(r['D_trace'] == 1 for r in rows), 'D_trace_UNKNOWN': sum(r['D_trace'] == 'UNKNOWN' for r in rows),
           'labels_or_model_scores_in_worker': False, 'target_program_execution': False,
           'source_program_safety_proved': False, 'elapsed_s': time.monotonic() - started})


if __name__ == '__main__': main()
