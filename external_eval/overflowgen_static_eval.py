#!/usr/bin/env python3
"""OverflowGen family TEST adapter for the shared, frozen native analyzers.

No training, neural inference, source execution or gold-line selection. Native
analysis runs once per source; location attribution restores the original query
unit. All successful and failed native receipts are resumable and immutable.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import json
from pathlib import Path
import time

import static_baselines as native

ROOT = native.ROOT
WORK = native.WORK / 'overflowgen_family_test'
DATA = ROOT / 'datasets/overflowgen_24455320/data/overflow_gen'
ROWS = ROOT / 'reproducibility/frozen_rows/test.txt'
META = ROOT / 'transformer_gru_v76_v2/assets/comparisons/overflowgen/overflow_interfaces_v51/output/row_metadata.jsonl.gz'


def stable_jsonl(path, rows):
    data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)
    if path.exists():
        if path.read_text() != data: raise ValueError('immutable manifest changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(data)


def primary_location(tool, alert):
    """Only native primary diagnostic locations, never path-trace locations."""
    try:
        if tool == 'cppcheck':
            loc = alert['locations'][0]
            return loc['file'], int(loc['line'])
        if tool == 'clang_sa':
            loc = alert['location']
            return alert['native_files'][int(loc['file'])], int(loc['line'])
        path = Path(alert['file'])
        if not path.is_absolute(): path = Path(alert['directory']) / path
        return str(path), int(alert['line'])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def attribute(tool, receipt, source, queries):
    hits = Counter(); outside = []; unresolved = []
    if receipt['status'] != 'PASS':
        return {q: None for q in queries}, receipt['failure_reason'], {}, [], []
    for alert in receipt['selected_alerts']:
        loc = primary_location(tool, alert)
        if loc is None or loc[1] <= 0:
            unresolved.append({'reason': 'MISSING_PRIMARY_LOCATION', 'alert': alert}); continue
        path = Path(loc[0])
        if not path.is_absolute(): path = ROOT / path
        if path.resolve() != Path(source).resolve():
            unresolved.append({'reason': 'FOREIGN_PRIMARY_LOCATION', 'alert': alert}); continue
        if loc[1] in queries: hits[loc[1]] += 1
        else: outside.append({'line': loc[1], 'alert': alert})
    failure = 'UNRESOLVED_TARGET_LOCATION' if unresolved else None
    predictions = {q: None if failure else int(hits[q] > 0) for q in queries}
    return predictions, failure, dict(hits), outside, unresolved


def acceptance():
    WORK.mkdir(parents=True, exist_ok=True)
    source = WORK / 'fixtures/location_binding.c'
    source.parent.mkdir(parents=True, exist_ok=True)
    code = '#include <stdio.h>\nint main(void) {\n  int a[4] = {0};\n  printf("%d", a[1]);\n  a[4] = 7;\n  return 0;\n}\n'
    if source.exists(): assert source.read_text() == code
    else: source.write_text(code)
    native.verify_tools(); config = native.signed(native.CONFIG)
    identity = {'execution_script_sha256': native.sha(native.__file__), 'adapter_sha256': native.sha(__file__),
                'config_sha256': native.sha(native.CONFIG), 'tool_registry_sha256': native.sha(native.WORK/'tool_registry.json')}
    results = []
    for tool in native.TOOLS:
        item = {'scene_id': 'hand:location_binding', 'source_path': str(source), 'source_sha256': native.sha(source)}
        receipt, cached = native.run_unit(tool, item, WORK/'acceptance'/native.digest(identity)[:16]/tool, config, identity)
        pred, failure, hits, outside, unresolved = attribute(tool, receipt, source, [4, 5])
        assert failure is None and pred == {4: 0, 5: 1}, (tool, pred, failure)
        # Native-format synthetic reports exercise secondary trace exclusion,
        # foreign/missing locations and warnings outside the registered queries.
        if tool == 'cppcheck':
            alert = {'locations': [{'file': str(source), 'line': '5'}, {'file': str(source), 'line': '4'}]}
        elif tool == 'clang_sa':
            alert = {'location': {'file': 0, 'line': 5}, 'native_files': [str(source)],
                     'path': [{'location': {'file': 0, 'line': 4}}]}
        else: alert = {'directory': str(source.parent), 'file': source.name, 'line': '5'}
        fake = {'status': 'PASS', 'selected_alerts': [alert]}
        assert attribute(tool, fake, source, [4, 5])[0] == {4: 0, 5: 1}
        assert attribute(tool, fake, source, [4])[0] == {4: 0}
        assert attribute(tool, fake, source.with_name('other.c'), [4, 5])[1] == 'UNRESOLVED_TARGET_LOCATION'
        assert attribute(tool, {'status': 'PASS', 'selected_alerts': [{}]}, source, [4])[0] == {4: None}
        assert attribute(tool, {'status': 'UNKNOWN', 'failure_reason': 'WALL_TIMEOUT'}, source, [4])[0] == {4: None}
        results.append({'tool': tool, 'predictions': pred, 'target_alerts': receipt['target_alert_count'],
                        'outside_query_alerts': len(outside), 'cached': cached})
    # cached is operational metadata, not part of the immutable acceptance result.
    native.immutable(WORK/'acceptance.json', {'status': 'PASS', 'identity': identity,
                     'results': [{k:v for k,v in r.items() if k != 'cached'} for r in results],
                     'cases': ['safe versus unsafe in one file', 'primary versus secondary trace',
                               'outside inventory', 'foreign or absent location', 'native UNKNOWN propagation']})
    print(json.dumps({'acceptance': 'PASS', 'tools': results}), flush=True)


def prepare():
    import make_manifest as front
    WORK.mkdir(parents=True, exist_ok=True)
    acceptance_receipt = native.signed(WORK/'acceptance.json')
    assert acceptance_receipt['identity']['adapter_sha256'] == native.sha(__file__)
    target = set(map(int, ROWS.read_text().split()))
    with gzip.open(META, 'rt') as stream:
        metadata = [r for line in stream if (r := json.loads(line))['row'] in target]
    assert len(metadata) == len(target) == 11577
    data_hashes = {}; queries = []; by_split = defaultdict(list)
    for r in metadata: by_split[r['original_split']].append(r)
    for split, subset in by_split.items():
        path = DATA / (split + '_data.json')
        data_hashes[str(path.relative_to(ROOT))] = native.sha(path)
        raw = native.read(path)
        for r in subset:
            assert r['family_split'] == 'test'
            item = raw[str(r['original_sample_id'])]
            assert int(item['label']) == r['label']
            index = item['key_line']; lines = item['line_num']
            assert isinstance(index, int) and 0 <= index < len(lines)
            line = int(lines[index]); assert line > 0
            assert int(Path(item['file_path']).stem) == int(r['source_case'])
            queries.append({'scene_id': f"og24455320:{split}:{r['original_sample_id']}",
                'row': r['row'], 'source_case': int(r['source_case']), 'family_id': r['family_id'],
                'label': r['label'], 'query_line': line, 'original_split': split})
        del raw
    queries.sort(key=lambda r: r['row'])
    grouped = defaultdict(list)
    for q in queries: grouped[q['source_case']].append(q)
    sources = []; audits = Counter()
    for case, items in sorted(grouped.items()):
        n = f'{case:09d}'; raw_path = DATA/'source-code'/n[:3]/n[3:6]/n[6:]/f'{case}.c'
        source = raw_path.read_bytes(); raw_hash = native.sha(raw_path)
        tree, functions = front.function_info(source, '.c')
        reasons = []
        if tree.root_node.has_error: reasons.append('SOURCE_PARSE_ERROR')
        if [f['name'] for f in functions] != ['main']: reasons.append('ENTRY_STRUCTURE_MISMATCH')
        sanitized, names, sanitizer_reasons = front.neutralize(source, tree, functions)
        reasons += sanitizer_reasons
        assert source.count(b'\n') == sanitized.count(b'\n')
        for q in items: assert q['query_line'] <= len(source.splitlines())
        assert len({q['query_line'] for q in items}) == len(items)
        source_hash = front.sha(sanitized)
        destination = WORK/'sources'/f'{source_hash}.c'
        if destination.exists(): assert native.sha(destination) == source_hash
        else: destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(sanitized)
        assert len({q['family_id'] for q in items}) == 1
        sources.append({'scene_id': f'og-source:{case}', 'source_case': case,
            'family_id': items[0]['family_id'], 'source_path': str(destination.resolve()),
            'source_sha256': source_hash, 'raw_path': str(raw_path.relative_to(ROOT)), 'raw_sha256': raw_hash,
            'status': 'UNKNOWN' if reasons else 'READY', 'failure_reason': ';'.join(reasons),
            'registered_queries': sorted(q['query_line'] for q in items), 'comments_removed': True,
            'renamed_identifiers': len(names), 'line_count': len(source.splitlines())})
        audits['programs'] += 1; audits['mixed_label_programs'] += len({q['label'] for q in items}) > 1
        audits['source_failures'] += bool(reasons)
    assert len(sources) == 9838 and sum(q['label'] for q in queries) == 5015
    stable_jsonl(WORK/'queries.jsonl', queries); stable_jsonl(WORK/'sources.jsonl', sources)
    protocol = {'schema': 'overflowgen-family-test-native-v1', 'split': 'historical frozen family TEST; not author TEST',
        'test_rows_sha256': native.sha(ROWS), 'row_metadata_sha256': native.sha(META), 'original_json_hashes': data_hashes,
        'queries_sha256': native.sha(WORK/'queries.jsonl'), 'sources_sha256': native.sha(WORK/'sources.jsonl'),
        'adapter_sha256': native.sha(__file__), 'sanitizer_sha256': native.sha(front.__file__),
        'execution_script_sha256': native.sha(native.__file__), 'config_sha256': native.sha(native.CONFIG),
        'tool_registry_sha256': native.sha(native.WORK/'tool_registry.json'), 'acceptance_sha256': native.sha(WORK/'acceptance.json'),
        'tools': list(native.TOOLS), 'target_slices': len(queries), 'positive_slices': sum(q['label'] for q in queries),
        'source_audit': dict(audits), 'family_count': len({q['family_id'] for q in queries}),
        'input': 'whole original source; comments blanked, answer-like identifiers consistently renamed, hash-only filename; numbers/types/guards unchanged',
        'primary_unit': 'original published candidate query slice',
        'query_binding': 'original one-based line_num[key_line]; only native PRIMARY diagnostic line in the same sanitized source; never line_label or secondary traces',
        'outside_query': 'retain separately; no file-wide broadcasting or oracle-based remapping',
        'unresolved_target_location': 'all queries in source UNKNOWN; never safe',
        'native_failure': 'all queries in source UNKNOWN, even when partial alerts exist',
        'secondary_unit': 'source program OR over mapped registered queries and OR over their labels; unknown if any query unknown',
        'negative': 'completed native analysis without matched target alert; detector negative, not safety proof',
        'ranking': 'AP/AUROC NA: no continuous score', 'bootstrap': {'replicates': 2000, 'seed': 20261018, 'unit': 'frozen coarse family; paired common draws'},
        'independence': 'frozen coarse lexical families, not certified semantic/generator separation; normalized train/test slice templates overlap',
        'training_steps': 0, 'neural_inference': False, 'new_test_debug_families': [],
        'label_input': False, 'sample_program_execution': False}
    native.immutable(WORK/'protocol.json', protocol)
    print(json.dumps({'status': 'FROZEN', 'audit': audits, 'slices': len(queries), 'families': protocol['family_count']}), flush=True)


def verify_protocol():
    p = native.signed(WORK/'protocol.json')
    checks = {ROWS: p['test_rows_sha256'], META: p['row_metadata_sha256'],
        WORK/'queries.jsonl': p['queries_sha256'], WORK/'sources.jsonl': p['sources_sha256'],
        Path(__file__): p['adapter_sha256'], Path(native.__file__): p['execution_script_sha256'],
        native.CONFIG: p['config_sha256'], native.WORK/'tool_registry.json': p['tool_registry_sha256']}
    for path, expected in checks.items():
        if native.sha(path) != expected: raise ValueError('frozen input changed: ' + str(path))
    for path, expected in p['original_json_hashes'].items():
        if native.sha(ROOT/path) != expected: raise ValueError('original JSON changed')
    return p


def run(workers):
    protocol = verify_protocol(); native.verify_tools(); config = native.signed(native.CONFIG)
    sources = native.jsonl(WORK/'sources.jsonl'); queries = native.jsonl(WORK/'queries.jsonl')
    # Raw input remains immutable as well as sanitized tool input.
    for source in sources:
        if native.sha(ROOT/source['raw_path']) != source['raw_sha256']: raise ValueError('raw source changed')
    index = []
    for tool in native.TOOLS:
        identity = {'protocol_sha256': native.sha(WORK/'protocol.json'), 'config_sha256': protocol['config_sha256'],
                    'tool_registry_sha256': protocol['tool_registry_sha256'], 'execution_script_sha256': protocol['execution_script_sha256']}
        run_id = native.digest({'tool': tool, **identity})[:24]
        directory = WORK/'runs'/run_id; directory.mkdir(parents=True, exist_ok=True)
        native.immutable(directory/'run_spec.json', {'tool': tool, 'run_id': run_id, **identity})
        outputs = {}; count = Counter(); started = time.monotonic()
        def one(source):
            if source['status'] != 'READY':
                return source, {'status': 'UNKNOWN', 'failure_reason': source['failure_reason'],
                                'selected_alerts': [], 'elapsed_seconds': 0}, True
            result, cached = native.run_unit(tool, source, directory/'units'/str(source['source_case']), config, identity)
            return source, result, cached
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(one, s) for s in sources]
            for f in as_completed(futures):
                source, result, cached = f.result(); case = source['source_case']
                pred, failure, hits, outside, unresolved = attribute(tool, result, source['source_path'], source['registered_queries'])
                outputs[case] = (source, result, pred, failure, hits, outside, unresolved)
                count['done'] += 1; count['cached'] += cached; count['unknown'] += failure is not None
                if count['done'] % 500 == 0 or count['done'] == len(sources):
                    print(json.dumps({'tool': tool, **count, 'elapsed_seconds': round(time.monotonic()-started, 1)}), flush=True)
        ledger = []; program_rows = []; diagnostics = []
        by_case = defaultdict(list)
        for q in queries: by_case[q['source_case']].append(q)
        for case, (source, result, pred, failure, hits, outside, unresolved) in sorted(outputs.items()):
            unit_path = directory/'units'/str(case)/'receipt.json'
            qrows = by_case[case]
            for q in qrows:
                ledger.append({**q, 'tool': tool, 'run_id': run_id, 'status': 'UNKNOWN' if failure else 'PASS',
                    'prediction': pred[q['query_line']], 'failure_reason': failure or '',
                    'matched_alerts': hits.get(q['query_line'], 0), 'source_target_alerts': len(result['selected_alerts']),
                    'outside_query_alerts': len(outside), 'elapsed_seconds_per_source': result['elapsed_seconds'],
                    'unit_receipt': str(unit_path.relative_to(ROOT)) if unit_path.exists() else ''})
            program_rows.append({'scene_id': source['scene_id'], 'source_case': case, 'family_id': source['family_id'],
                'label': max(q['label'] for q in qrows), 'tool': tool, 'run_id': run_id,
                'status': 'UNKNOWN' if failure else 'PASS', 'prediction': None if failure else max(pred.values()),
                'failure_reason': failure or '', 'registered_queries': len(qrows),
                'target_alerts': len(result['selected_alerts']), 'mapped_alerts': sum(hits.values()),
                'outside_query_alerts': len(outside), 'unresolved_alerts': len(unresolved),
                'elapsed_seconds': result['elapsed_seconds']})
            if outside or unresolved:
                diagnostics.append({'source_case': case, 'outside_inventory': outside, 'unresolved': unresolved})
        ledger.sort(key=lambda r: r['row'])
        stable_jsonl(directory/'location_diagnostics.jsonl', diagnostics)
        # Re-derived ledgers must match an already complete run exactly.
        for name, rows in [('slice_predictions.csv', ledger), ('program_predictions.csv', program_rows)]:
            path = directory/name
            if path.exists():
                tmp = directory/('check_'+name); native.csv_write(tmp, rows)
                if native.sha(path) != native.sha(tmp): raise ValueError('saved prediction differs: '+str(path))
                # Retain the tiny comparison artifact; no source/prediction deletion.
            else: native.csv_write(path, rows)
        receipt = {'tool': tool, 'run_id': run_id, 'identity': identity,
            'slice_predictions_sha256': native.sha(directory/'slice_predictions.csv'),
            'program_predictions_sha256': native.sha(directory/'program_predictions.csv'),
            'location_diagnostics_sha256': native.sha(directory/'location_diagnostics.jsonl'),
            'target_slices': len(ledger), 'target_sources': len(sources),
            'slice_status': dict(Counter(r['status'] for r in ledger)),
            'source_status': dict(Counter(r['status'] for r in program_rows)),
            'source_failures': dict(Counter(r['failure_reason'] for r in program_rows if r['failure_reason'])),
            'total_native_cpu_wall_sum_seconds': sum(r['elapsed_seconds'] for r in program_rows),
            'mapped_alerts': sum(r['mapped_alerts'] for r in program_rows),
            'outside_inventory_alerts': sum(r['outside_query_alerts'] for r in program_rows),
            'unresolved_alerts': sum(r['unresolved_alerts'] for r in program_rows)}
        native.immutable(directory/'run_receipt.json', receipt)
        index.append({'tool': tool, 'run_id': run_id, 'directory': str(directory.relative_to(ROOT)),
                      'receipt_sha256': native.sha(directory/'run_receipt.json')})
        print(json.dumps({'completed': tool, **receipt}), flush=True)
    native.immutable(WORK/'index.json', {'runs': index, 'protocol_sha256': native.sha(WORK/'protocol.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['acceptance', 'prepare', 'run'])
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args(); WORK.mkdir(parents=True, exist_ok=True)
    with (WORK/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'acceptance': acceptance()
        elif args.action == 'prepare': prepare()
        else: run(args.workers)


if __name__ == '__main__': main()
