"""Prepare one shared frozen tensor cache from a reviewed static scene manifest.

Only sanitized single-file units enter the pinned legacy parser. Every original
candidate is formed before scene ownership is applied. Labels remain sidecars.
No source text is printed, and no model is instantiated. The default invocation
processes at most 100 pending units; a later invocation resumes the same ledger.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import legacy_frontend as legacy
from input_contract import PREPROCESS_TIMEOUT_SECONDS, literal_issues, verify_profile

HERE = Path(__file__).resolve().parent
BATCH_SIZE = 32
PARSER_TIMEOUT_SECONDS = 180
DEFERRED_CONTEXT = {'referenced_file_scope_state_context_unresolved',
                    'external_helper_context_review_required'}


def sha(path):
    return legacy.sha(path)


def digest(value):
    return legacy.digest(value)


def save(path, value):
    legacy.save(path, value)


def eligible_scene(scene):
    return (scene.get('record_kind', 'scene') == 'scene' and scene.get('label') in [0, 1]
            and scene.get('status') in ['candidate_pending_frontend', 'context_unresolved']
            and set(scene.get('context_reasons', [])).issubset(DEFERRED_CONTEXT)
            and scene.get('sanitization_accepted', True) and bool(scene.get('source_line_spans')))


def slice_context_rejections(scene, public, graph):
    """Use only existing source spans and original slice/target evidence.

    Global state and external helpers are deferred syntactic flags. An unrelated
    call outside this slice does not become a necessary context dependency.
    This function does not resolve names, expand macros, or infer C semantics.
    """
    reasons = set(scene.get('context_reasons', []))
    facts = scene.get('context_facts', {})
    included = set(public['line_num'])
    failures = []
    if 'external_helper_context_review_required' in reasons:
        calls = facts.get('external_helper_calls')
        if calls is None:
            failures.append('external_helper_context_facts_missing')
        elif any(call['line'] in included for call in calls):
            failures.append('external_helper_in_original_slice_without_supported_context')
    if 'referenced_file_scope_state_context_unresolved' in reasons:
        globals_ = facts.get('file_scope_references')
        if globals_ is None:
            failures.append('file_scope_reference_facts_missing')
        else:
            for fact in globals_:
                if not included.intersection(fact['reference_lines']):
                    continue
                start, end = fact['declaration_line_span']
                if fact['is_extern_or_unknown'] or not set(range(start, end + 1)).issubset(included):
                    failures.append('required_file_scope_definition_absent_or_extern')
    if any(node['target'] != int(node['line'] == public['key_line']) for node in graph['nodes']):
        failures.append('original_target_binding_mismatch')
    return sorted(set(failures))


def proven_original_nontermination(source, original):
    """Conservative rejection proof; never produces replacement condition flags.

    dataset_generator.py:112-175: with every indent zero, the first if/else
    opens area 1/2. Its immediately next line stays N unless it contains '{':
    indent==1 and idx-pred>1 are both false. Later outer iterations have indent
    >=1, so no zero-indented line can ever be assigned and the loop cannot end.
    """
    lines = source.splitlines(keepends=True)
    indents = original['get_indentation'](lines)
    if not lines or any(indents):
        return None
    first = next((i for i, line in enumerate(lines) if 'if' in line or 'else' in line), None)
    if first is None or first + 1 >= len(lines) or '{' in lines[first + 1]:
        return None
    return {'shortcut_reason': 'proven_original_nontermination', 'first_branch_source_line': first + 1,
            'permanently_unassigned_source_line': first + 2,
            'proof': 'all original indentations are zero; first branch successor has no opening brace',
            'original_source_reference': 'dataset_generator.py:112-175'}


def owner(scene, line):
    return any(start <= line <= end for start, end in scene.get('source_line_spans', []))


def prepare_unit(source, table, scenes, original, api, cache_dir, unit_id):
    """Transform one parser result; never receives labels in original functions."""
    with source.open('r', encoding='utf-8', errors='ignore') as f:
        source_text = f.read()
    shortcut = proven_original_nontermination(source_text, original)
    if shortcut:
        return {'status': 'rejected_original_nontermination', **shortcut, 'inventory_complete': False, 'slices': []}
    try:
        with legacy.preprocessing_deadline(PREPROCESS_TIMEOUT_SECONDS):
            result = legacy.public_slices(source, table, original, api)
    except legacy.OriginalPreprocessingTimeout:
        return {'status': 'rejected_preprocessing_timeout', 'inventory_complete': False, 'slices': []}
    except (ValueError, KeyError, IndexError, TypeError):
        return {'status': 'rejected_original_preprocessing', 'inventory_complete': False, 'slices': []}
    arrays = []
    records = []
    directional_edges = legacy.line_edges(table)
    for item in result['slices']:
        key_line = item['key_source_line']
        owners = [scene['scene_id'] for scene in scenes if eligible_scene(scene) and owner(scene, key_line)]
        slice_id = hashlib.sha256(f'{unit_id}:{key_line}'.encode()).hexdigest()
        record = {'slice_id': slice_id, 'key_source_line': key_line, 'owner_scene_ids': owners,
                  'status': item['status'].lower(), 'reason': item.get('reason'), 'tensor_row': None}
        if not owners:
            record['status'] = 'outside_eligible_scene'
        elif len(owners) != 1:
            record['status'] = 'rejected_ambiguous_scene_ownership'
        elif item['status'] == 'ENCODED':
            scene = next(s for s in scenes if s['scene_id'] == owners[0])
            context_failures = slice_context_rejections(scene, item['public'], item['graph'])
            issues = literal_issues(source_text, item['public']['line_num'])
            if context_failures:
                record.update(status='rejected_required_context', context_reasons=context_failures)
            elif issues:
                record.update(status='rejected_unsupported_literal', literal_reasons=sorted({x['reason'] for x in issues}))
            else:
                x, graph = legacy.encode(item['public'], directional_edges, original)
                record.update(status='encoded', tensor_row=len(arrays),
                              source_line_numbers=item['public']['line_num'],
                              line_count=int(x['line_count']), node_count=int(x['node_count']),
                              query_memory_candidate=any(n['target'] and n['op_name'] in
                                  {'ACCESS', 'COPY_DEST', 'COPY_SOURCE', 'ARRAY', 'MALLOC'} for n in graph['nodes']),
                              unknown_node_count=sum(not node['initial_known'] for node in graph['nodes']))
                arrays.append(x)
        records.append(record)
    artifacts = {}
    if arrays:
        cache_dir.mkdir(exist_ok=True)
        destination = cache_dir / (unit_id + '.npz')
        stacked = {key: np.stack([row[key] for row in arrays]) for key in arrays[0]}
        if destination.exists():
            # Recover only byte-value-equivalent output from an interrupted
            # write-before-receipt boundary; never replace differing arrays.
            with np.load(destination, allow_pickle=False) as previous:
                if set(previous.files) != set(stacked) or any(previous[k].dtype != v.dtype or not np.array_equal(previous[k], v) for k, v in stacked.items()):
                    raise ValueError('Existing interrupted unit cache differs')
        else:
            temporary = destination.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, **stacked)
            temporary.replace(destination)
        artifacts[str(destination)] = sha(destination)
        for record in records:
            if record['status'] == 'encoded':
                record['tensor_cache'] = str(destination)
    return {'status': 'processed', 'inventory_complete': True,
            'candidate_source_lines': result['candidate_source_lines'], 'slices': records,
            'accepted_slice_count': len(arrays), 'artifact_sha256': artifacts}


def parser_batch(java, jars, sources, output):
    """Call the reviewed Java main directly, never the destructive wrapper."""
    source_dir = output / 'source'
    source_dir.mkdir(parents=True, exist_ok=False)
    for path in sources:
        destination = source_dir / path.name
        if destination.exists():
            raise ValueError('Sanitized filename collision')
        # Byte-identical copies avoid mutating the manifest files via hardlinks.
        shutil.copyfile(path, destination)
    csv_dir = output / 'csv'  # Parser owns creation; it requires a fresh path.
    argv = [str(java), '-Xmx1024m', '-cp', os.pathsep.join(map(str, jars)),
            'tools.parser.ParserMain', '-outformat', 'csv', '-outdir', str(csv_dir), str(source_dir)]
    log_path = output / 'parser.log'
    try:
        with log_path.open('wb') as log:
            proc = subprocess.run(argv, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=PARSER_TIMEOUT_SECONDS, check=False)
        status = 'parsed' if proc.returncode == 0 else 'parser_failed'
        code = proc.returncode
    except subprocess.TimeoutExpired:
        status, code = 'parser_timeout', None
    return {'status': status, 'returncode': code, 'argv': argv, 'log_sha256': sha(log_path),
            'csv_dir': str(csv_dir), 'source_dir': str(source_dir)}


def scene_outcomes(scenes, units, completed):
    records = []
    by_path = {unit['relative']: key for key, unit in units.items()}
    for scene in scenes:
        record = dict(scene)
        record['primary_processable'] = False  # Methods/seeds must still all succeed.
        record['frontend_processable'] = False
        record['slice_inventory'] = []
        record['slice_inventory_complete'] = False
        record['manifest_status'] = scene.get('status')
        if not eligible_scene(scene):
            status = scene.get('status', 'unknown')
            if status in {'parse_error', 'source_static_parse_error', 'source_processing_timeout'}:
                record['preparation_status'] = status
            elif status == 'label_unresolved':
                record['preparation_status'] = 'label_unresolved'
            elif status in {'context_unresolved', 'candidate_pending_frontend'}:
                record['preparation_status'] = 'context_rejected_by_finite_profile'
            else:
                record['preparation_status'] = status if status != 'unknown' else 'manifest_rejected_or_unresolved'
            record['preparation_reasons'] = scene.get('context_reasons', [])
        else:
            key = by_path.get(scene.get('sanitized_path'))
            result = completed.get(key)
            if result is None:
                record['preparation_status'] = 'not_attempted'
            elif not result['inventory_complete']:
                record['preparation_status'] = result['status']
            else:
                inventory = [s for s in result['slices'] if owner(scene, s['key_source_line'])]
                record['slice_inventory'] = inventory
                record['slice_inventory_complete'] = True
                if not inventory:
                    record['preparation_status'] = 'zero_original_candidate_slices'
                elif all(s['status'] == 'encoded' and s['owner_scene_ids'] == [scene['scene_id']] for s in inventory):
                    record['preparation_status'] = 'ready_for_frozen_inference'
                    record['frontend_processable'] = True
                else:
                    record['preparation_status'] = 'partial_or_rejected_slice_inventory'
        records.append(record)
    return records


def write_views(output, scenes, units, ledger):
    outcomes = scene_outcomes(scenes, units, ledger['completed'])
    scene_temp = output / 'prepared_scenes.tmp'
    with scene_temp.open('w') as f:
        for record in outcomes:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')
    scene_temp.replace(output / 'prepared_scenes.jsonl')
    slice_temp = output / 'slices.tmp'
    with slice_temp.open('w') as f:
        for key in sorted(ledger['completed']):
            for record in ledger['completed'][key]['slices']:
                if record['status'] == 'encoded':
                    f.write(json.dumps(record, ensure_ascii=True) + '\n')
    slice_temp.replace(output / 'slices.jsonl')
    ledger['views'] = {name: sha(output / name) for name in ['prepared_scenes.jsonl', 'slices.jsonl']}
    ledger['counts'] = {'manifest_records': len(outcomes), 'total_units': len(units),
                        'completed_units': len(ledger['completed']),
                        'scene_preparation_status': dict(Counter(r['preparation_status'] for r in outcomes)),
                        'accepted_slices': sum(r.get('accepted_slice_count', 0) for r in ledger['completed'].values())}


def self_test():
    # Pure hand-authored tests; no Java, model, or external source is opened.
    original, _ = legacy.load_original_functions(HERE / 'vendor/legacy_joern/data/sensiAPI.txt')
    assert proven_original_nontermination('void f(){\n    if(1){\n        x=1;\n    }\n}\n', original)
    assert proven_original_nontermination('void f(){\n\tif(1){\n\t\tx=1;\n\t}\n}\n', original) is None
    assert proven_original_nontermination('int x=1;\n', original) is None
    space_lines = 'void f(){\n    if(1){\n        x=1;\n    }\n}\n'.splitlines(keepends=True)
    timed_out = False
    try:
        with legacy.preprocessing_deadline(PREPROCESS_TIMEOUT_SECONDS):
            original['get_condition'](space_lines, original['get_indentation'](space_lines))
    except legacy.OriginalPreprocessingTimeout:
        timed_out = True
    assert timed_out, 'The shortcut must agree with unchanged original helper nontermination'
    scene = {'scene_id': 'fixture', 'label': 1, 'status': 'candidate_pending_frontend',
             'source_line_spans': [[2, 4]], 'sanitized_path': 'u.c'}
    units = {'u': {'relative': 'u.c'}}
    partial = {'u': {'status': 'processed', 'inventory_complete': True, 'slices': [
        {'key_source_line': 3, 'status': 'encoded', 'owner_scene_ids': ['fixture']},
        {'key_source_line': 4, 'status': 'rejected_unsupported_literal', 'owner_scene_ids': ['fixture']}]}}
    assert not scene_outcomes([scene], units, partial)[0]['frontend_processable']
    assert not scene_outcomes([scene], units, {'u': {'status': 'processed', 'inventory_complete': True, 'slices': []}})[0]['frontend_processable']
    assert not scene_outcomes([dict(scene, status='context_unresolved')], units, partial)[0]['frontend_processable']
    global_scene = {'context_reasons': ['referenced_file_scope_state_context_unresolved'], 'context_facts': {
        'file_scope_references': [{'reference_lines': [3], 'declaration_line_span': [1, 1], 'is_extern_or_unknown': False}]}}
    public = {'line_num': [1, 3], 'key_line': 1}
    graph = {'nodes': [{'line': 0, 'target': 0}, {'line': 1, 'target': 1}]}
    assert not slice_context_rejections(global_scene, public, graph)
    assert slice_context_rejections(global_scene, {'line_num': [3], 'key_line': 0}, {'nodes': [{'line': 0, 'target': 1}]})
    external_call = {'context_reasons': ['external_helper_context_review_required'], 'context_facts': {'external_helper_calls': [{'line': 9}]}}
    assert not slice_context_rejections(external_call, public, graph)
    record = {'status': 'PASS', 'script_sha256': sha(__file__), 'scope': 'synthetic failure-proof, context facts and scene aggregation only',
              'unchanged_original_helper_timeout_confirmed': timed_out, 'timeout_seconds': PREPROCESS_TIMEOUT_SECONDS,
              'new_condition_values_generated': False, 'model_forward': False, 'external_source_read': False}
    destination = HERE / 'prepare_external_self_test.json'
    if destination.exists():
        previous = json.loads(destination.read_text()); expected = previous.pop('result_sha256', None)
        if digest(previous) == expected and previous == record:
            print(json.dumps({'status': 'SKIP_VERIFIED_SELF_TEST', 'receipt': str(destination), 'scope': record['scope']}))
            return
        print(json.dumps({'status': 'PASS_CURRENT_TESTS_EXISTING_RECEIPT_PRESERVED',
                          'receipt': str(destination), 'scope': record['scope'],
                          'receipt_matches_current_implementation': False}))
        return
    else:
        save(destination, record)
    print(json.dumps({'status': 'PASS', 'receipt': str(destination), 'scope': record['scope']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--dataset', choices=['juliet', 'juliet_1_3_1', 'mit'])
    parser.add_argument('--manifest-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--unit-limit', type=int, default=100, help='Maximum pending units this invocation; 0 means all')
    parser.add_argument('--java', type=Path, default=Path((__import__('shutil').which('java') or 'java')))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dataset or not args.manifest_dir or not args.output or args.unit_limit < 0:
        parser.error('--dataset, --manifest-dir, --output and nonnegative --unit-limit are required')
    profile = verify_profile()
    manifest_dir, output = args.manifest_dir.resolve(), args.output.resolve()
    if not manifest_dir.is_relative_to(HERE) or not output.is_relative_to(HERE) or output == HERE or output.is_relative_to(manifest_dir):
        raise ValueError('Manifest and dedicated output must be separate external_eval directories')
    receipt_path, manifest_path = manifest_dir / 'manifest_receipt.json', manifest_dir / 'scenes.jsonl'
    receipt = json.loads(receipt_path.read_text())
    receipt_hash = receipt.pop('receipt_sha256', None)
    if digest(receipt) != receipt_hash or receipt.get('status') != 'PASS':
        raise ValueError('Static manifest receipt is not hash-verified PASS')
    config = HERE / 'dataset_configs.json'
    if receipt.get('dataset') != args.dataset or receipt.get('completion') != 'static_manifest_only_not_model_acceptance':
        raise ValueError('Static manifest is not complete for this dataset')
    if receipt['request']['config_sha256'] != sha(config) or receipt['request']['implementation_sha256'] != sha(HERE / 'make_manifest.py'):
        raise ValueError('Manifest config or implementation changed')
    source_config_rel = receipt['request'].get('source_config_path')
    if source_config_rel:
        source_config_path = (HERE / source_config_rel).resolve()
        if not source_config_path.is_relative_to(HERE) or sha(source_config_path) != receipt['request'].get('source_config_sha256'):
            raise ValueError('Source dataset config changed or escapes external_eval')
    elif args.dataset == 'juliet_1_3_1':
        raise ValueError('Juliet 1.3.1 manifest lacks its pinned source config')
    if receipt['scenes_sha256'] != sha(manifest_path):
        raise ValueError('Static scene manifest changed')
    scenes = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len({s['scene_id'] for s in scenes}) != len(scenes):
        raise ValueError('Duplicate scene identities')
    units = {}
    for scene in scenes:
        if not eligible_scene(scene):
            continue
        relative = scene.get('sanitized_path')
        if not relative:
            raise ValueError('Eligible scene lacks sanitized source')
        path = legacy.inside(manifest_dir, relative)
        expected = scene['sanitized_sha256']
        if sha(path) != expected:
            raise ValueError('Sanitized unit hash mismatch')
        key = hashlib.sha256(relative.encode()).hexdigest()
        unit = units.setdefault(key, {'relative': relative, 'path': str(path), 'sha256': expected, 'scenes': []})
        if unit['sha256'] != expected:
            raise ValueError('Conflicting sanitized unit identity')
        unit['scenes'].append(scene)
    api, jars, asset_receipt = legacy.verify_assets(HERE / 'vendor/legacy_joern')
    original, source_receipts = legacy.load_original_functions(api)
    request = {'script_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path), 'manifest_receipt_sha256': sha(receipt_path),
               'profile_sha256': sha(HERE / 'finite_frontend_profile.json'), 'config_sha256': sha(config),
               'dataset': args.dataset, 'source_receipts': source_receipts, 'asset_receipt': asset_receipt,
               'units': {k: {'relative': u['relative'], 'sha256': u['sha256']} for k, u in units.items()},
               'unit_order': 'ascending sanitized content hash then opaque unit id', 'batch_size': BATCH_SIZE,
               'parser_timeout_seconds': PARSER_TIMEOUT_SECONDS, 'preprocessing_timeout_seconds': PREPROCESS_TIMEOUT_SECONDS,
               'java_sha256': sha(args.java), 'python': sys.version, 'numpy': np.__version__}
    if source_config_rel:
        request['source_config_path'] = source_config_rel
        request['source_config_sha256'] = receipt['request']['source_config_sha256']
        request['adapter_sha256'] = receipt['request'].get('adapter_sha256')
    output.mkdir(exist_ok=True)
    ledger_path = output / 'preparation_receipt.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        expected = ledger.pop('result_sha256', None)
        if digest(ledger) != expected or ledger['request_sha256'] != digest(request):
            raise ValueError('Existing preparation identity/hash differs; refusing overwrite')
        for record in ledger['completed'].values():
            for filename, expected in record.get('artifact_sha256', {}).items():
                if sha(filename) != expected:
                    raise ValueError('Completed cache changed')
        if ledger['status'] == 'COMPLETE':
            for name, expected in ledger['views'].items():
                if sha(output / name) != expected:
                    raise ValueError('Prepared view changed')
            print(json.dumps({'status': 'SKIP_VERIFIED_COMPLETE', 'output': str(output), 'counts': ledger['counts']}))
            return
    else:
        if any(output.iterdir()):
            raise ValueError('Untracked output is not empty')
        ledger = {'status': 'RUNNING', 'request': request, 'request_sha256': digest(request), 'completed': {}, 'batches': [],
                  'profile_scope': profile['status'], 'legacy_fixture_overall_status': 'FAIL',
                  'labels_used_for_candidate_selection': False, 'model_forward': False,
                  'source_text_printed': False, 'original_source_files_modified': False,
                  'primary_processable_finalized': False, 'unknown_means_safe': False}
        save(ledger_path, ledger)
    pending = sorted((k for k in units if k not in ledger['completed']), key=lambda k: (units[k]['sha256'], k))
    if args.unit_limit:
        pending = pending[:args.unit_limit]
    # Proven failures are rejected before Java. Uncertain cases keep the exact
    # original helper and fixed deadline; no substitute conditions are emitted.
    ready = []
    for key in pending:
        with Path(units[key]['path']).open('r', encoding='utf-8', errors='ignore') as f:
            shortcut = proven_original_nontermination(f.read(), original)
        if shortcut:
            ledger['completed'][key] = {'status': 'rejected_original_nontermination', 'inventory_complete': False,
                                         'slices': [], 'artifact_sha256': {}, **shortcut}
        else:
            ready.append(key)
    save(ledger_path, ledger)
    for start in range(0, len(ready), BATCH_SIZE):
        keys = ready[start:start + BATCH_SIZE]
        batch_dir = output / f'batch_{len(ledger["batches"]):06d}'
        batch = {'unit_ids': keys, 'directory': str(batch_dir), 'status': 'RUNNING'}
        ledger['batches'].append(batch)
        save(ledger_path, ledger)
        parsed = parser_batch(args.java, jars, [Path(units[k]['path']) for k in keys], batch_dir)
        batch.update(parser=parsed, status='CSV_READY' if parsed['status'] == 'parsed' else 'FAILED')
        save(ledger_path, ledger)
        for key in keys:
            unit = units[key]
            if parsed['status'] != 'parsed':
                result = {'status': parsed['status'], 'inventory_complete': False, 'slices': []}
            else:
                try:
                    table, _ = legacy.csv_tables(Path(parsed['csv_dir']), Path(unit['path']).name)
                    result = prepare_unit(Path(unit['path']), table, unit['scenes'], original, api, output / 'cache', key)
                    result.setdefault('artifact_sha256', {}).update({str(table / name): sha(table / name) for name in ['nodes.csv', 'edges.csv']})
                except (ValueError, OSError):
                    result = {'status': 'rejected_parser_schema_or_io', 'inventory_complete': False, 'slices': []}
            result.setdefault('artifact_sha256', {})
            ledger['completed'][key] = result
            save(ledger_path, ledger)
        batch['status'] = 'COMPLETE'
        save(ledger_path, ledger)
        print(json.dumps({'completed_units': len(ledger['completed']), 'total_units': len(units)}), flush=True)
    ledger['status'] = 'COMPLETE' if len(ledger['completed']) == len(units) else 'PARTIAL_EXECUTION_BUDGET'
    write_views(output, scenes, units, ledger)
    save(ledger_path, ledger)
    print(json.dumps({'status': ledger['status'], 'output': str(output), 'counts': ledger['counts']}))


if __name__ == '__main__':
    main()
