"""Resume only target-bearing original condition failures, with immutable origins.

The accepted condition_repair worker is imported unchanged. This coordinator
does not select slices, modify C, infer labels, or invoke any model.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import condition_repair as repair
import legacy_frontend as legacy
import prepare_external as prepare

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / 'condition_processor_contract.json'
PROBE = HERE / 'condition_repair_probe/result.json'
REGISTRY = HERE / 'condition_repair_development_registry.json'
TEST = HERE / 'condition_overlay_metadata_self_test.json'
read = repair.verified
sha, digest, save = legacy.sha, legacy.digest, legacy.save
POLICY = {'base_status': 'rejected_original_nontermination',
          'selection': 'unit contains at least one record_kind scene with task_applicability target',
          'mixed_unit': 'all original candidates; diagnostic scenes retained; no label-based slice selection',
          'pure_diagnostic_units': 'retain original records; explicitly not repaired',
          'unknown_condition': 'N is rejected at original slice access; never converted to U/T/F',
          'context_numeric_layout_guards': 'unchanged', 'max_workers': 8}


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def verify_reference(item):
    if sha(item['path']) != item['sha256']:
        raise ValueError('Bound evidence changed: ' + str(item['path']))


def artifacts(record):
    for path, expected in record.get('artifact_sha256', {}).items():
        if sha(path) != expected:
            raise ValueError('Completed artifact changed: ' + path)


def choose_units(base, scenes):
    by_source = defaultdict(list)
    for scene in scenes:
        by_source[scene.get('sanitized_path')].append(scene)
    selected, retained = [], []
    for key, record in base['completed'].items():
        if record['status'] != POLICY['base_status']:
            continue
        unit = base['request']['units'][key]
        target = any(s.get('record_kind') == 'scene' and s.get('task_applicability') == 'target'
                     for s in by_source[unit['relative']])
        (selected if target else retained).append(key)
    selected.sort(key=lambda k: (base['request']['units'][k]['sha256'], k))
    return selected, sorted(retained), by_source


def checked_resume(path, request):
    value = read(path)
    if value['request'] != request or value['request_sha256'] != digest(request):
        raise ValueError('Existing execution identity differs; refusing reuse or overwrite')
    return value


def classify_failure(returncode, timed_out, error):
    if timed_out:
        return 'rejected_unit_wall_timeout'
    if returncode == -signal.SIGXCPU:
        return 'rejected_worker_cpu_limit'
    reason = error.get('reason', '')
    if error.get('error_type') == 'MemoryError' or reason.startswith('JVM_RESOURCE_'):
        return 'rejected_address_space_or_memory_resource'
    if error.get('error_type') == 'TimeoutExpired':
        return 'rejected_parser_wall_timeout'
    if reason.startswith('SOURCE_RESOURCE_'):
        return 'rejected_source_resource_cap'
    if reason.startswith('PARSER_FAILED'):
        return 'rejected_parser_returncode'
    if reason.startswith('ORIGINAL_PREPARATION_DID_NOT_COMPLETE:'):
        return reason.split(':', 1)[1]
    if returncode and returncode < 0:
        return 'rejected_worker_signal_' + str(-returncode)
    return 'rejected_worker_exception_requires_review'


def bounded_process(command, log_path, seconds, environment):
    """Own a fresh process group; reap descendants on every exit, including crash."""
    before = time.monotonic(); timed_out = False
    with Path(log_path).open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            try:
                code = process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        finally:
            # Any surviving Java process belongs to this worker's private group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return code, timed_out, time.monotonic() - before


def production_worker(task_path):
    task = read(task_path); root = Path(task['output'])
    try:
        repair.worker(task_path)
    except Exception as error:
        save(root / 'worker_exception.json', {'status': 'WORKER_EXCEPTION',
             'error_type': type(error).__name__, 'reason': str(error),
             'model_forward': False, 'C_executed': False})
        raise
    finally:
        # No guessed zeros: these stages were not separately instrumented in
        # the accepted worker. Parent wall includes worker startup/verification.
        save(root / 'resource_instrumentation.json', {
            'status': 'OBSERVED', 'source_read_seconds': None, 'encoder_seconds': None,
            'serialization_seconds': None,
            'unmeasured_reason': 'accepted worker records aggregate preparation; original code not instrumented',
            'child_peak_RSS_MiB': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
            'child_peak_RSS_scope': 'Linux RUSAGE_CHILDREN maximum for reaped direct children; not simultaneous aggregate'})


def freeze_contract():
    test = read(TEST)
    if test['status'] != 'PASS' or test['coordinator_sha256'] != sha(__file__):
        raise ValueError('Matching metadata-only coordinator tests required')
    plan, dependencies = repair.inputs(HERE / 'condition_repair_representatives.json')
    probe = read(PROBE)
    if probe['status'] != 'COMPLETE' or len(probe['completed']) != 6:
        raise ValueError('Six successful mechanism executions required')
    for key, expected in dependencies.items():
        if probe['request'].get(key) != expected:
            raise ValueError('Accepted worker dependency changed: ' + key)
    exact = 0
    for item in probe['completed'].values():
        if sha(item['receipt']) != item['receipt_sha256']:
            raise ValueError('Probe receipt changed')
        worker = read(item['receipt']); artifacts(worker['unit_result'])
        if worker['status'] != 'PASS_EXECUTION_WITH_EXPLICIT_SLICE_REJECTIONS':
            raise ValueError('Unsuccessful accepted probe')
        for result in worker['original_tensor_regressions']:
            if result['all_15_fields_exact'] is not True:
                raise ValueError('Prior encoded tensor parity failed')
            exact += 1
        if worker['context_policy_expanded'] or worker['source_layout_modified']:
            raise ValueError('Unexpected probe semantic expansion')
    if not exact:
        raise ValueError('No prior successful tensor regression evidence')
    dependency_files = ['condition_repair.py', 'partial_condition_probe.py', 'prepare_external.py',
                        'legacy_frontend.py', 'input_contract.py', 'dataset_configs.json',
                        'finite_frontend_profile.json', 'prepare_condition_overlay.py']
    value = {'status': 'PASS_FROZEN_PROCESSOR_CONTRACT',
        'checks': {name: True for name in ['original_12_fixture_tensor_parity',
            'six_real_mechanism_probes_complete', 'prior_encoded_tensor_regressions_exact',
            'required_N_rejected', 'unchanged_original_semantics', 'coordinator_metadata_self_test']},
        'evidence': {'partial_parity': reference(HERE / 'partial_condition_audit/verified_result.json'),
                     'representative_execution': reference(PROBE), 'metadata_self_test': reference(TEST)},
        'dependency_hashes': {str(HERE / name): sha(HERE / name) for name in dependency_files},
        'development_registry': reference(REGISTRY), 'policy': POLICY,
        'unit_caps': {k: v for k, v in repair.CAPS.items() if k not in ['workers', 'source_units_total']},
        'representative_prior_exact_slices': exact, 'model_forward': False, 'C_executed': False}
    if CONTRACT.exists():
        if read(CONTRACT) != value:
            raise ValueError('Frozen processor contract differs')
    else:
        save(CONTRACT, value)
    print(json.dumps({'status': value['status'], 'contract': str(CONTRACT)}))


def verify_contract():
    value = read(CONTRACT)
    if value['status'] != 'PASS_FROZEN_PROCESSOR_CONTRACT' or not all(value['checks'].values()):
        raise ValueError('Unaccepted processor contract')
    for path, expected in value['dependency_hashes'].items():
        if sha(path) != expected:
            raise ValueError('Frozen processor dependency changed')
    for item in value['evidence'].values():
        verify_reference(item)
    verify_reference(value['development_registry'])
    return value


def load_base(dataset):
    directory = HERE / ('prepared_' + dataset)
    receipt = read(directory / 'preparation_receipt.json')
    if receipt['status'] != 'COMPLETE':
        raise ValueError('Base preparation incomplete')
    for name, expected in receipt['views'].items():
        if sha(directory / name) != expected:
            raise ValueError('Base view changed')
    for record in receipt['completed'].values():
        artifacts(record)
    with (directory / 'prepared_scenes.jsonl').open() as stream:
        scenes = [json.loads(line) for line in stream]
    return directory, receipt, scenes


def candidate_task(key, dataset, base, scenes, output, request, java):
    unit = base['request']['units'][key]
    representative = {'dataset': dataset, 'unit_id': key, 'sanitized_path': unit['relative'],
                      'sanitized_sha256': unit['sha256'], 'old_tensor_artifacts': {},
                      'reusable_parser_artifacts': {}, 'purpose': 'fixed_target_condition_overlay'}
    return {'representative': representative,
            'source_path': str(HERE / 'manifests' / dataset / unit['relative']),
            'scenes': scenes, 'old_completed_unit': base['completed'][key],
            'output': str(output / 'units' / key), 'java': str(java), 'request': request}


def probe_match(task, probe):
    key = task['representative']['dataset'] + ':' + task['representative']['unit_id']
    item = probe['completed'].get(key)
    if item is None:
        return None
    if sha(item['receipt']) != item['receipt_sha256']:
        raise ValueError('Probe worker receipt changed')
    worker = read(item['receipt']); task_path = Path(item['receipt']).parent / 'task.json'
    if sha(task_path) != worker['task_sha256']:
        raise ValueError('Probe task changed')
    old = read(task_path)
    fields = ['source_path', 'scenes', 'old_completed_unit', 'java']
    if any(old.get(field) != task.get(field) for field in fields):
        raise ValueError('Same-unit probe has incompatible source/scene/config identity')
    for field in ['dataset', 'unit_id', 'sanitized_path', 'sanitized_sha256']:
        if old['representative'][field] != task['representative'][field]:
            raise ValueError('Probe sanitized source identity differs')
    for field in ['processor_sha256', 'original_prepare_driver_sha256', 'original_legacy_driver_sha256',
                  'original_finite_profile_sha256', 'config_sha256', 'caps', 'java_sha256']:
        if old['request'][field] != probe['request'][field]:
            raise ValueError('Probe policy identity differs')
    if sha(task['source_path']) != task['representative']['sanitized_sha256']:
        raise ValueError('Probe source changed')
    artifacts(worker['unit_result'])
    return {'status': 'REUSED_VERIFIED_PROBE', 'receipt': item['receipt'],
            'receipt_sha256': item['receipt_sha256'], 'unit_result': worker['unit_result'],
            'probe_task_sha256': sha(task_path)}


def execute_task(task, root):
    task_path = root / 'task.json'; result_path = root / 'unit_completion.json'
    if result_path.exists():
        result = read(result_path)
        if result['task_identity_sha256'] != digest(task):
            raise ValueError('Completed unit task changed')
        if result.get('receipt') and sha(result['receipt']) != result['receipt_sha256']:
            raise ValueError('Completed worker changed')
        artifacts(result['unit_result'])
        return result
    root.mkdir(parents=True, exist_ok=True)
    if task_path.exists():
        if read(task_path) != task:
            raise ValueError('Interrupted task identity changed')
        # Recover a successful write-before-controller-commit boundary exactly.
        if (root / 'worker_result.json').exists():
            worker = read(root / 'worker_result.json')
            if worker['task_sha256'] != sha(task_path):
                raise ValueError('Interrupted worker task mismatch')
            artifacts(worker['unit_result'])
            result = {'status': 'RECOVERED_SUCCESS_NO_REEXECUTION', 'receipt': str(root / 'worker_result.json'),
                      'receipt_sha256': sha(root / 'worker_result.json'), 'unit_result': worker['unit_result'],
                      'task_identity_sha256': digest(task)}
            save(result_path, result); return result
        # No completed observation means unknown; do not rerun already accessed
        # source implicitly or delete its partial parser/cache files.
        result = {'status': 'INTERRUPTED_ATTEMPT_RETAINED', 'task_identity_sha256': digest(task),
                  'unit_result': {'status': 'rejected_interrupted_worker_attempt',
                    'inventory_complete': False, 'slices': [], 'artifact_sha256': {}}}
        save(result_path, result); return result
    save(task_path, copy.deepcopy(task))
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    code, timeout, wall = bounded_process([sys.executable, str(Path(__file__)), '--worker-task', str(task_path)],
        root / 'worker.log', repair.CAPS['unit_wall_seconds'], environment)
    if code == 0 and (root / 'worker_result.json').exists():
        worker = read(root / 'worker_result.json')
        if worker['task_sha256'] != sha(task_path):
            raise ValueError('Worker task identity changed')
        artifacts(worker['unit_result'])
        result = {'status': 'EXECUTED', 'receipt': str(root / 'worker_result.json'),
                  'receipt_sha256': sha(root / 'worker_result.json'), 'unit_result': worker['unit_result']}
    else:
        error_path = root / 'worker_exception.json'
        error = read(error_path) if error_path.exists() else {}
        reason = classify_failure(code, timeout, error)
        result = {'status': 'FAILED_BOUNDED_ATTEMPT', 'returncode': code, 'wall_timeout': timeout,
                  'worker_exception': error, 'unit_result': {'status': reason, 'inventory_complete': False,
                      'slices': [], 'artifact_sha256': {}, 'failure_record': str(result_path)},
                  'loop_claimed': False}
    result.update(task_identity_sha256=digest(task), process_wall_seconds=wall,
                  log_sha256=sha(root / 'worker.log'))
    save(result_path, result)
    return result


def finish(output, base_dir, base, scenes, request, selected, retained, records):
    completed = copy.deepcopy(base['completed'])
    for key, record in records.items():
        completed[key] = copy.deepcopy(record['unit_result'])
    origins, fingerprints = {}, {}
    base_ref = reference(base_dir / 'preparation_receipt.json')
    for key, unit in completed.items():
        for row in unit['slices']:
            if row['status'] != 'encoded':
                continue
            path = str(Path(row['tensor_cache']).resolve())
            if path not in origins:
                if key not in records:
                    origin = {'kind': 'base', 'artifact_sha256': sha(path),
                              'preparation_receipt_path': base_ref['path'],
                              'preparation_receipt_sha256': base_ref['sha256']}
                elif records[key]['status'] == 'REUSED_VERIFIED_PROBE':
                    origin = {'kind': 'validated_probe', 'artifact_sha256': sha(path),
                              'worker_receipt_path': records[key]['receipt'],
                              'worker_receipt_sha256': records[key]['receipt_sha256'],
                              'probe_ledger_path': str(PROBE), 'probe_ledger_sha256': sha(PROBE)}
                else:
                    if not Path(path).is_relative_to(output):
                        raise ValueError('New tensor escapes overlay output')
                    origin = {'kind': 'overlay', 'artifact_sha256': sha(path)}
                origins[path] = origin
                with np.load(path, allow_pickle=False) as arrays:
                    if set(arrays.files) != set(repair.KEYS):
                        raise ValueError('Unexpected tensor field set')
                    for index in range(arrays['tokens'].shape[0]):
                        fingerprints[(path, index)] = repair.tensor_fingerprint(arrays, index)
    # Original completed records are left byte-value-identical. Only views gain
    # tensor provenance; scene inventory comes from the same original readout.
    outcomes = prepare.scene_outcomes(scenes, base['request']['units'], completed)
    slice_rows = [copy.deepcopy(row) for key in sorted(completed)
                  for row in completed[key]['slices'] if row['status'] == 'encoded']
    for row in slice_rows:
        path = str(Path(row['tensor_cache']).resolve())
        row['tensor_cache'] = path
        row['tensor_origin'] = dict(origins[path], tensor_row=row['tensor_row'])
        row['tensor_stored_sha256'] = fingerprints[(path, row['tensor_row'])]
    for name, rows in [('prepared_scenes.jsonl', outcomes), ('slices.jsonl', slice_rows)]:
        temporary = output / (name + '.tmp')
        with temporary.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True) + '\n')
        temporary.replace(output / name)
    receipt = {'status': 'COMPLETE', 'request': request, 'request_sha256': digest(request),
        'completed': completed, 'artifact_origins': origins,
        'views': {name: sha(output / name) for name in ['prepared_scenes.jsonl', 'slices.jsonl']},
        'counts': {'manifest_records': len(scenes), 'total_units': len(completed),
                   'selected_target_condition_units': len(selected), 'attempted_or_reused_units': len(records),
                   'retained_pure_diagnostic_condition_units': len(retained),
                   'ready_scenes': sum(bool(s['frontend_processable']) for s in outcomes),
                   'encoded_slices': len(slice_rows),
                   'scene_preparation_status': dict(Counter(s['preparation_status'] for s in outcomes)),
                   'overlay_unit_status': dict(Counter(r['unit_result']['status'] for r in records.values()))},
        'not_repaired_pure_diagnostic_unit_ids': retained,
        'unit_execution_receipts': {key: reference(output / 'units' / key / 'unit_completion.json') for key in records},
        'model_forward': False, 'C_executed': False, 'labels_used_for_candidate_selection': False}
    save(output / 'preparation_receipt.json', receipt)
    return receipt


def run(dataset, output, java, workers, unit_limit):
    verify_contract()
    base_dir, base, scenes = load_base(dataset)
    selected, retained, by_source = choose_units(base, scenes)
    probe = read(PROBE)
    if sha(java) != probe['request']['java_sha256']:
        raise ValueError('Java runtime changed')
    request = {'preparation_kind': 'condition_semantics_overlay', 'dataset': dataset,
        'base_preparation': reference(base_dir / 'preparation_receipt.json'),
        'development_registry': reference(REGISTRY), 'processor_contract': reference(CONTRACT),
        'processor_sha256': sha(__file__), 'config_sha256': base['request']['config_sha256'],
        'profile_sha256': base['request']['profile_sha256'],
        'source_manifest_sha256': base['request']['manifest_sha256'],
        'policy': POLICY, 'selected_unit_ids_sha256': digest(selected),
        'java_sha256': sha(java), 'unit_caps': read(CONTRACT)['unit_caps'],
        'probe_execution': reference(PROBE)}
    output = output.resolve()
    if not output.is_relative_to(HERE) or output in {HERE, base_dir} or output.is_relative_to(base_dir):
        raise ValueError('Use a separate output under external_eval')
    output.mkdir(exist_ok=True)
    with (output / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run_locked(output, base_dir, base, scenes, selected, retained, by_source,
                    probe, request, dataset, java, workers, unit_limit)


def _run_locked(output, base_dir, base, scenes, selected, retained, by_source,
                probe, request, dataset, java, workers, unit_limit):
    progress_path = output / 'execution_receipt.json'
    final_path = output / 'preparation_receipt.json'
    if final_path.exists():
        result = checked_resume(final_path, request)
        for unit in result['completed'].values():
            artifacts(unit)
        for name, expected in result['views'].items():
            if sha(output / name) != expected:
                raise ValueError('Completed overlay view changed')
        for item in result['unit_execution_receipts'].values():
            verify_reference(item)
        print(json.dumps({'status': 'SKIP_VERIFIED_COMPLETE', 'counts': result['counts']})); return
    if progress_path.exists():
        progress = checked_resume(progress_path, request)
    else:
        if set(p.name for p in output.iterdir()) - {'coordinator.lock'}:
            raise ValueError('Untracked output is not empty')
        progress = {'status': 'RUNNING', 'request': request, 'request_sha256': digest(request),
                    'selected_units': len(selected), 'retained_pure_diagnostic_units': len(retained)}
        save(progress_path, progress)
    records, pending = {}, []
    for key in selected:
        unit = base['request']['units'][key]
        task = candidate_task(key, dataset, base, by_source[unit['relative']], output, request, java)
        root = Path(task['output']); completion = root / 'unit_completion.json'
        if completion.exists():
            records[key] = execute_task(task, root)
            continue
        reused = probe_match(task, probe)
        if reused is not None:
            root.mkdir(parents=True, exist_ok=True)
            reused['task_identity_sha256'] = digest(task)
            save(completion, reused); records[key] = reused
        else:
            pending.append((key, task, root))
    if unit_limit:
        pending = pending[:unit_limit]
    print(json.dumps({'event': 'overlay_start', 'dataset': dataset, 'selected': len(selected),
                      'already_committed_or_probe_reused': len(records), 'pending_this_run': len(pending),
                      'retained_pure_diagnostic': len(retained), 'workers': workers}), flush=True)
    last = time.monotonic(); committed = len(records); fatal = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        todo = iter(pending); active = {}
        def launch():
            for _ in range(workers - len(active)):
                try:
                    key, task, root = next(todo)
                except StopIteration:
                    break
                active[pool.submit(execute_task, task, root)] = key
        launch()
        while active:
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                key = active.pop(future)
                try:
                    record = future.result(); records[key] = record
                    if record['unit_result']['status'] == 'rejected_worker_exception_requires_review':
                        fatal = 'Unexpected worker exception; no additional units scheduled'
                except Exception as error:
                    fatal = type(error).__name__ + ': ' + str(error)
            if len(records) - committed >= 64 or time.monotonic() - last >= 30 or not active:
                progress.update(status='STOPPED_FOR_REVIEW' if fatal else 'RUNNING',
                                committed_units=len(records), failure=fatal)
                save(progress_path, progress)
                print(json.dumps({'event': 'overlay_progress', 'dataset': dataset, 'committed': len(records),
                                  'selected': len(selected), 'active': len(active), 'failure': fatal}), flush=True)
                last = time.monotonic(); committed = len(records)
            if not fatal:
                launch()
    if fatal:
        raise RuntimeError(fatal)
    if len(records) == len(selected):
        result = finish(output, base_dir, base, scenes, request, selected, retained, records)
        progress.update(status='COMPLETE', preparation_receipt=reference(output / 'preparation_receipt.json'))
        save(progress_path, progress)
        print(json.dumps({'status': 'COMPLETE', 'dataset': dataset, 'counts': result['counts']}))
    else:
        progress.update(status='RUNNING', committed_units=len(records)); save(progress_path, progress)
        print(json.dumps({'status': 'BOUNDED_BATCH_COMPLETE', 'committed': len(records), 'selected': len(selected)}))


def self_test():
    """Hand metadata only; no original helper, external source, Java or model."""
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); path = root / 'receipt.json'; request = {'test': 1}
        save(path, {'status': 'COMPLETE', 'request': request, 'request_sha256': digest(request)})
        assert checked_resume(path, request)['status'] == 'COMPLETE'
        try:
            checked_resume(path, {'test': 2})
        except ValueError:
            pass
        else:
            raise AssertionError('Changed request accepted')
        value = json.loads(path.read_text()); value['status'] = 'changed'; path.write_text(json.dumps(value))
        try:
            read(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed content accepted')
        task = {'output': str(root / 'unit'), 'request': {}}
        unit = Path(task['output']); unit.mkdir(); save(unit / 'task.json', copy.deepcopy(task))
        interrupted = execute_task(task, unit)
        assert interrupted['status'] == 'INTERRUPTED_ATTEMPT_RETAINED'
        assert execute_task(task, unit)['unit_result'] == interrupted['unit_result']
        code, timeout, _ = bounded_process([sys.executable, '-c', 'import time; time.sleep(2)'],
                                           root / 'timeout.log', .05, dict(os.environ))
        assert timeout and classify_failure(code, timeout, {}) == 'rejected_unit_wall_timeout'
        assert classify_failure(-signal.SIGXCPU, False, {}) == 'rejected_worker_cpu_limit'
        assert classify_failure(1, False, {'error_type': 'MemoryError'}) == 'rejected_address_space_or_memory_resource'
        scene = {'scene_id': 's', 'status': 'candidate_pending_frontend', 'label': 1,
                 'sanitized_path': 'u.c', 'source_line_spans': [[1, 5]]}
        empty = {'status': 'processed', 'inventory_complete': True, 'slices': []}
        assert not prepare.scene_outcomes([scene], {'u': {'relative': 'u.c'}}, {'u': empty})[0]['frontend_processable']
    result = {'status': 'PASS', 'coordinator_sha256': sha(__file__),
              'checks': ['same_request_skip', 'changed_request_reject', 'selfhash_reject',
                         'interrupted_attempt_preserved_no_repeat', 'private_process_group_wall_timeout',
                         'CPU_AS_wall_distinct', 'empty_inventory_never_ready'],
              'external_source_read': False, 'Java_or_model_forward': False, 'original_helper_repeated': False}
    if TEST.exists():
        if read(TEST) != result:
            raise ValueError('Existing metadata test identity differs')
    else:
        save(TEST, result)
    print(json.dumps({'status': 'PASS', 'receipt': str(TEST)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--freeze-contract', action='store_true')
    parser.add_argument('--worker-task', type=Path)
    parser.add_argument('--dataset', choices=['mit', 'juliet'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--java', type=Path, default=Path((__import__('shutil').which('java') or 'java')))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--unit-limit', type=int, default=0)
    args = parser.parse_args()
    if sum(bool(x) for x in [args.self_test, args.freeze_contract, args.worker_task, args.dataset]) != 1:
        parser.error('Choose one action')
    if args.self_test:
        self_test()
    elif args.freeze_contract:
        freeze_contract()
    elif args.worker_task:
        production_worker(args.worker_task)
    else:
        if not 1 <= args.workers <= 8 or args.unit_limit < 0:
            parser.error('workers 1..8; unit-limit nonnegative')
        run(args.dataset, args.output or HERE / ('prepared_' + args.dataset + '_condition_overlay'),
            args.java.resolve(), args.workers, args.unit_limit)


if __name__ == '__main__':
    main()
