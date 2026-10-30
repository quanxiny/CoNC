"""Bounded representative driver for stable original condition states.

Original modules/defaults/cache/predictions remain untouched. This entrypoint
does not enable bulk processing; only the reviewed six-unit plan is accepted.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import legacy_frontend as legacy
import prepare_external as prepare
import partial_condition_probe as partial

HERE = Path(__file__).resolve().parent
CAPS = {'unit_wall_seconds': 40, 'worker_cpu_seconds': 20, 'worker_address_space_GiB': 3,
        'parser_wall_seconds': 20, 'parser_heap_MiB': 512, 'condition_wall_seconds': 1,
        'workers': 1, 'source_units_total': 6}
KEYS = ['tokens', 'numbers', 'marks', 'line_count', 'key', 'ops', 'parents', 'node_line',
        'condition', 'unit', 'known', 'bits', 'target', 'node_count', 'edges']


def verified(path):
    value = json.loads(Path(path).read_text()); expected = value.pop('result_sha256', None)
    if legacy.digest(value) != expected:
        raise ValueError('Receipt self hash mismatch: ' + str(path))
    return value


class OriginalConditionVector:
    """Expose original stable values; reject every access to an unassigned line."""
    def __init__(self, observation, rejected_lines):
        if observation['status'] not in {'NORMAL_RETURN', 'PROVED_STABLE_PARTIAL_STATE'}:
            raise ValueError('ORIGINAL_CONDITION_OBSERVATION_' + observation['status'])
        self.values = tuple(observation['conditions'])
        self.rejected_lines = rejected_lines

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0 or index >= len(self.values):
            raise ValueError('ORIGINAL_CONDITION_SOURCE_LINE_OUT_OF_RANGE')
        value = self.values[index]
        if value not in {'U', 'T', 'F'}:
            self.rejected_lines.add(index + 1)
            raise ValueError('ORIGINAL_CONDITION_UNASSIGNED_LINE')
        return value


def inputs(plan_path):
    plan = verified(plan_path)
    if len(plan['representatives']) != 6 or any(row['actual_new_source_access'] for row in plan['representatives']):
        raise ValueError('Expected the fixed metadata-only six-unit plan')
    parity_path = HERE / 'partial_condition_audit/verified_result.json'
    parity = verified(parity_path)
    if (parity['status'] != 'PASS' or len(parity['historical_regressions']) != 12
            or parity['request']['script_sha256'] != legacy.sha(HERE / 'partial_condition_probe.py')):
        raise ValueError('Matching successful original-state regression required')
    prepare.verify_profile()
    return plan, {'processor_sha256': legacy.sha(__file__), 'plan_sha256': legacy.sha(plan_path),
        'partial_condition_parity_receipt_sha256': legacy.sha(parity_path),
        'original_prepare_driver_sha256': legacy.sha(HERE / 'prepare_external.py'),
        'original_legacy_driver_sha256': legacy.sha(HERE / 'legacy_frontend.py'),
        'original_finite_profile_sha256': legacy.sha(HERE / 'finite_frontend_profile.json'),
        'config_sha256': legacy.sha(HERE / 'dataset_configs.json'), 'caps': CAPS}


def registry(plan_path):
    plan, request = inputs(plan_path)
    groups = {}
    for row in plan['representatives']:
        for group in row['template_groups']:
            key = row['dataset'] + ':' + group
            groups.setdefault(key, {'dataset': row['dataset'], 'template_group': group,
                'scope': 'entire MIT basic case across variants' if row['dataset'] == 'mit' else 'entire Juliet template across flows and fileparts',
                'exclude_from_new_unseen_primary_evaluation': True, 'source_units': []})['source_units'].append(
                    {'unit_id': row['unit_id'], 'sanitized_sha256': row['sanitized_sha256'], 'purpose': row['purpose']})
    record = {'status': 'REGISTERED_BEFORE_REAL_SOURCE_ACCESS', 'request': request,
        'historical_registry_path': str(HERE / 'development_diagnostics.json'),
        'historical_registry_sha256': legacy.sha(HERE / 'development_diagnostics.json'),
        'groups': list(groups.values()), 'historical_results_relabelled': False,
        'new_round_scope': 'mechanism engineering diagnosis only; no prediction-conditioned selection'}
    path = HERE / 'condition_repair_development_registry.json'
    if path.exists():
        if verified(path) != record:
            raise ValueError('Existing development registry differs')
    else:
        legacy.save(path, record)
    print(json.dumps({'status': record['status'], 'registry': str(path), 'groups': list(groups)}))


def tensor_fingerprint(arrays, row):
    import hashlib
    h = hashlib.sha256()
    for key in KEYS:
        value = np.asarray(arrays[key][row])
        specification = {'key': key, 'dtype': value.dtype.str, 'shape': list(value.shape)}
        h.update(json.dumps(specification, separators=(',', ':')).encode() + b'\0')
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def worker(task_path):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (CAPS['worker_cpu_seconds'], CAPS['worker_cpu_seconds'] + 1))
    resource.setrlimit(resource.RLIMIT_AS, (CAPS['worker_address_space_GiB'] * 2**30,) * 2)
    task = verified(task_path)
    root = Path(task['output']); source = Path(task['source_path'])
    if not root.is_relative_to(HERE) or legacy.sha(source) != task['representative']['sanitized_sha256']:
        raise ValueError('Unit source identity changed')
    started = time.monotonic()
    timings = defaultdict(float); observations = []; unresolved = set()
    api, jars, assets = legacy.verify_assets(HERE / 'vendor/legacy_joern')
    original, original_sources = legacy.load_original_functions(api)
    evidence = partial.proof(original)
    if source.stat().st_size > partial.MAX_SOURCE_BYTES:
        raise ValueError('SOURCE_RESOURCE_CAP_BEFORE_PARSER')
    with source.open(encoding='utf-8', errors='ignore') as stream:
        source_lines = stream.readlines()
    if (sum(len(x.encode()) for x in source_lines) > partial.MAX_SOURCE_BYTES or len(source_lines) > partial.MAX_LINES
            or max(original['get_indentation'](source_lines), default=0) > partial.MAX_INDENT):
        raise ValueError('SOURCE_RESOURCE_CAP_BEFORE_PARSER')
    old = task['old_completed_unit']
    for path, expected in task['representative']['old_tensor_artifacts'].items():
        if legacy.sha(path) != expected:
            raise ValueError('Previously successful tensor artifact changed')
    existing = task['representative']['reusable_parser_artifacts']
    if existing:
        for path, expected in existing.items():
            if legacy.sha(path) != expected:
                raise ValueError('Previously successful parser artifact changed')
        tables = {Path(path).parent for path in existing}
        if len(tables) != 1:
            raise ValueError('Ambiguous old parser table origin')
        table = tables.pop(); parser_receipt = {'status': 'REUSED_HASH_VERIFIED_CSV', 'artifacts': existing}
    else:
        source_dir = root / 'source'; source_dir.mkdir()
        shutil.copyfile(source, source_dir / source.name)
        csv_dir, log = root / 'csv', root / 'parser.log'
        command = [task['java'], '-Xmx512m', '-Xss512k', '-XX:ActiveProcessorCount=2',
                   '-XX:ReservedCodeCacheSize=64m', '-XX:CompressedClassSpaceSize=64m',
                   '-cp', os.pathsep.join(map(str, jars)), 'tools.parser.ParserMain',
                   '-outformat', 'csv', '-outdir', str(csv_dir), str(source_dir)]
        before = time.monotonic()
        with log.open('wb') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=CAPS['parser_wall_seconds'])
        timings['parser_seconds'] = time.monotonic() - before
        parser_receipt = {'status': 'PARSED' if process.returncode == 0 else 'PARSER_FAILED',
                          'returncode': process.returncode, 'log_sha256': legacy.sha(log), 'argv': command}
        if process.returncode:
            # Inspect a bounded log prefix locally for JVM allocation failures;
            # never print parser log/source contents or mislabel this as C syntax.
            with log.open('rb') as stream:
                prefix = stream.read(16384).lower()
            if any(marker in prefix for marker in [b'could not reserve enough space', b'cannot allocate memory',
                                                   b'native memory allocation', b'out of memory', b'os::commit_memory']):
                raise ValueError('JVM_RESOURCE_RESERVATION_FAILURE_NOT_C_PARSE_FAILURE')
            raise ValueError('PARSER_FAILED_SEE_RETAINED_LOG')
        table, _ = legacy.csv_tables(csv_dir, source.name)
    injected = dict(original)
    for name, stage in [('build_PDG', 'pdg_seconds'), ('build_XFG', 'xfg_seconds'),
                        ('clean_gadget', 'symbolizer_seconds'), ('tokenize_code_line', 'tokenizer_seconds'),
                        ('graph_build', 'graph_builder_seconds')]:
        function = original[name]
        def measured(*args, _function=function, _stage=stage, **kwargs):
            before = time.monotonic()
            try:
                return _function(*args, **kwargs)
            finally:
                timings[_stage] += time.monotonic() - before
        injected[name] = measured

    def condition_getter(lines, indents):
        if original['get_indentation'](lines) != indents:
            raise ValueError('Original indentation input changed')
        observation = partial.observe(lines, original, evidence)
        timings['condition_seconds'] += observation['wall_seconds']
        observations.append({key: value for key, value in observation.items() if key != 'conditions'} |
            {'assigned_lines': sum(value != 'N' for value in observation.get('conditions') or []),
             'unassigned_lines': sum(value == 'N' for value in observation.get('conditions') or [])})
        return OriginalConditionVector(observation, unresolved)
    injected['get_condition'] = condition_getter
    # Reuse the existing function body with one explicit driver-policy dependency
    # replaced: the old whole-file preflight is superseded by the verified getter.
    # Neither prepare's default namespace nor any original function is mutated.
    scope = dict(prepare.prepare_unit.__globals__)
    scope['proven_original_nontermination'] = lambda source, original: None
    operation = types.FunctionType(prepare.prepare_unit.__code__, scope, prepare.prepare_unit.__name__,
                                   prepare.prepare_unit.__defaults__, prepare.prepare_unit.__closure__)
    before = time.monotonic()
    result = operation(source, table, task['scenes'], injected, api, root / 'cache', task['representative']['unit_id'])
    timings['complete_original_preparation_seconds'] = time.monotonic() - before
    if not result.get('inventory_complete'):
        raise ValueError('ORIGINAL_PREPARATION_DID_NOT_COMPLETE:' + result['status'])
    old_slices = {row['slice_id']: row for row in old.get('slices', []) if row['status'] == 'encoded'}
    new_slices = {row['slice_id']: row for row in result['slices'] if row['status'] == 'encoded'}
    comparisons = []
    for slice_id, previous in old_slices.items():
        if slice_id not in new_slices:
            raise ValueError('Previously encoded slice was lost')
        new = new_slices[slice_id]
        with np.load(previous['tensor_cache'], allow_pickle=False) as a, np.load(new['tensor_cache'], allow_pickle=False) as b:
            old_arrays, new_arrays = {k: a[k] for k in KEYS}, {k: b[k] for k in KEYS}
            if (set(a.files) != set(KEYS) or set(b.files) != set(KEYS)
                    or any(old_arrays[k].dtype != new_arrays[k].dtype or not np.array_equal(
                        old_arrays[k][previous['tensor_row']], new_arrays[k][new['tensor_row']]) for k in KEYS)):
                raise ValueError('Previously successful original tensor changed')
            fingerprint = tensor_fingerprint(old_arrays, previous['tensor_row'])
        comparisons.append({'slice_id': slice_id, 'all_15_fields_exact': True, 'stored_tensor_sha256': fingerprint,
                            'old_cache_path': previous['tensor_cache'], 'old_cache_row': previous['tensor_row']})
    results = prepare.scene_outcomes(task['scenes'], {task['representative']['unit_id']: {
        'relative': task['representative']['sanitized_path']}}, {task['representative']['unit_id']: result})
    result.setdefault('artifact_sha256', {}).update({str(table / name): legacy.sha(table / name) for name in ['nodes.csv', 'edges.csv']})
    receipt = {'status': 'PASS_EXECUTION_WITH_EXPLICIT_SLICE_REJECTIONS', 'task_sha256': legacy.sha(task_path),
        'unit_result': result, 'scene_outcomes': results, 'original_tensor_regressions': comparisons,
        'condition_observations': observations, 'required_N_source_lines_rejected': sorted(unresolved),
        'stage_seconds': dict(timings), 'unit_wall_seconds': time.monotonic() - started,
        'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'parser': parser_receipt, 'original_sources': original_sources, 'vendor_assets': assets,
        'context_policy_expanded': False, 'source_layout_modified': False, 'original_driver_defaults_modified': False,
        'original_prepare_function_code_unchanged': operation.__code__ is prepare.prepare_unit.__code__,
        'model_forward': False, 'C_executed': False}
    legacy.save(root / 'worker_result.json', receipt)
    print(json.dumps({'status': receipt['status'], 'encoded_slices': len(new_slices),
                      'old_exact_slices': len(comparisons), 'needed_N_rejected_lines': len(unresolved)}))


def run_plan(plan_path, output, java):
    plan, request = inputs(plan_path)
    registration_path = HERE / 'condition_repair_development_registry.json'
    registration = verified(registration_path)
    if registration['request'] != request:
        raise ValueError('Matching pre-access development registry required')
    request.update(registry_sha256=legacy.sha(registration_path), java_sha256=legacy.sha(java))
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    path = output / 'result.json'
    if path.exists():
        ledger = verified(path)
        if ledger['request'] != request:
            raise ValueError('Existing representative execution identity differs')
        if ledger['status'] == 'COMPLETE':
            for item in ledger['completed'].values():
                if legacy.sha(item['receipt']) != item['receipt_sha256']:
                    raise ValueError('Completed worker receipt changed')
            print(json.dumps({'status': 'SKIP_VERIFIED_COMPLETE', 'output': str(output)})); return
    else:
        if any(output.iterdir()):
            raise ValueError('Untracked output directory')
        ledger = {'status': 'RUNNING', 'request': request, 'completed': {}, 'attempted_source_units': [],
                  'successful_old_predictions_read': False, 'model_forward': False}
        legacy.save(path, ledger)
    for index, representative in enumerate(plan['representatives']):
        key = representative['dataset'] + ':' + representative['unit_id']
        if key in ledger['completed']:
            continue
        root = output / representative['unit_id']; root.mkdir(exist_ok=False)
        dataset = representative['dataset']; old_dir = HERE / ('prepared_' + dataset)
        old_receipt_path = old_dir / 'preparation_receipt.json'; old_receipt = verified(old_receipt_path)
        if legacy.sha(old_receipt_path) != plan['request']['prepared_receipts'][dataset]:
            raise ValueError('Base preparation identity changed')
        if legacy.sha(old_dir / 'prepared_scenes.jsonl') != old_receipt['views']['prepared_scenes.jsonl']:
            raise ValueError('Base scene view changed')
        scenes = [json.loads(line) for line in (old_dir / 'prepared_scenes.jsonl').open()
                  if json.loads(line).get('sanitized_path') == representative['sanitized_path']]
        task = {'representative': representative, 'source_path': str(HERE / 'manifests' / dataset / representative['sanitized_path']),
                'scenes': scenes, 'old_completed_unit': old_receipt['completed'][representative['unit_id']],
                'output': str(root), 'java': str(java), 'request': request}
        task_path = root / 'task.json'; legacy.save(task_path, task)
        ledger['attempted_source_units'].append(key); legacy.save(path, ledger)
        print(json.dumps({'event': 'unit_start', 'unit': key, 'index': index + 1, 'total': 6}), flush=True)
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            with (root / 'worker.log').open('wb') as log:
                process = subprocess.run([sys.executable, str(Path(__file__)), '--worker-task', str(task_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=CAPS['unit_wall_seconds'], env=environment)
            if process.returncode or not (root / 'worker_result.json').exists():
                raise ValueError('WORKER_FAILED_EXIT_' + str(process.returncode))
        except Exception as error:
            failure = {'status': 'FAILED_BOUNDED_ATTEMPT', 'task_sha256': legacy.sha(task_path),
                       'error_type': type(error).__name__, 'reason': str(error),
                       'log_sha256': legacy.sha(root / 'worker.log'), 'source_retained': True}
            if (root / 'worker_exception.json').exists():
                failure['worker_exception'] = verified(root / 'worker_exception.json')
            legacy.save(root / 'worker_failure.json', failure)
            ledger.update(status='STOPPED_AT_FAILED_UNIT', failed_unit=key,
                          failure_receipt=str(root / 'worker_failure.json'))
            legacy.save(path, ledger)
            raise
        worker_receipt = verified(root / 'worker_result.json')
        ledger['completed'][key] = {'receipt': str(root / 'worker_result.json'),
            'receipt_sha256': legacy.sha(root / 'worker_result.json'), 'status': worker_receipt['status']}
        legacy.save(path, ledger)
        print(json.dumps({'event': 'unit_complete', 'unit': key,
                          'encoded_slices': worker_receipt['unit_result']['accepted_slice_count'],
                          'wall_seconds': worker_receipt['unit_wall_seconds']}), flush=True)
    ledger['status'] = 'COMPLETE'; legacy.save(path, ledger)


def self_test():
    rejected = set()
    vector = OriginalConditionVector({'status': 'PROVED_STABLE_PARTIAL_STATE', 'conditions': ['U', 'N', 'T', 'F']}, rejected)
    assert [vector[0], vector[2], vector[3]] == ['U', 'T', 'F']
    try:
        vector[1]
    except ValueError as error:
        assert str(error) == 'ORIGINAL_CONDITION_UNASSIGNED_LINE'
    else:
        raise AssertionError('N must not become a model condition')
    assert rejected == {2}
    for status in ['WALL_TIMEOUT_NOT_NONTERMINATION_PROOF', 'TRACE_RESOURCE_CAP', 'ORIGINAL_OR_TRACE_EXCEPTION']:
        try:
            OriginalConditionVector({'status': status, 'conditions': None}, set())
        except ValueError:
            continue
        raise AssertionError('Failed observation must not be consumed')
    result = {'status': 'PASS', 'processor_sha256': legacy.sha(__file__),
              'scope': 'handwritten condition vector only; no original helper or source accessed'}
    path = HERE / 'condition_repair_vector_self_test.json'
    if path.exists():
        if verified(path) != result:
            raise ValueError('Existing vector fixture differs')
    else:
        legacy.save(path, result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--register-plan', type=Path)
    parser.add_argument('--probe-plan', type=Path)
    parser.add_argument('--worker-task', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'condition_repair_probe')
    parser.add_argument('--java', type=Path, default=Path((__import__('shutil').which('java') or 'java')))
    args = parser.parse_args()
    if sum(bool(x) for x in [args.self_test, args.register_plan, args.probe_plan, args.worker_task]) != 1:
        parser.error('Choose exactly one action')
    if args.self_test:
        self_test()
    elif args.register_plan:
        registry(args.register_plan)
    elif args.probe_plan:
        run_plan(args.probe_plan, args.output, args.java)
    else:
        try:
            worker(args.worker_task)
        except Exception as error:
            task = verified(args.worker_task)
            legacy.save(Path(task['output']) / 'worker_exception.json', {
                'status': 'WORKER_EXCEPTION', 'error_type': type(error).__name__, 'reason': str(error),
                'resource_failure': isinstance(error, (MemoryError, subprocess.TimeoutExpired))
                    or str(error).startswith(('JVM_RESOURCE_', 'SOURCE_RESOURCE_')),
                'model_forward': False, 'C_executed': False})
            raise


if __name__ == '__main__':
    main()
