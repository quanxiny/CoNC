"""Read stable states of the unchanged original whole-file condition function.

Only completed outer-pass snapshots may be consumed. No N value is converted
to a model condition; every required source line must already contain U/T/F.
This CLI performs synthetic/previous-author-TRAIN compatibility tests only.
"""
import argparse
import ast
import hashlib
import json
import marshal
from pathlib import Path
import sys
import time

import legacy_frontend as legacy

HERE = Path(__file__).resolve().parent
MAX_SOURCE_BYTES = 1024 * 1024
MAX_LINES = 10000
MAX_INDENT = 256
MAX_TRACE_EVENTS = 2_000_000
WALL_SECONDS = 1.0


class StopAtProvedBound(Exception):
    pass


class TraceResourceLimit(Exception):
    pass


def proof(original):
    path = legacy.BUNDLE / legacy.UPSTREAM / 'dataset_generator.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_condition')
    parents = {child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
    writes, initializations, appends = [], [], []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'condition':
                    initializations.append(node)
                    if not isinstance(node.value, ast.List) or node.value.elts:
                        raise ValueError('Original condition initialization changed')
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == 'condition':
                    if not isinstance(node.value, ast.Constant) or node.value.value not in {'U', 'T', 'F'}:
                        raise ValueError('Original condition write is not a final U/T/F value')
                    ancestor, guarded = parents.get(node), False
                    while ancestor is not None:
                        if isinstance(ancestor, ast.If):
                            test = ancestor.test
                            if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                                    and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == 'N'
                                    and ast.dump(test.left, include_attributes=False) == ast.dump(ast.Subscript(
                                        value=target.value, slice=target.slice, ctx=ast.Load()), include_attributes=False)):
                                guarded = True
                        ancestor = parents.get(ancestor)
                    if not guarded:
                        raise ValueError('Condition assignment lacks its same-index N guard')
                    writes.append({'line': node.lineno, 'value': node.value.value})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'condition':
            if node.func.attr != 'append' or len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or node.args[0].value != 'N':
                raise ValueError('Unexpected original condition mutation')
            appends.append(node.lineno)
        if isinstance(node, (ast.AugAssign, ast.Delete)):
            targets = [node.target] if isinstance(node, ast.AugAssign) else node.targets
            if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'condition' for t in targets):
                raise ValueError('Unexpected original in-place condition mutation')
    boundaries = [n for n in ast.walk(function) if isinstance(n, ast.AugAssign)
                  and isinstance(n.target, ast.Name) and n.target.id == 'current_indentation']
    if len(initializations) != 1 or len(appends) != 1 or not writes or len(boundaries) != 1:
        raise ValueError('Original monotonic state proof structure differs')
    boundary = boundaries[0]
    if not isinstance(boundary.op, ast.Add) or not isinstance(boundary.value, ast.Constant) or boundary.value.value != 1:
        raise ValueError('Original outer-pass increment changed')
    return {'source_sha256': legacy.sha(path),
            'original_function_ast_sha256': hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
            'original_function_code_sha256': hashlib.sha256(marshal.dumps(original['get_condition'].__code__)).hexdigest(),
            'guarded_final_writes': writes, 'initial_N_append_line': appends[0],
            'complete_outer_pass_boundary_line': boundary.lineno,
            'proof': 'Every condition element changes only N to U/T/F; all assigned values are immutable for the remainder of this original call.'}


def observe(lines, original, evidence, *, wall_seconds=WALL_SECONDS, max_events=MAX_TRACE_EVENTS):
    started = time.monotonic()
    indents = original['get_indentation'](lines)
    if (sum(len(line.encode('utf-8')) for line in lines) > MAX_SOURCE_BYTES or len(lines) > MAX_LINES
            or max(indents, default=0) > MAX_INDENT):
        return {'status': 'RESOURCE_CAP_BEFORE_CALL', 'conditions': None, 'complete_passes': 0,
                'wall_seconds': time.monotonic() - started}
    function = original['get_condition']
    code_before = hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    if code_before != evidence['original_function_code_sha256']:
        raise ValueError('Original function bytecode changed')
    snapshot, complete_passes, events = None, 0, 0
    max_indent = max(indents, default=0)
    previous_trace = sys.gettrace()
    if previous_trace is not None:
        raise ValueError('Refusing to replace another active trace hook')

    def trace(frame, event, arg):
        nonlocal snapshot, complete_passes, events
        if frame.f_code is not function.__code__:
            return None
        events += 1
        if events > max_events:
            raise TraceResourceLimit()
        if event == 'line' and frame.f_lineno == evidence['complete_outer_pass_boundary_line']:
            state = list(frame.f_locals['condition'])
            if snapshot is not None and any(old != 'N' and old != new for old, new in zip(snapshot, state)):
                raise AssertionError('Original state violated the verified monotonic write invariant')
            snapshot = state
            complete_passes += 1
            # This event is after the entire inner pass, immediately before its
            # original increment. Once current indent > max+1, neither equality
            # with current indent nor current indent+1 can hold on any line.
            if frame.f_locals['current_indentation'] > max_indent + 1 and 'N' in state:
                raise StopAtProvedBound()
        return trace

    result, status, error_type = None, None, None
    try:
        with legacy.preprocessing_deadline(wall_seconds):
            sys.settrace(trace)
            result = function(lines, indents)
        status = 'NORMAL_RETURN'
    except StopAtProvedBound:
        result, status = snapshot, 'PROVED_STABLE_PARTIAL_STATE'
    except TraceResourceLimit:
        status = 'TRACE_RESOURCE_CAP'
    except legacy.OriginalPreprocessingTimeout:
        status = 'WALL_TIMEOUT_NOT_NONTERMINATION_PROOF'
    except Exception as error:
        status, error_type = 'ORIGINAL_OR_TRACE_EXCEPTION', type(error).__name__
    finally:
        sys.settrace(previous_trace)
    after = hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    if after != code_before:
        raise AssertionError('Original function bytecode mutated')
    # Timeout/resource/error results never expose partial values for consumption.
    return {'status': status, 'conditions': result, 'complete_passes': complete_passes,
            'trace_events': events, 'max_original_tab_indent': max_indent,
            'wall_seconds': time.monotonic() - started, 'error_type': error_type,
            'source_scope': 'complete_file', 'snapshot_boundary': 'complete_original_outer_pass',
            'original_function_bytecode_unchanged': True}


def required_values(observation, one_based_lines):
    if observation['status'] not in {'NORMAL_RETURN', 'PROVED_STABLE_PARTIAL_STATE'}:
        raise ValueError('Condition observation is not consumable')
    values = observation['conditions']
    result = []
    for line in one_based_lines:
        if line < 1 or line > len(values) or values[line - 1] not in {'U', 'T', 'F'}:
            raise ValueError('REQUIRED_SOURCE_LINE_CONDITION_UNRESOLVED')
        result.append(values[line - 1])
    return result


def structural_differences(actual, expected, path='$'):
    """Paths/types only; do not expose source text or waive semantic values."""
    if type(actual) is not type(expected):
        differences = [{'path': path, 'actual_type': type(actual).__name__, 'expected_type': type(expected).__name__}]
    else:
        differences = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                differences.append({'path': path + '.' + str(key), 'kind': 'missing_key'})
            else:
                differences.extend(structural_differences(actual[key], expected[key], path + '.' + str(key)))
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            differences.append({'path': path, 'kind': 'length_mismatch'})
        for index, (left, right) in enumerate(zip(actual, expected)):
            differences.extend(structural_differences(left, right, path + '[' + str(index) + ']'))
    elif actual != expected:
        differences.append({'path': path, 'kind': 'value_mismatch'})
    return differences


def self_test(output):
    api = HERE / 'vendor/legacy_joern/data/sensiAPI.txt'
    original, sources = legacy.load_original_functions(api)
    evidence = proof(original)
    historical_path = HERE / 'legacy_frontend_fixture_bounded/result.json'
    historical = json.loads(historical_path.read_text())
    historical_hash = historical.pop('result_sha256')
    if legacy.digest(historical) != historical_hash:
        raise ValueError('Historical fixture receipt changed')
    request = {'script_sha256': legacy.sha(__file__), 'original_sources': sources,
               'proof': evidence, 'historical_fixture_receipt_sha256': legacy.sha(historical_path),
               'caps': {'source_bytes': MAX_SOURCE_BYTES, 'lines': MAX_LINES, 'indent': MAX_INDENT,
                        'trace_events': MAX_TRACE_EVENTS, 'wall_seconds': WALL_SECONDS}}
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'verified_result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text()); expected = previous.pop('result_sha256', None)
        if legacy.digest(previous) != expected or previous['request'] != request or previous['status'] != 'PASS':
            raise ValueError('Existing probe evidence differs or is not successful')
        print(json.dumps({'status': 'SKIP_VERIFIED_PASS', 'output': str(output)})); return
    ledger = {'status': 'RUNNING', 'request': request, 'new_fixtures': [], 'historical_regressions': [],
              'external_real_source_access': False, 'Java_or_model_forward': False,
              'original_function_body_modified': False, 'original_receipts_modified': False}
    original_result = output / 'result.json'
    if original_result.exists():
        failed = json.loads(original_result.read_text()); prior_hash = failed.pop('result_sha256', None)
        if (legacy.digest(failed) != prior_hash or failed['status'] != 'FAIL'
                or failed.get('error_type') != 'AssertionError' or failed.get('error') != 'array_safe'
                or failed.get('historical_regressions')
                or len(failed.get('new_fixtures', [])) != 4
                or any(row['status'] != 'PASS' for row in failed['new_fixtures'])
                or failed.get('trace_resource_cap_fixture', {}).get('status') != 'TRACE_RESOURCE_CAP'
                or {k: v for k, v in failed['request'].items() if k != 'script_sha256'} !=
                   {k: v for k, v in request.items() if k != 'script_sha256'}):
            raise ValueError('Prior failure is not the explicitly reviewed JSON container-type harness failure')
        ledger['new_fixtures'] = failed['new_fixtures']
        ledger['trace_resource_cap_fixture'] = failed['trace_resource_cap_fixture']
        ledger['explicit_harness_repair'] = {'prior_failed_receipt_sha256': legacy.sha(original_result),
            'prior_request_sha256': legacy.digest(failed['request']),
            'prior_driver_sha256': failed['request']['script_sha256'],
            'repair_scope': 'compare all result fields through exact JSON serialization; tuple/list artifact representation only',
            'reused_successful_hand_fixtures': 4, 'original_failure_receipt_modified': False}
    elif any(output.iterdir()):
        raise ValueError('Untracked probe output retained; refusing overwrite')
    legacy.save(result_path, ledger)
    try:
        fixtures = [
            ('finite_plain', 'void f(){\n\tchar a[8];\n\ta[1]=0;\n}\n', [2, 3], 'NORMAL_RETURN', True),
            ('finite_tab_guard', 'void f(){\n\tchar a[8];\n\tif(1){\n\t\ta[1]=0;\n\t}\n}\n', [2, 4], 'NORMAL_RETURN', True),
            ('unrelated_macro_successor_unset', '#ifndef FLAG\n#define FLAG\nvoid f(){\n    char a[8];\n    a[1]=0;\n}\n#endif\n', [4, 5], 'PROVED_STABLE_PARTIAL_STATE', True),
            ('required_guard_successor_unset', 'void f(){\n    char a[8];\n    if(1){\n        a[1]=0;\n    }\n}\n', [2, 4], 'PROVED_STABLE_PARTIAL_STATE', False),
        ]
        for name, source, needed, expected_status, expected_accept in fixtures:
            if any(row['name'] == name for row in ledger['new_fixtures']):
                prior = next(row for row in ledger['new_fixtures'] if row['name'] == name)
                if (prior['source_sha256'] != hashlib.sha256(source.encode()).hexdigest()
                        or prior['required_source_lines'] != needed or prior['accepted'] != expected_accept
                        or prior['observation']['status'] != expected_status):
                    raise ValueError('Reviewed hand fixture definition changed')
                continue
            result = observe(source.splitlines(keepends=True), original, evidence)
            assert result['status'] == expected_status, (name, result['status'])
            try:
                selected = required_values(result, needed); accepted = True
            except ValueError:
                selected, accepted = None, False
            assert accepted == expected_accept, name
            ledger['new_fixtures'].append({'name': name, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                          'required_source_lines': needed, 'accepted': accepted,
                                          'selected_original_conditions': selected, 'observation': result, 'status': 'PASS'})
            legacy.save(result_path, ledger)
        if 'trace_resource_cap_fixture' not in ledger:
            limited = observe(['void f(){}\n'], original, evidence, max_events=1)
            assert limited['status'] == 'TRACE_RESOURCE_CAP' and limited['conditions'] is None
            ledger['trace_resource_cap_fixture'] = limited
        # No parser is re-run: original fixed source bytes/tables and tensor
        # hashes from every earlier PASS fixture are the regression oracle.
        for name, record in historical['completed'].items():
            if record['status'] != 'PASS':
                continue
            root = Path(record.get('artifact_root', historical_path.parent))
            for relative, expected in record['artifact_sha256'].items():
                if legacy.sha(root / relative) != expected:
                    raise ValueError('Historical fixture artifact changed')
            argv = record['parser']['argv']
            source_dir, csv_dir = Path(argv[-1]), Path(argv[argv.index('-outdir') + 1])
            candidates = [p for p in source_dir.iterdir() if p.suffix in {'.c', '.cpp'}
                          and legacy.sha(p) == record['result']['source_sha256']]
            if len(candidates) != 1:
                raise ValueError('Historical source fixture identity not unique')
            source = candidates[0]
            table, _ = legacy.csv_tables(csv_dir, source.name)
            observations = []
            injected = dict(original)
            def getter(lines, indents):
                observation = observe(lines, original, evidence)
                observations.append(observation)
                required_values(observation, range(1, len(lines) + 1))
                assert observation['status'] == 'NORMAL_RETURN'
                return observation['conditions']
            injected['get_condition'] = getter
            actual = legacy.public_slices(source, table, injected, api)
            canonical_actual = json.loads(json.dumps(actual))
            if canonical_actual != record['result']:
                ledger['semantic_regression_difference'] = {'fixture': name,
                    'differences': structural_differences(canonical_actual, record['result'])}
                raise AssertionError('Semantic or tensor regression: ' + name)
            representation_diff = structural_differences(actual, record['result'])
            if any(row.get('actual_type') != 'tuple' or row.get('expected_type') != 'list'
                   for row in representation_diff):
                raise AssertionError('Unreviewed artifact representation difference: ' + name)
            ledger['historical_regressions'].append({'name': name, 'status': 'PASS',
                'all_original_public_fields_and_15_tensor_hashes_exact': True,
                'full_result_canonical_JSON_exact': True,
                'runtime_tuple_vs_saved_JSON_list_paths': representation_diff,
                'result_sha256': legacy.digest(actual), 'condition_observations': observations})
            legacy.save(result_path, ledger)
        assert len(ledger['historical_regressions']) == 12
        ledger['status'] = 'PASS'
    except Exception as error:
        ledger.update(status='FAIL', error_type=type(error).__name__, error=str(error))
        legacy.save(result_path, ledger)
        raise
    legacy.save(result_path, ledger)
    print(json.dumps({'status': 'PASS', 'output': str(output), 'historical_exact_regressions': 12,
                      'new_condition_fixtures': len(ledger['new_fixtures'])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'partial_condition_audit')
    args = parser.parse_args()
    self_test(args.output.resolve())


if __name__ == '__main__':
    main()
