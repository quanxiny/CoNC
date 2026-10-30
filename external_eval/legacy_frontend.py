"""Audit original finite preprocessing with pinned legacy Joern on synthetic C.

No external dataset adapter, training, model forward, compiler, C execution,
archive main, shell wrapper, gold-selected slicing, or replacement parser.
Original function AST bodies are compiled without modification. The only
transcribed algorithms are graph2seq's public field mapping and prepare_cache's
tensor encoding; references are recorded below.
"""
import argparse
import ast
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
BUNDLE = WORKSPACE / 'transformer_gru_v76_v2'
UPSTREAM = 'third_party/VDTransformer/src/preprocess/'
GRAPH = 'external/overflowgen/joint_graph_study_v1/implementation_family_isolated_v2/graph_input.py'
WORDS = 'external/overflowgen/safe_numeric_artifacts/word_index.json'
PINNED_COMMIT = '28449e57262dd24c0938108687e1019f0fc6ddd5'
AST_NAMES = {
    'data_generator.py': ['extract_line_number', 'read_csv', 'extract_nodes_with_location_info', 'build_PDG', 'build_XFG'],
    'dataset_generator.py': ['get_indentation', 'get_condition'],
    'symbolizer.py': ['operators3', 'operators2', 'operators1', 'keywords', 'main_set', 'main_args', 'clean_gadget', 'tokenize_code_line'],
}


class OriginalPreprocessingTimeout(Exception):
    """An unchanged original helper failed to terminate within its audit budget."""


@contextmanager
def preprocessing_deadline(seconds=3):
    # Bound legacy helper execution from outside its unchanged AST body.
    def expired(signum, frame):
        raise OriginalPreprocessingTimeout('ORIGINAL_PREPROCESSING_TIMEOUT; source preserved without indentation repair')
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    value.pop('result_sha256', None)
    value['result_sha256'] = digest(value)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def inside(root, rel):
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes approved root: ' + str(rel))
    return path


def load_original_functions(api_path):
    """Select exact original AST nodes; no module imports or main blocks run."""
    namespace = {'nx': nx, 're': re, 'List': List, 'Set': Set, 'Tuple': Tuple, 'Dict': Dict,
                 'join': os.path.join, 'exists': os.path.exists}
    source_receipts = {}
    upstream_manifest = {item['path']: item['candidate_sha256'] for item in
                         read(BUNDLE / 'docs/VDTRANSFORMER_SOURCE_MANIFEST.json')['files']}
    for name, wanted in AST_NAMES.items():
        rel = UPSTREAM + name
        path = BUNDLE / rel
        if sha(path) != upstream_manifest[rel]:
            raise ValueError('Original upstream source hash mismatch: ' + rel)
        tree = ast.parse(path.read_text(), filename=str(path))
        selected = []
        seen = set()
        for node in tree.body:
            names = [node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ] if isinstance(node, ast.Assign) else []
            picked = set(names) & set(wanted)
            # keywords has a second assignment keywords.union(a); execute its
            # initial original frozenset here, then bind original API bytes below.
            if picked and not picked.issubset(seen):
                selected.append(node)
                seen.update(picked)
        if seen != set(wanted):
            raise ValueError('Missing original AST definition: ' + name)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
        if name == 'symbolizer.py':
            # Original symbolizer.py:42-45; injected path, identical split/union.
            with Path(api_path).open('r') as f:
                api_words = f.read().split(',')
            namespace['keywords'] = namespace['keywords'].union(api_words)
        source_receipts[rel] = {'sha256': sha(path), 'definitions': wanted,
                                'line_numbers': [getattr(n, 'lineno', None) for n in selected]}
    contract = read(BUNDLE / 'configs/MIGRATION_CONTRACT_v2.json')['sha256']
    for rel in [GRAPH, WORDS]:
        if sha(BUNDLE / rel) != contract[rel]:
            raise ValueError('Frozen input dependency hash mismatch: ' + rel)
        source_receipts[rel] = {'sha256': sha(BUNDLE / rel)}
    graph_namespace = {}
    # graph_input is the previously audited pure builder; it has no main/IO.
    exec(compile((BUNDLE / GRAPH).read_bytes(), str(BUNDLE / GRAPH), 'exec'), graph_namespace)
    namespace['graph_build'] = graph_namespace['build']
    words = read(BUNDLE / WORDS)
    namespace['vocabulary'] = {w: i + 3 for i, w in enumerate(words)} | {'<PAD>': 0, '<UNK>': 1, '<MASK>': 2}
    return namespace, source_receipts


def verify_assets(vendor):
    manifest_path = vendor.parent / 'source_assets.json'
    manifest = read(manifest_path)
    if manifest['commit'] != PINNED_COMMIT:
        raise ValueError('Unexpected upstream tool commit')
    paths = {}
    for item in manifest['assets']:
        path = inside(vendor, item['path'])
        if path.stat().st_size != item['size_bytes'] or sha(path) != item['sha256']:
            raise ValueError('Pinned vendor asset mismatch: ' + str(path))
        if item.get('git_blob_sha1'):
            h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != item['git_blob_sha1']:
                raise ValueError('Vendor git blob mismatch: ' + str(path))
        paths[item['path']] = path
    api = paths.get('data/sensiAPI.txt')
    jars = [p for name, p in sorted(paths.items()) if name.endswith('.jar')]
    if api is None or not jars:
        raise ValueError('Missing original API resource or Java classpath')
    return api, jars, {'manifest_sha256': sha(manifest_path), 'repository': manifest['repository'],
                       'commit': manifest['commit'], 'verified_asset_count': len(paths)}


def csv_tables(directory, source_name):
    candidates = [p for p in directory.rglob('nodes.csv')
                  if p.parent.name == source_name and (p.parent / 'edges.csv').is_file()]
    if len(candidates) != 1:
        raise ValueError(f'Expected one CSV pair for {source_name}, found {len(candidates)}')
    table = candidates[0].parent
    headers = {}
    for name, required in [('nodes.csv', {'key', 'type', 'location', 'code', 'operator'}),
                           ('edges.csv', {'type', 'start', 'end'})]:
        with (table / name).open() as f:
            headers[name] = next(csv.reader(f, delimiter='\t'))
        if not required.issubset(headers[name]):
            raise ValueError('Incompatible CSV header: ' + name)
    return table, headers


def line_edges(table):
    """Exact direction/typing convention of prepare_cache.py:38-46,69-72."""
    locations = {}
    with (table / 'nodes.csv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            head = row.get('location', '').split(':')[0]
            if head.isdigit():
                locations[row['key']] = int(head)
    edges = []
    with (table / 'edges.csv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            kind = row.get('type', '')
            a, b = locations.get(row.get('start')), locations.get(row.get('end'))
            if kind in ['REACHES', 'CONTROLS'] and a is not None and b is not None:
                edges.append((a, b, kind))
    return edges


def encode(public, edges, original):
    """prepare_cache.py:15-17,56-67,69-72; reject, never truncate or clamp."""
    lines, key = public['line_code'], public['key_line']
    if not lines or not 0 <= key < len(lines):
        raise ValueError('INVALID_SLICE_KEY')
    if len(lines) > 12 or any(len(line) > 32 for line in lines):
        raise ValueError('SLICE_EXCEEDS_12_LINES_OR_32_TOKENS')
    graph = original['graph_build'](lines, key)
    if graph['stats']['node_cap'] or len(graph['nodes']) > 64:
        raise ValueError('GRAPH_EXCEEDS_64_NODES')
    x = {k: np.zeros(shape, dtype=dtype) for k, (shape, dtype) in {
        'tokens': ((12, 32), 'uint16'), 'numbers': ((12, 32), 'int16'), 'marks': ((12, 32), 'uint8'),
        'line_count': ((), 'uint8'), 'key': ((), 'uint8'), 'ops': ((64,), 'uint8'),
        'parents': ((64, 2), 'int16'), 'node_line': ((64,), 'uint8'), 'condition': ((64,), 'uint8'),
        'unit': ((64,), 'uint8'), 'known': ((64,), 'uint8'), 'bits': ((64, 32), 'uint8'),
        'target': ((64,), 'uint8'), 'node_count': ((), 'uint8'), 'edges': ((12, 12), 'uint8')}.items()}
    x['line_count'][...] = len(lines)
    x['key'][...] = key
    x['marks'][...] = 1
    vocab = original['vocabulary']
    for li, line in enumerate(lines):
        for ti, tok in enumerate(line):
            numeric = tok.lstrip('+-').isdigit()
            x['tokens'][li, ti] = vocab['<NUM>'] if numeric else vocab.get(tok, 1)
            if numeric:
                value = int(tok)
                if not 0 <= value <= 999:
                    raise ValueError('NUMBER_OUTSIDE_0_999')
                x['numbers'][li, ti] = value
                x['marks'][li, ti] = 0
    for ni, node in enumerate(graph['nodes']):
        for key_out, key_in in [('ops', 'op'), ('node_line', 'line'), ('condition', 'condition'),
                                 ('unit', 'unit'), ('known', 'initial_known'), ('target', 'target')]:
            x[key_out][ni] = node[key_in]
        x['parents'][ni, :len(node['parents'])] = node['parents']
        if node['initial_known']:
            value = node['literal'] & 0xffffffff
            x['bits'][ni] = [(value >> bit) & 1 for bit in range(32)]
    x['node_count'][...] = len(graph['nodes'])
    lookup = {int(v): j for j, v in enumerate(public['line_num'])}
    for a, b, kind in edges:
        if a in lookup and b in lookup:
            x['edges'][lookup[a], lookup[b]] |= 1 if kind == 'REACHES' else 2
    return x, graph


def public_slices(source, table, original, api):
    """Original build_PDG/XFG and dataset_generator:195-214, graph2seq:26-48."""
    with source.open('r', encoding='utf-8', errors='ignore') as f:
        file_contents = f.readlines()
    pdg, candidates = original['build_PDG'](str(table), str(api), str(source), 'synthetic-audit')
    xfgs = original['build_XFG'](pdg, candidates, vul_lines=None)
    if xfgs is None:
        raise ValueError('NO_PDG_OR_CANDIDATES')
    indentation = original['get_indentation'](file_contents)
    conditions = original['get_condition'](file_contents, indentation)
    edges = line_edges(table)
    results = []
    # Candidate selection is completed above without any expected fixture line
    # or dataset label. All produced XFGs are transformed and retained.
    for xfg in xfgs['suspicious']:
        key_source_line = xfg.graph['key_line']
        entry = {'key_source_line': key_source_line, 'status': 'REJECTED'}
        try:
            node_lines = list(xfg)
            if any(n < 1 or n > len(file_contents) for n in node_lines):
                raise ValueError('NODE_LOCATION_OUTSIDE_SOURCE')
            code_lines = [file_contents[n - 1] for n in node_lines]
            if any(not line.endswith('\n') and '// Tag' not in line for line in code_lines):
                raise ValueError('ORIGINAL_SYMBOLIZER_WOULD_DROP_EOF_CHARACTER')
            symbolic = original['clean_gadget'](code_lines)
            remove = []
            for idx, n in enumerate(xfg):
                xfg.nodes[n]['code_sym_token'] = original['tokenize_code_line'](symbolic[idx], False)
                xfg.nodes[n]['indentation'] = indentation[n - 1]
                xfg.nodes[n]['condition'] = conditions[n - 1]
                if not xfg.nodes[n]['code_sym_token']:
                    remove.append(n)
            xfg.remove_nodes_from(remove)
            if not xfg.nodes or key_source_line not in xfg:
                raise ValueError('EMPTY_SLICE_OR_REMOVED_QUERY')
            line_num = sorted(xfg)
            line_code = [list(xfg.nodes[n]['code_sym_token']) for n in line_num]
            levels = [xfg.nodes[n]['indentation'] for n in line_num]
            minimum = min(levels)
            for code, n, level in zip(line_code, line_num, levels):
                for _ in range(level - minimum):
                    code.insert(0, '\t')
                code.insert(0, '<' + xfg.nodes[n]['condition'] + '>')
            public = {'line_num': line_num, 'key_line': line_num.index(key_source_line), 'line_code': line_code}
            entry['public'] = public
            x, graph = encode(public, edges, original)
            entry.update(status='ENCODED', graph=graph,
                         tensors={k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                                      'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in x.items()},
                         edge_matrix=x['edges'][:len(line_num), :len(line_num)].tolist())
        except (ValueError, IndexError, KeyError) as error:
            entry['reason'] = str(error)
        results.append(entry)
    return {'candidate_source_lines': sorted(candidates), 'slices': sorted(results, key=lambda x: x['key_source_line']),
            'line_edges': sorted(set(edges)), 'source_sha256': sha(source)}


def fixtures():
    def one(name, text, queries, expectation, **extra):
        return {'name': name, 'sources': {'fixture.c': text}, 'entry': 'fixture.c',
                'query_lines_for_assertions_only': queries, 'expectation': expectation, **extra}
    return [
        one('array_safe', 'void f() {\n\tchar a[8];\n\ta[7] = 0;\n}\n', [3], 'SATISFIED'),
        one('array_overflow', 'void f() {\n\tchar a[8];\n\ta[8] = 0;\n}\n', [3], 'VIOLATED'),
        one('sizeof_array_unknown', 'void f() {\n\tchar a[8];\n\tmemset(a, 0, sizeof(a));\n}\n', [3], 'UNKNOWN'),
        one('sizeof_pointer_unknown', 'void f() {\n\tchar *p = malloc(8);\n\tmemset(p, 0, sizeof(p));\n}\n', [3], 'UNKNOWN'),
        one('byte_element_unknown', 'void f() {\n\tint a[8];\n\tmemset(a, 0, 8);\n}\n', [3], 'UNKNOWN'),
        one('element_index_safe', 'void f() {\n\tint a[8];\n\ta[7] = 0;\n}\n', [3], 'SATISFIED'),
        one('sizeof_char_safe', 'void f() {\n\tchar a[8];\n\tmemset(a, 0, 8 * sizeof(char));\n}\n', [3], 'SATISFIED'),
        one('guard_tabs', 'void f() {\n\tchar a[8];\n\tif (1) {\n\t\ta[7] = 0;\n\t}\n}\n', [4], 'SATISFIED', marker='<T>'),
        one('guard_spaces', 'void f() {\n    char a[8];\n    if (1) {\n        a[7] = 0;\n    }\n}\n', [4], 'SATISFIED', marker='<U>'),
        one('eof_no_newline', 'void f() {\n\tchar a[8];\n\ta[7] = 0;}', [3], 'REJECT_EOF'),
        one('multiple_queries', 'void f() {\n\tchar a[8];\n\ta[7] = 0;\n\ta[8] = 0;\n}\n', [3, 4], 'MULTIPLE'),
        one('numeric_domain_rejection', 'void f() {\n\tchar a[8];\n\ta[1024] = 0;\n}\n', [3], 'REJECT_NUMBER'),
        {'name': 'cross_file_unknown', 'sources': {
            'caller.c': 'extern char shared[];\nvoid f() {\n\tshared[7] = 0;\n}\n',
            'definition.c': 'char shared[8];\n'}, 'entry': 'caller.c',
         'query_lines_for_assertions_only': [3], 'expectation': 'UNKNOWN'},
    ]


def check_fixture(fixture, result):
    checks = {}
    by_line = {entry['key_source_line']: entry for entry in result['slices']}
    for line in fixture['query_lines_for_assertions_only']:
        entry = by_line.get(line)
        checks[f'query_{line}_present'] = entry is not None
        if entry is None:
            continue
        wanted = fixture['expectation']
        if wanted.startswith('REJECT_'):
            reason = 'ORIGINAL_SYMBOLIZER_WOULD_DROP_EOF_CHARACTER' if wanted == 'REJECT_EOF' else 'NUMBER_OUTSIDE_0_999'
            checks[f'query_{line}_explicit_rejection'] = entry['status'] == 'REJECTED' and entry.get('reason') == reason
            continue
        checks[f'query_{line}_encoded'] = entry['status'] == 'ENCODED'
        if entry['status'] != 'ENCODED':
            continue
        graph, public = entry['graph'], entry['public']
        evidence = graph['rules']['evidence']
        if wanted == 'MULTIPLE':
            wanted = 'SATISFIED' if line == 3 else 'VIOLATED'
        checks[f'query_{line}_relation_{wanted}'] = bool(evidence) and all(e['relation'] == wanted for e in evidence)
        checks[f'query_{line}_target_binding'] = all(n['target'] == int(n['line'] == public['key_line']) for n in graph['nodes'])
        checks[f'query_{line}_unknown_sentinel'] = graph['nodes'][0]['op_name'] == 'OPAQUE' and graph['nodes'][0]['exact'] is None
        checks[f'query_{line}_parents_valid'] = all(0 <= p < len(graph['nodes']) for n in graph['nodes'] for p in n['parents'])
        if fixture.get('marker'):
            checks[f'query_{line}_original_condition_marker'] = public['line_code'][public['key_line']][0] == fixture['marker']
    return checks


def run_parser(java, jars, source_dir, output_dir):
    # Legacy ParserCSVOutput owns creation and refuses an existing directory.
    # Preserve the fresh-path requirement without pre-creating that directory.
    if output_dir.exists():
        raise FileExistsError('Parser output path already exists: ' + str(output_dir))
    command = [str(java), '-Xmx512m', '-cp', os.pathsep.join(str(p) for p in jars),
               'tools.parser.ParserMain', '-outformat', 'csv', '-outdir', str(output_dir), str(source_dir)]
    start = time.monotonic()
    with (output_dir.parent / 'parser.log').open('wb') as log:
        try:
            proc = subprocess.run(command, cwd=output_dir.parent, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=90, check=False)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Legacy parser timeout; partial output retained')
    receipt = {'argv': command, 'returncode': proc.returncode, 'wall_seconds': time.monotonic() - start,
               'log_sha256': sha(output_dir.parent / 'parser.log')}
    if proc.returncode:
        raise RuntimeError('Legacy parser failed; see retained parser.log, exit=' + str(proc.returncode))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', action='store_true', required=True)
    parser.add_argument('--java', type=Path, default=Path((__import__('shutil').which('java') or 'java')))
    parser.add_argument('--vendor', type=Path, default=HERE / 'vendor/legacy_joern')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'legacy_frontend_fixture_run')
    parser.add_argument('--golden', type=Path, default=HERE / 'train_frontend_golden.json')
    parser.add_argument('--resume-from', type=Path, help='Explicit repair: reuse only verified PASS jobs from an older audit; original inputs must match')
    args = parser.parse_args()
    vendor, output = args.vendor.resolve(), args.output_dir.resolve()
    if not output.is_relative_to(HERE) or output == HERE or output.is_relative_to(vendor):
        raise ValueError('Audit output must be a dedicated external_eval child outside vendor')
    api, jars, asset_receipt = verify_assets(vendor)
    original, source_receipts = load_original_functions(api)
    golden = read(args.golden)
    if golden['original_split'] != 'author_train' or len(golden['table_candidates']) != 1:
        raise ValueError('Golden must be explicitly selected original author TRAIN')
    golden_source = inside(WORKSPACE, golden['source_path'])
    old_nodes = inside(WORKSPACE, golden['table_candidates'][0]['nodes.csv'])
    old_edges = inside(WORKSPACE, golden['table_candidates'][0]['edges.csv'])
    if old_nodes.parent != old_edges.parent:
        raise ValueError('Golden table pair mismatch')
    request = {'script_sha256': sha(__file__), 'asset_receipt': asset_receipt,
               'original_sources': source_receipts, 'golden_descriptor_sha256': sha(args.golden),
               'golden_source_sha256': sha(golden_source), 'golden_nodes_sha256': sha(old_nodes),
               'golden_edges_sha256': sha(old_edges), 'fixtures': fixtures(), 'java_sha256': sha(args.java),
               'python': sys.version, 'numpy': np.__version__, 'networkx': nx.__version__}
    fingerprint = digest(request)
    output.mkdir(exist_ok=True)
    ledger_path = output / 'result.json'
    if ledger_path.exists():
        ledger = read(ledger_path)
        old_hash = ledger.pop('result_sha256', None)
        if ledger.get('request_sha256') != fingerprint or digest(ledger) != old_hash:
            raise ValueError('Existing audit identity/hash mismatch; retain it and select a new explicit output directory')
        for record in ledger['completed'].values():
            artifact_root = Path(record.get('artifact_root', output)).resolve()
            if not artifact_root.is_relative_to(HERE):
                raise ValueError('Reused artifact root escapes external_eval')
            for rel, expected in record.get('artifact_sha256', {}).items():
                if sha(inside(artifact_root, rel)) != expected:
                    raise ValueError('Completed audit artifact changed: ' + rel)
        if ledger['status'] != 'RUNNING':
            print(json.dumps({'status': 'SKIP_VERIFIED_' + ledger['status'], 'result': str(ledger_path)}))
            return
    else:
        if any(output.iterdir()):
            raise ValueError('Untracked output directory is not empty; refusing reuse')
        ledger = {'status': 'RUNNING', 'request_sha256': fingerprint, 'request': request,
                  'completed': {}, 'attempts': [], 'external_C_read': False,
                  'golden_scope': 'original author TRAIN, not asserted to be selected v76 TRAIN',
                  'source_labels_used_for_slicing': False, 'model_forward': False,
                  'C_compiled_or_executed': False, 'cross_file_semantics': 'UNSUPPORTED',
                  'public_mapping_reference': 'graph2seq.py:26-34,41-48; labels omitted',
                  'tensor_encoding_reference': 'prepare_cache.py:15-17,56-67,69-72'}
        if args.resume_from:
            previous_path = args.resume_from.resolve()
            if not previous_path.is_relative_to(HERE) or previous_path == ledger_path:
                raise ValueError('Resume source must be a distinct existing external_eval receipt')
            previous = read(previous_path)
            previous_self_hash = previous.pop('result_sha256', None)
            if digest(previous) != previous_self_hash:
                raise ValueError('Resume source receipt hash mismatch')
            old_request = {k: v for k, v in previous['request'].items() if k != 'script_sha256'}
            current_request = {k: v for k, v in request.items() if k != 'script_sha256'}
            if old_request != current_request:
                raise ValueError('Resume input identity differs beyond explicitly repaired driver script')
            receipt_hash = sha(previous_path)
            for name, record in previous['completed'].items():
                if record['status'] != 'PASS':
                    continue
                artifact_root = Path(record.get('artifact_root', previous_path.parent)).resolve()
                if not artifact_root.is_relative_to(HERE):
                    raise ValueError('Resume artifact root escapes external_eval')
                for rel, expected in record.get('artifact_sha256', {}).items():
                    if sha(inside(artifact_root, rel)) != expected:
                        raise ValueError('Resume artifact changed: ' + rel)
                record['artifact_root'] = str(artifact_root)
                record['reused_from_result'] = {'path': str(previous_path), 'sha256': receipt_hash}
                ledger['completed'][name] = record
            ledger['explicit_driver_repair_resume'] = {'source': str(previous_path), 'sha256': receipt_hash,
                                                       'old_script_sha256': previous['request']['script_sha256'],
                                                       'reused_PASS_jobs': sorted(ledger['completed'])}
        save(ledger_path, ledger)
    jobs = fixtures() + [{'name': 'author_train_golden', 'sources': {golden_source.name: golden_source.read_text()},
                         'entry': golden_source.name, 'golden': True}]
    for job in jobs:
        name = job['name']
        if name in ledger['completed']:
            continue
        attempt_number = sum(a['name'] == name for a in ledger['attempts']) + 1
        job_dir = output / f'{name}_attempt_{attempt_number}'
        source_dir = job_dir / 'source'
        source_dir.mkdir(parents=True, exist_ok=False)
        for filename, text in job['sources'].items():
            (source_dir / filename).write_text(text)
        attempt = {'name': name, 'directory': str(job_dir.relative_to(output)), 'status': 'RUNNING'}
        ledger['attempts'].append(attempt)
        save(ledger_path, ledger)
        record = {'name': name, 'status': 'FAIL'}
        stage = 'parser_startup'
        fatal_infrastructure_error = False
        try:
            record['parser'] = run_parser(args.java, jars, source_dir, job_dir / 'csv')
            stage = 'csv_contract'
            table, headers = csv_tables(job_dir / 'csv', job['entry'])
            record['csv_headers'] = headers
            stage = 'original_preprocessing'
            with preprocessing_deadline():
                record['result'] = public_slices(source_dir / job['entry'], table, original, api)
            if job.get('golden'):
                sample = golden['sample']
                expected = {k: sample[k] for k in ['line_num', 'key_line', 'line_code']}
                wanted_line = sample['line_num'][sample['key_line']]
                rebuilt = next((s for s in record['result']['slices'] if s['key_source_line'] == wanted_line), None)
                with preprocessing_deadline():
                    saved = public_slices(golden_source, old_nodes.parent, original, api)
                saved_query = next((s for s in saved['slices'] if s['key_source_line'] == wanted_line), None)
                old_headers = {}
                for path in [old_nodes, old_edges]:
                    with path.open() as stream:
                        old_headers[path.name] = next(csv.reader(stream, delimiter='\t'))
                checks = {'csv_headers_equal_saved': headers == old_headers,
                          'directional_line_edges_equal_saved': record['result']['line_edges'] == saved['line_edges'],
                          'saved_tables_reconstruct_author_public_tokens': saved_query is not None and saved_query.get('public') == expected,
                          'fresh_tables_reconstruct_author_public_tokens': rebuilt is not None and rebuilt.get('public') == expected,
                          'golden_encodable': rebuilt is not None and rebuilt['status'] == 'ENCODED'}
                record.update(expected_public=expected, saved_table_result=saved, checks=checks)
            else:
                record['checks'] = check_fixture(job, record['result'])
                record['expectation'] = job['expectation']
                record['frozen_feature_behavior'] = ('unknown is the original legal representation; never substituted with safe'
                                                    if job['expectation'] == 'UNKNOWN' else 'original finite feature behavior checked')
                record['semantic_precision'] = ('UNKNOWN_ORIGINAL_FEATURE' if job['expectation'] == 'UNKNOWN'
                                                else 'REJECTED_INPUT' if job['expectation'].startswith('REJECT_')
                                                else 'FINITE_TOKEN_RELATION_ONLY')
                record['source_program_proof'] = False
                record['source_context_status'] = ('MISSING_CROSS_FILE_CONTEXT' if name == 'cross_file_unknown'
                                                   else 'SELF_CONTAINED_SYNTHETIC_FIXTURE')
                record['scenario_input_eligibility'] = ('REJECT_MISSING_CONTEXT' if name == 'cross_file_unknown'
                                                        else 'REJECT_ORIGINAL_INPUT_CONTRACT' if job['expectation'].startswith('REJECT_')
                                                        else 'ELIGIBLE_IF_FROZEN_CONTRACT_CHECKS_PASS')
            record['status'] = 'PASS' if record['checks'] and all(record['checks'].values()) else 'FAIL'
        except Exception as error:
            record['error_type'] = type(error).__name__
            record['error'] = str(error)
            record['failure_stage'] = stage
            fatal_infrastructure_error = isinstance(error, (RuntimeError, OSError)) or stage in ['parser_startup', 'csv_contract']
            if isinstance(error, OriginalPreprocessingTimeout):
                record['scenario_input_eligibility'] = 'REJECT_ORIGINAL_PREPROCESSING_NONTERMINATION'
                record['expected_fixture_assertions_not_relaxed'] = True
                record['source_program_proof'] = False
        record['artifact_sha256'] = {str(p.relative_to(output)): sha(p) for p in job_dir.rglob('*') if p.is_file()}
        ledger['completed'][name] = record
        attempt['status'] = record['status']
        save(ledger_path, ledger)
        print(json.dumps({'fixture': name, 'status': record['status'], 'error': record.get('error')}), flush=True)
        if fatal_infrastructure_error:
            ledger['stopped_early'] = {'fixture': name, 'failure_stage': stage,
                                       'reason': 'Shared tool or I/O contract failure; remaining jobs not launched'}
            ledger['unattempted_jobs'] = [j['name'] for j in jobs if j['name'] not in ledger['completed']]
            save(ledger_path, ledger)
            break
    passed = all(r['status'] == 'PASS' for r in ledger['completed'].values())
    ledger['status'] = 'PASS' if passed else 'FAIL'
    ledger['external_adapter_authorized_by_result'] = False
    ledger['interpretation'] = 'Only the recorded finite cases and one author TRAIN golden were tested; cross-file resolution remains unsupported.'
    save(ledger_path, ledger)
    print(json.dumps({'status': ledger['status'], 'result': str(ledger_path)}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
