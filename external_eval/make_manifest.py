#!/usr/bin/env python3
"""Static scene sidecars; never a model parser. No external source is read by --self-test.

External runs require an explicit acceptance receipt; this tool does not create one.
Sources stay local and are never printed. Unknown context is retained, never safe.
"""
from __future__ import annotations
import argparse
import contextlib
import io
from collections import Counter, defaultdict
import hashlib
import fcntl
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import sqlite3
import xml.etree.ElementTree as ET
import zipfile

import tree_sitter
import tree_sitter_c
import tree_sitter_cpp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_EXT = {'.c', '.cpp', '.cc', '.cxx'}
LABEL_TOKEN = re.compile(r'good|bad|CWE\d+|basic_?\d+', re.I)
FAMILY = re.compile(r'^(.*)_(\d{2})(?:[a-z]|_(?:bad|good\w*))?$')
IDENT_NODES = {'identifier', 'field_identifier', 'type_identifier', 'namespace_identifier'}
# These are language/library primitives, not a dataset-tuned API shortlist.
STANDARD_CALLS = set('memcpy memmove memset memcmp strlen strcpy strncpy strcat strncat malloc calloc realloc free printf fprintf sprintf snprintf puts putchar getchar fgets scanf fscanf sscanf fread fwrite exit abort rand srand time wcslen wcscpy wcsncpy wcscat wcsncat wmemcpy wmemmove wmemset'.split())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def walk(node):
    """Preorder depth-first AST traversal without Python/C-stack recursion.

    Juliet files can contain deeply nested syntax that made the recursive
    generator above segfault inside the tree-sitter binding. Reversing each
    child list preserves its exact historical preorder.
    """
    if node is None:
        return
    pending = [node]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(reversed(current.children))


def text(node, source):
    return source[node.start_byte:node.end_byte].decode('utf-8', 'surrogateescape')


def blank(data):
    return bytes(b if b in (9, 10, 13) else 32 for b in data)


def parse(source, suffix):
    module = tree_sitter_cpp if suffix != '.c' else tree_sitter_c
    parser = tree_sitter.Parser(tree_sitter.Language(module.language()))
    return parser.parse(source)


def name_of(function, source):
    decl = function.child_by_field_name('declarator')
    while decl is not None:
        if decl.type == 'function_declarator':
            name = decl.child_by_field_name('declarator')
            ids = [n for n in walk(name) if n.type in {'identifier', 'field_identifier'}]
            return text(ids[-1], source) if ids else None
        decl = decl.child_by_field_name('declarator')
    return None


def calls_of(function, source):
    out = []
    body = function.child_by_field_name('body')
    for n in walk(body):
        if n.type == 'call_expression':
            callee = n.child_by_field_name('function')
            direct = callee.type == 'identifier'
            out.append({'name': text(callee, source), 'direct': direct,
                        'line': (source.count(b'\n', 0, n.start_byte) + 1)})
    return out


def function_info(source, suffix):
    tree = parse(source, suffix)
    functions = []
    for node in walk(tree.root_node):
        if node.type == 'function_definition':
            name = name_of(node, source)
            if name:
                declarator = node.child_by_field_name('declarator')
                kinds = {n.type for n in walk(declarator)} if declarator is not None else set()
                ancestor, class_member = node.parent, False
                while ancestor is not None:
                    class_member |= ancestor.type in {'class_specifier', 'struct_specifier', 'union_specifier'}
                    ancestor = ancestor.parent
                functions.append({'name': name, 'start_line': (source.count(b'\n', 0, node.start_byte) + 1),
                                  'end_line': (source.count(b'\n', 0, node.end_byte) + 1),
                                  'start_byte': node.start_byte, 'end_byte': node.end_byte,
                                  'calls': calls_of(node, source), '_node': node,
                                  'entry_structure': {'has_return_type': node.child_by_field_name('type') is not None,
                                                      'destructor_declarator': 'destructor_name' in kinds,
                                                      'qualified_declarator': 'qualified_identifier' in kinds,
                                                      'class_member_definition': class_member}})
    return tree, functions


def neutralize(source, tree, functions):
    """Token-based names only; literals, operators, numerical/type syntax untouched.

    Identifier replacement is consistent by spelling throughout a translation unit.
    This is deliberately not a scope resolver. Opaque macro payloads are rejected.
    """
    declared = {f['name'] for f in functions} - {'main'}
    identifiers = {text(n, source) for n in walk(tree.root_node)
                   if n.type in IDENT_NODES}
    selected = sorted(n for n in identifiers if n in declared or LABEL_TOKEN.search(n))
    # Collision-free and independent of label class; names never encode good/bad.
    names = {}
    for index, old in enumerate(selected):
        new = f'neutral_symbol_{index:05d}'
        while new in identifiers or new in names.values():
            new += '_'
        names[old] = new
    replacements, reasons = [], []
    for n in walk(tree.root_node):
        if n.type == 'comment':
            replacements.append((n.start_byte, n.end_byte, blank(source[n.start_byte:n.end_byte])))
        elif n.type in IDENT_NODES and text(n, source) in names:
            replacements.append((n.start_byte, n.end_byte, names[text(n, source)].encode()))
        elif n.type == 'preproc_arg':
            if any(re.search(r'\b' + re.escape(old) + r'\b', text(n, source)) for old in names):
                reasons.append('opaque_macro_payload_needs_identifier_resolution')
        elif n.type == 'preproc_include':
            path = n.child_by_field_name('path')
            if path is not None and path.type == 'string_literal':
                raw_path = text(path, source)
                # Never join or copy header source. Nonstandard include is unresolved.
                if LABEL_TOKEN.search(raw_path):
                    value = ('"header_' + sha(raw_path.encode())[:20] + '.h"').encode()
                    replacements.append((path.start_byte, path.end_byte, value))
                    reasons.append('renamed_local_header_context_unresolved')
    result = bytearray(source)
    for start, end, value in sorted(replacements, reverse=True):
        result[start:end] = value
    if result.count(b'\n') != source.count(b'\n'):
        raise AssertionError('sanitization changed line mapping')
    # Detect unresolved identifiers after rewriting; strings are intentionally preserved.
    new_tree = parse(bytes(result), '.cpp')
    for n in walk(new_tree.root_node):
        if n.type in IDENT_NODES and LABEL_TOKEN.search(text(n, bytes(result))):
            reasons.append('unresolved_label_identifier')
    return bytes(result), names, sorted(set(reasons))


def group_from_member(member):
    stem = PurePosixPath(member).stem
    m = FAMILY.match(stem)
    return (m.group(1) + '_' + m.group(2), m.group(1)) if m else (stem, None)


def zero_args_void(function, source):
    node = function['_node']
    typ = node.child_by_field_name('type')
    if typ is None or text(typ, source) != 'void':
        return False
    decl = node.child_by_field_name('declarator')
    params = next((n for n in walk(decl) if n.type == 'parameter_list'), None)
    return params is not None and text(params, source).replace(' ', '').replace('\n', '') in {'()', '(void)'}


def independent_good_branches(function, by_name, source):
    """Recognize only an exact, semantically empty sequence-of-calls wrapper."""
    if not zero_args_void(function, source):
        return None
    body = function['_node'].child_by_field_name('body')
    statements = [n for n in body.named_children if n.type != 'comment']
    branches = []
    for statement in statements:
        children = [n for n in statement.named_children if n.type != 'comment']
        if statement.type != 'expression_statement' or len(children) != 1 or children[0].type != 'call_expression':
            return None
        call = children[0]
        callee, args = call.child_by_field_name('function'), call.child_by_field_name('arguments')
        if callee.type != 'identifier' or args.named_child_count:
            return None
        name = text(callee, source)
        if not re.fullmatch(r'(?:.*_)?good(?:G2B|B2G)\d*', name):
            return None
        if len(by_name.get(name, [])) != 1 or not zero_args_void(by_name[name][0], source):
            return None
        branches.append(by_name[name][0])
    return branches if branches and len({f['name'] for f in branches}) == len(branches) else None



def context_facts(tree, function, functions, source, names):
    references = defaultdict(list)
    for node in walk(function['_node']):
        if node.type == 'identifier':
            references[text(node, source)].append((source.count(b'\n', 0, node.start_byte) + 1))
    facts = []
    for node in walk(tree.root_node):
        if node.type != 'declaration' or any(f['start_byte'] <= node.start_byte < f['end_byte'] for f in functions):
            continue
        if any(n.type == 'function_declarator' for n in walk(node)):
            continue
        declarators = [child for index, child in enumerate(node.children)
                       if node.field_name_for_child(index) == 'declarator']
        for declarator in declarators:
            name_node = declarator
            while name_node is not None and name_node.type != 'identifier':
                name_node = name_node.child_by_field_name('declarator')
            if name_node is None:
                continue
            name = text(name_node, source)
            if name not in references:
                continue
            facts.append({'sanitized_identifier': names.get(name, name),
                          'declaration_line_span': [(source.count(b'\n', 0, node.start_byte) + 1), (source.count(b'\n', 0, node.end_byte) + 1)],
                          'reference_lines': sorted(set(references[name])),
                          'is_extern_or_unknown': any(n.type == 'storage_class_specifier' and text(n, source) == 'extern' for n in node.children)})
    local = {f['name'] for f in functions}
    helpers = [dict(c, sanitized_name=names.get(c['name'], c['name'])) for c in function['calls']
               if c['direct'] and c['name'] not in local and c['name'] not in STANDARD_CALLS]
    return {'file_scope_references': facts, 'external_helper_calls': helpers,
            'evidence_scope': 'syntactic_declarations_and_entry_references_only_not_semantic_proof'}

def structural_context(tree, function, functions, source, family_files):
    reasons = []
    by_name = {f['name'] for f in functions}
    call_facts = function['calls']
    if tree.root_node.has_error:
        reasons.append('structural_parse_error')
    if len(family_files) > 1:
        reasons.append('multiple_family_files_context_not_proven_local')
    for call in call_facts:
        if not call['direct']:
            reasons.append('indirect_or_member_call_context_unresolved')
        elif call['name'] in by_name:
            reasons.append('interprocedural_user_function_context_unsupported')
        elif call['name'] not in STANDARD_CALLS:
            reasons.append('external_helper_context_review_required')
    if len([f for f in functions if f['name'] == function['name']]) != 1:
        reasons.append('ambiguous_function_ownership')
    # Only referenced external state/macros trigger review. This is syntactic evidence,
    # not a new C semantic engine. Unused declarations/includes do not reject a scene.
    entry_ids = {text(n, source) for n in walk(function['_node']) if n.type in IDENT_NODES}
    for node in walk(tree.root_node):
        if any(f['start_byte'] <= node.start_byte < f['end_byte'] for f in functions):
            continue
        if node.type in {'preproc_def', 'preproc_function_def'}:
            name = node.child_by_field_name('name')
            if name is not None and text(name, source) in entry_ids:
                reasons.append('referenced_macro_context_review_required')
    if context_facts(tree, function, functions, source, {})['file_scope_references']:
        reasons.append('referenced_file_scope_state_context_unresolved')
    return sorted(set(reasons))


def source_scenes(source, member, dataset, family_files, metadata=None):
    """Pure in-memory transformation; this function does not read any files."""
    tree, functions = function_info(source, PurePosixPath(member).suffix)
    syntax_error = tree.root_node.has_error
    sanitized, names, sanitation_reasons = neutralize(source, tree, functions)
    family, template = group_from_member(member)
    by_name = defaultdict(list)
    for f in functions:
        by_name[f['name']].append(f)
    entries = []
    if dataset == 'mit':
        roots = by_name.get('main', [])
        for f in roots:
            entries.append((f, metadata.get('label'), 'official_SARIF_state_variant_result', None))
    else:
        for f in functions:
            if re.fullmatch(r'(?:.*_)?bad', f['name']):
                entries.append((f, 1, 'Juliet_bad_entry_definition', None))
            elif re.fullmatch(r'(?:.*_)?good', f['name']):
                branches = independent_good_branches(f, by_name, source)
                if branches:
                    entries.extend((b, 0, 'Juliet_good_entry_pure_wrapper_branch', f['name']) for b in branches)
                else:
                    entries.append((f, 0, 'Juliet_good_entry_definition', None))
    scenes = []
    for f, label, provenance, wrapper in entries:
        reasons = structural_context(tree, f, functions, source, family_files) + sanitation_reasons
        if syntax_error:
            reasons.append('tree_sitter_syntax_error')
        scene = {
            'scene_id': dataset + ':' + family + ':' + f['name'],
            'record_kind': 'scene',
            'label': label, 'label_provenance': {'rule': provenance, 'entry': f['name'], 'wrapper': wrapper},
            'family_id': family, 'template_group': template or family,
            'source_member': member, 'source_line_spans': [[f['start_line'], f['end_line']]],
            'owned_entry': names.get(f['name'], f['name']),
            'sanitization_accepted': not sanitation_reasons,
            'raw_sha256': sha(source), 'effective_sha256': sha(source),
            'source_hash_scope': 'whole_source_member_not_scene_clone_evidence',
            'entry_raw_sha256': sha(source[f['start_byte']:f['end_byte']]),
            'sanitized_sha256': sha(sanitized), 'line_map': 'identity_1_based',
            'context_reasons': sorted(set(reasons)),
            'context_facts': context_facts(tree, f, functions, source, names),
            'crossfile_context': 'unresolved' if len(family_files) > 1 else 'not_indicated_by_family_members',
            'family_source_members': sorted(family_files),
            'same_file_helper_definitions': [
                {'name': helper['name'], 'source_line_spans': [[helper['start_line'], helper['end_line']]]}
                for helper in functions if helper['name'] in {c['name'] for c in f['calls']}
            ],
            'calls': f['calls'],
            'status': 'parse_error' if syntax_error else 'context_unresolved' if reasons else 'candidate_pending_frontend',
            'task_applicability': 'pending_dataset_rule',
            'slice_inventory': None, 'primary_processable': False,
            'model_parser': 'legacy_frontend_only',
        }
        if label is None:
            scene['status'] = 'label_unresolved'
        if dataset == 'juliet':
            scene['entry_structure'] = f['entry_structure']
        scenes.append(scene)
    if not scenes:
        scenes.append({'scene_id': dataset + ':' + family + ':unresolved:' + sha(member.encode())[:12],
                       'label': metadata.get('label') if dataset == 'mit' else None,
                       'record_kind': 'scene' if dataset == 'mit' else 'source_placeholder_not_resolved_scene',
                       'label_provenance': {'rule': 'official_SARIF_state_variant_result' if dataset == 'mit' else 'no_reliable_entry'},
                       'family_id': family, 'template_group': template or family,
                       'source_member': member, 'source_line_spans': [],
                       'raw_sha256': sha(source), 'effective_sha256': sha(source),
            'source_hash_scope': 'whole_source_member_not_scene_clone_evidence',
            'entry_raw_sha256': None,
                       'sanitized_sha256': sha(sanitized), 'line_map': 'identity_1_based',
                       'status': 'parse_error' if syntax_error else 'context_unresolved' if dataset == 'mit' and metadata.get('label') is not None else 'label_unresolved',
                       'context_reasons': ['entry_ownership_unresolved'],
                       'crossfile_context': 'unknown', 'task_applicability': 'pending_dataset_rule',
                       'slice_inventory': None, 'primary_processable': False,
                       'parse_error': syntax_error})
    return scenes, sanitized


def aggregate_scene(risks, expected_slice_ids, inventory_complete=True):
    """Useful to downstream evaluation; missing scores cannot produce a safe verdict."""
    expected = set(expected_slice_ids)
    if set(risks) - expected:
        raise ValueError('score outside original slice inventory')
    values = [v for v in risks.values() if v is not None]
    if any(not 0 <= v <= 1 for v in values):
        raise ValueError('invalid risk')
    complete = bool(expected) and inventory_complete and all(risks.get(k) is not None for k in expected)
    maximum = max(values, default=None)
    verdict = 1 if maximum is not None and maximum >= .5 else 0 if complete else None
    return {'max_observed_risk': maximum, 'prediction': verdict,
            'complete': complete, 'requires_shared_scene_and_method_gate': True,
            'scope': 'complete_scene' if complete else 'partial_diagnostic_only'}


def qualified_scene_identity(scene, dataset):
    """Qualify identity and retain unproven independent entry units as unresolved.

    Cached source transforms remain immutable. A spelling may denote several
    definitions (including constructor/destructor or distinct translation units).
    Existing inferred label evidence is preserved as a sidecar, never treated as
    a known independent scene when the syntax/ownership does not support that.
    """
    if dataset != 'juliet' or scene.get('record_kind', 'scene') != 'scene':
        return scene
    identity = {key: scene.get(key) for key in
                ['source_member', 'source_line_spans', 'entry_raw_sha256']}
    if not identity['source_member'] or not identity['source_line_spans'] or not identity['entry_raw_sha256']:
        raise ValueError('Resolved scene lacks definition identity metadata')
    record = dict(scene)
    record['legacy_scene_id'] = scene['scene_id']
    record['scene_id'] = scene['scene_id'] + ':definition:' + digest(identity)
    record['identity_scope'] = 'source_member_line_span_entry_bytes_not_cpp_ownership_resolution'
    structure = scene.get('entry_structure', {})
    unresolved = []
    if 'ambiguous_function_ownership' in scene.get('context_reasons', []):
        unresolved.append('ambiguous_function_ownership')
    if structure.get('has_return_type') is False or structure.get('destructor_declarator'):
        unresolved.append('constructor_destructor_or_missing_return_type_not_independent_entry')
    if structure.get('class_member_definition') or structure.get('qualified_declarator'):
        unresolved.append('class_or_qualified_definition_entry_ownership_unproven')
    if unresolved:
        record.update(record_kind='unresolved_entry', label=None, status='label_unresolved',
                      inferred_entry_label=scene.get('label'),
                      inferred_entry_label_provenance=scene.get('label_provenance'),
                      independent_scene_multiplicity='unknown',
                      unresolved_source_group=dataset + ':source:' + sha(scene['source_member'].encode()),
                      unresolved_entry_reasons=unresolved)
        record['label_provenance'] = {'rule': 'independent_scene_label_unit_unresolved',
                                      'original_inference_retained': True}
    return record


def metadata_label(run, variant, mapping):
    state = run.get('properties', {}).get('state')
    label = {'good': 0, 'bad': 1}.get(state)
    kinds = {r.get('kind', 'fail') for r in run.get('results', [])}
    expected = 'pass' if label == 0 else 'fail'
    return label if label == mapping.get(variant) and kinds == {expected} else None


def verify_archive(path, expected):
    if file_sha(path) != expected:
        raise ValueError('archive SHA256 mismatch: ' + str(path))


def gate(receipt, config):
    from input_contract import verify_profile
    record = verify_profile(receipt)
    if record.get('static_manifest_gate_enabled') is not True:
        raise ValueError('finite profile does not authorize static manifest processing')
    if record.get('tool_and_author_train_golden_parity') != 'PASS':
        raise ValueError('original author TRAIN parity missing')
    if record['request']['receipt_sha256']['dataset_configs.json'] != file_sha(config):
        raise ValueError('finite profile does not freeze this dataset config')
    # This is explicitly a finite static-processing gate, never all-C support or inference.
    return record


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


class MemberCache:
    """Each successful source transformation is one SQLite transaction, safe to resume."""
    def __init__(self, output):
        self.output = Path(output)
        self.db = sqlite3.connect(self.output / 'member_progress.sqlite3')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS members (member_key TEXT PRIMARY KEY, payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL)')
        self.db.commit()

    def get(self, key):
        row = self.db.execute('SELECT payload,payload_sha256 FROM members WHERE member_key=?', (key,)).fetchone()
        if row is None:
            return None
        if sha(row[0].encode()) != row[1]:
            raise ValueError('committed member payload hash mismatch')
        records = json.loads(row[0])
        checked = set()
        for record in records:
            relative = record.get('sanitized_path')
            if relative and relative not in checked:
                path = self.output / relative
                if file_sha(path) != record['sanitized_sha256']:
                    raise ValueError('committed sanitized source missing or changed')
                checked.add(relative)
        return records

    def put(self, key, records):
        payload = json.dumps(records, sort_keys=True, separators=(',', ':'))
        with self.db:
            self.db.execute('INSERT INTO members VALUES (?,?,?)', (key, payload, sha(payload.encode())))

    def verify_all(self):
        keys = [r[0] for r in self.db.execute('SELECT member_key FROM members ORDER BY member_key')]
        for key in keys:
            self.get(key)
        return len(keys)

    def close(self):
        self.db.close()


def write_source(output, archive_id, member, sanitized):
    relative = Path('sanitized') / (sha((archive_id + ':' + member).encode())[:32] + PurePosixPath(member).suffix)
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha(path) != sha(sanitized):
            raise ValueError('existing sanitized source differs; refusing overwrite')
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(sanitized)
        temporary.replace(path)
    return str(relative)


def juliet_records(config, output, summary, cache=None):
    path = ROOT / config['archive']
    verify_archive(path, config['archive_sha256'])
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        by_base, families = defaultdict(list), defaultdict(list)
        for member in members:
            if PurePosixPath(member).suffix in SOURCE_EXT | {'.h'}:
                by_base[PurePosixPath(member).name].append(member)
                families[group_from_member(member)[0]].append(member)
        census = Counter()
        processed_members = set()
        for member in members:
            if PurePosixPath(member).suffix not in SOURCE_EXT:
                continue
            match = re.search(r'(?:^|/)CWE(\d+)[_/]', member)
            if match:
                census[int(match.group(1))] += 1
        summary['raw_source_file_count_by_cwe'] = {str(key): value for key, value in sorted(census.items())}
        summary['candidate_cwes_absent'] = sorted(set(config['target_cwes']) - set(census))
        official = json.loads((ROOT / config['official_metadata']).read_text())
        stale = {r['xml_testcase_ordinal_1_based']: r for r in
                 official['juliet_1_3']['metadata_member_crosscheck']['stale_testcases']}
        summary['known_removed_xml_case_count'] = len(stale)
        summary['known_removed_source_file_count'] = sum(len(r['missing_file_paths']) for r in stale.values())
        summary['known_removed_counts_are_not_scene_counts'] = True
        xml = ET.fromstring(archive.read(config['metadata_member']))
        summary['xml_testcase_count'] = len(list(xml))
        for ordinal, testcase in enumerate(xml, 1):
            paths = [f.attrib['path'] for f in testcase.findall('file')]
            if not paths:
                continue
            match = re.match(r'CWE(\d+)_', PurePosixPath(paths[0]).name)
            if not match:
                continue
            cwe = int(match.group(1))
            if cwe not in config['target_cwes'] + config['arithmetic_diagnostic_cwes']:
                continue
            family, template = group_from_member(paths[0])
            absent = [p for p in paths if len(by_base[PurePosixPath(p).name]) != 1]
            if absent:
                yield {'scene_id': 'juliet:' + family + ':metadata_unresolved', 'label': None,
                       'label_provenance': {'rule': 'no_source_entry_available'},
                       'family_id': family, 'template_group': template or family,
                       'source_archive': config['archive'], 'source_archive_sha256': config['archive_sha256'],
                       'source_members': paths, 'source_line_spans': [], 'xml_testcase_ordinal': ordinal,
                       'status': 'official_known_removed_metadata_remnant' if ordinal in stale else 'source_missing_or_ambiguous',
                       'record_kind': 'case_placeholder_not_counted_as_resolved_scene',
                       'official_removal_evidence': stale.get(ordinal),
                       'context_reasons': ['source_entry_unavailable'],
                       'task_applicability': 'target' if cwe in config['target_cwes'] else 'diagnostic_pending_memory_check',
                       'cwe': cwe, 'primary_processable': False, 'raw_sha256': None, 'effective_sha256': None,
                       'missing_or_ambiguous_members': absent, 'crossfile_context': 'unknown'}
                continue
            source_members = [by_base[PurePosixPath(p).name][0] for p in paths
                              if PurePosixPath(p).suffix in SOURCE_EXT]
            for member in source_members:
                if member in processed_members:
                    raise ValueError('duplicate XML source ownership; requires metadata resolution')
                processed_members.add(member)
                cached = cache.get(member) if cache else None
                if cached is not None:
                    yield from cached
                    continue
                source = archive.read(member)
                scenes, sanitized = source_scenes(source, member, 'juliet', families[family])
                sanitized_path = write_source(output, config['archive_sha256'], member, sanitized)
                for scene in scenes:
                    scene.update(source_archive=config['archive'], source_archive_sha256=config['archive_sha256'],
                                 xml_testcase_ordinal=ordinal, cwe=cwe, sanitized_path=sanitized_path,
                                 task_applicability='target' if cwe in config['target_cwes'] else 'diagnostic_pending_memory_check')
                if cache:
                    cache.put(member, scenes)
                yield from scenes
        summary['target_members_missing_from_xml'] = [m for m in members
            if PurePosixPath(m).suffix in SOURCE_EXT and m not in processed_members
            and (re.search(r'(?:^|/)CWE(\d+)[_/]', m) is not None)
            and int(re.search(r'(?:^|/)CWE(\d+)[_/]', m).group(1)) in config['target_cwes']]
        # Explicit unresolved rows retain otherwise unrepresented target sources.
        for member in summary['target_members_missing_from_xml']:
            family, template = group_from_member(member)
            yield {'scene_id': 'juliet:unindexed:' + sha(member.encode())[:20], 'label': None,
                   'label_provenance': {'rule': 'source_without_XML_case'},
                   'record_kind': 'source_placeholder_not_resolved_scene', 'family_id': family,
                   'template_group': template or family, 'source_archive': config['archive'],
                   'source_member': member, 'source_line_spans': [], 'status': 'label_unresolved',
                   'task_applicability': 'target', 'raw_sha256': None, 'effective_sha256': None,
                   'crossfile_context': 'unknown', 'context_reasons': ['metadata_ownership_unresolved'],
                   'primary_processable': False}


def mit_records(config, output, summary, cache=None):
    path = ROOT / config['archive']
    verify_archive(path, config['archive_sha256'])
    supplement = config['supplement']
    verify_archive(ROOT / supplement['archive'], supplement['archive_sha256'])
    variant_counts, groups = Counter(), defaultdict(list)
    with zipfile.ZipFile(path) as archive, zipfile.ZipFile(ROOT / supplement['archive']) as patch:
        for manifest in sorted(n for n in archive.namelist() if n.endswith('/manifest.sarif')):
            original_id = manifest.split('/')[0]
            runs = json.loads(archive.read(manifest)).get('runs', [])
            if len(runs) != 1:
                raise ValueError('ambiguous SARIF runs')
            run = runs[0]
            artifacts = run.get('artifacts', [])
            matches = [(a, re.fullmatch(r'(basic-\d+)-(min|med|large|ok)\.c', PurePosixPath(a['location']['uri']).name)) for a in artifacts]
            matches = [(a, m) for a, m in matches if m]
            if len(matches) != 1:
                raise ValueError('ambiguous MIT variant source metadata')
            artifact, match = matches[0]
            base, variant = match.groups()
            label = metadata_label(run, variant, config['variants'])
            member = original_id + '/' + artifact['location']['uri']
            variant_counts[variant] += 1
            groups[base].append(variant)
            cached = cache.get(member) if cache else None
            if cached is not None:
                yield from cached
                continue
            raw = archive.read(member)
            original_hash = sha(raw)
            if original_hash != artifact.get('hashes', {}).get('sha-256'):
                raise ValueError('original MIT source hash mismatch')
            effective, effective_id, effective_archive, effective_member = raw, original_id, config['archive'], member
            effective_run = run
            reason = None
            if original_id == supplement['original_identifier']:
                if (base, variant) != (supplement['base_case'], supplement['variant']):
                    raise ValueError('supplement mapping mismatch')
                effective_run = json.loads(patch.read('manifest.sarif'))['runs'][0]
                properties = effective_run.get('properties', {})
                if str(properties.get('id')) + '-v' + str(properties.get('version')) != supplement['effective_identifier']:
                    raise ValueError('supplement official ID/version mismatch')
                if properties.get('status') != 'candidate':
                    raise ValueError('supplement is not an official candidate')
                art = effective_run['artifacts'][0]
                effective_member = art['location']['uri']
                effective = patch.read(effective_member)
                if sha(effective) != art['hashes']['sha-256']:
                    raise ValueError('supplement source hash mismatch')
                effective_id, effective_archive = supplement['effective_identifier'], supplement['archive']
                reason = supplement['replacement_reason']
                label = metadata_label(effective_run, variant, config['variants'])
            scenes, sanitized = source_scenes(effective, effective_member, 'mit', [effective_member], {'label': label})
            # Exactly one official record is one scene, even if entry recovery fails.
            if len(scenes) != 1:
                scenes = [dict(scenes[0], status='context_unresolved', context_reasons=['ambiguous_main_entry'])]
            scene = scenes[0]
            scene.update(scene_id='mit:' + base + ':' + variant, label=label,
                         family_id=base, template_group=base, variant=variant,
                         original_identifier=original_id, original_label=metadata_label(run, variant, config['variants']),
                         original_properties=run.get('properties', {}),
                         effective_identifier=effective_id, effective_properties=effective_run.get('properties', {}),
                         replacement_reason=reason, raw_sha256=original_hash, effective_sha256=sha(effective),
                         source_archive=config['archive'], source_archive_sha256=config['archive_sha256'],
                         source_member=member, effective_archive=effective_archive, effective_member=effective_member,
                         effective_archive_sha256=supplement['archive_sha256'] if reason else config['archive_sha256'],
                         sanitized_path=write_source(output, effective_archive, effective_member, sanitized),
                         task_applicability='target', label_provenance={'rule': 'official_SARIF_state_variant_result',
                         'original_manifest': manifest, 'effective_state': effective_run.get('properties', {}).get('state')})
            if label is None:
                scene['status'] = 'label_unresolved'
            if cache:
                cache.put(member, [scene])
            yield scene
    summary['official_variant_counts'] = dict(variant_counts)
    summary['base_case_count'] = len(groups)
    summary['incomplete_or_duplicate_variant_groups'] = {k: v for k, v in groups.items() if sorted(v) != sorted(config['variants'])}


def self_test():
    # JSON object keys must already be strings before receipt hashing: numeric
    # order (78, 121) differs from the JSON-reloaded lexical order (121, 78).
    census_receipt = {'raw_source_file_count_by_cwe': {str(key): value for key, value in [(78, 2), (121, 3), (680, 1)]}}
    census_receipt['receipt_sha256'] = digest(census_receipt)
    roundtrip = json.loads(json.dumps(census_receipt, sort_keys=True))
    expected = roundtrip.pop('receipt_sha256')
    assert digest(roundtrip) == expected
    # Hand-authored synthetic fixtures; no archive or dataset source reads.
    source = b'''/* bad CWE121 hint */\n#include <string.h>\nvoid sample_01_bad(void) { char badBuffer[4]; memcpy(badBuffer, "abcdef", 6); }\nstatic void goodG2B(void) { char goodBuffer[8]; memcpy(goodBuffer, "abcdef", 6); }\nstatic void goodB2G(void) { unsigned int n = 2; char a[4]; if(n < sizeof(a)) a[n] = 'x'; }\nvoid sample_01_good(void) { goodG2B(); goodB2G(); }\n'''
    records, sanitized = source_scenes(source, 'CWE121_fixture_01.c', 'juliet', ['CWE121_fixture_01.c'])
    assert len(records) == 3 and sorted(r['label'] for r in records) == [0, 0, 1]
    assert all(r['status'] == 'candidate_pending_frontend' for r in records)
    assert b'badBuffer' not in sanitized and b'goodG2B' not in sanitized and b'CWE121 hint' not in sanitized
    assert b'"abcdef"' in sanitized and b'unsigned int' in sanitized and b'sizeof(a)' in sanitized
    assert sanitized.count(b'\n') == source.count(b'\n')
    guarded = b'void goodG2B(void) {} void x_good(void) { if (1) goodG2B(); }'
    records, _ = source_scenes(guarded, 'CWE121_fixture_02.c', 'juliet', ['x.c'])
    assert len(records) == 1 and records[0]['context_reasons'] == ['interprocedural_user_function_context_unsupported']
    cross = b'void x_bad(void) { externalSink(); }'
    records, _ = source_scenes(cross, 'CWE121_fixture_51a.c', 'juliet', ['a.c', 'b.c'])
    assert records[0]['status'] == 'context_unresolved'
    assert group_from_member('CWE121_fixture_51a.c') == ('CWE121_fixture_51', 'CWE121_fixture')
    assert group_from_member('CWE121_fixture_84_goodG2B.cpp') == ('CWE121_fixture_84', 'CWE121_fixture')
    cpp = b'namespace CWE121_test { void bad() { int badArray[2]; badArray[3]=1; } }'
    records, sanitized = source_scenes(cpp, 'CWE121_fixture_01.cpp', 'juliet', ['a.cpp'])
    assert len(records) == 1 and records[0]['status'] == 'candidate_pending_frontend'
    assert b'CWE121' not in sanitized
    facts_src = b'int global=5; extern int remote;\nvoid x_bad(void) { int x=global+remote; printIntLine(x); }\n'
    facts_rows, _ = source_scenes(facts_src, 'CWE121_fixture_01.c', 'juliet', ['a.c'])
    facts = facts_rows[0]['context_facts']
    assert len(facts['file_scope_references']) == 2
    assert [x['is_extern_or_unknown'] for x in facts['file_scope_references']] == [False, True]
    assert facts['external_helper_calls'][0]['line'] == 2
    unknown, _ = source_scenes(b'int mystery(void) { return 0; }', 'CWE121_fixture_01.c', 'juliet', ['a.c'])
    assert unknown[0]['label'] is None and unknown[0]['status'] == 'label_unresolved'
    assert unknown[0]['entry_raw_sha256'] is None
    empty, _ = source_scenes(b'int only_global;\n', 'CWE121_fixture_01.c', 'juliet', ['a.c'])
    assert empty[0]['entry_raw_sha256'] is None and empty[0]['label'] is None
    high_line = b'\n' * 300 + b'void test_bad(void) { char a[4]; a[1]=0; }\n'
    for _ in range(50):
        high_rows, _ = source_scenes(high_line, 'CWE121_hand_01.c', 'juliet', ['a.c'])
        assert high_rows[0]['source_line_spans'] == [[301, 301]]
    macro = b'#define CALL bad()\nvoid bad(void) {}'
    records, _ = source_scenes(macro, 'CWE121_fixture_01.c', 'juliet', ['a.c'])
    assert 'opaque_macro_payload_needs_identifier_resolution' in records[0]['context_reasons']
    assert aggregate_scene({'a': .7}, ['a', 'b'])['prediction'] == 1
    assert not aggregate_scene({'a': .7}, ['a', 'b'])['complete']
    assert aggregate_scene({'a': .2}, ['a', 'b'])['prediction'] is None
    assert aggregate_scene({}, [])['prediction'] is None
    assert aggregate_scene({'a': .2}, ['a'])['prediction'] == 0
    assert metadata_label({'properties': {'state': 'good'}, 'results': [{'kind': 'pass'}]}, 'ok', {'ok': 0}) == 0
    assert metadata_label({'properties': {'state': 'bad'}, 'results': [{}]}, 'ok', {'ok': 0}) is None
    integration_self_test()
    print(json.dumps({'self_test': 'PASS', 'scope': 'handwritten C/C++ and handwritten ZIP/XML/SARIF fixtures only',
                      'external_source_reads': 0, 'training': False, 'inference': False,
                      'implementation_sha256': file_sha(__file__),
                      'config_sha256': file_sha(HERE / 'dataset_configs.json'),
                      'accepted_external_scenes': None,
                      'checks': ['multi_digit_CWE_receipt_JSON_roundtrip', 'entry_and_helper_ownership', 'good_pure_wrapper_split', 'guarded_wrapper_not_split',
                                 'crossfile_context', 'comment_and_identifier_sanitization', 'literal_type_line_preservation',
                                 'opaque_macro_rejection', 'partial_slice_unknown', 'zero_slice_unknown',
                                 'SARIF_label_disagreement', 'stale_XML_case_retention', 'file_vs_scene_counts',
                                 'MIT_4_variant_pairing', 'MIT_supplement_original_effective_hashes', 'original_archive_unchanged', 'no_function_entry_hash_null', 'nonlabel_function_entry_hash_null',
                                 'interrupted_member_resume_without_reprocessing', 'complete_hash_skip', 'changed_request_refused', 'deferred_global_and_helper_context_facts', 'high_line_301_native_point_regression']}))



def integration_self_test():
    """Hand-built archive schemas exercise orchestration without real external sources."""
    with tempfile.TemporaryDirectory(prefix='manifest-handfixture-') as directory:
        root = Path(directory)
        juliet = root / 'juliet.zip'
        member = 'C/testcases/CWE121_fixture/CWE121_hand_01.c'
        missing = 'CWE121_missing_21.c'
        src = b'void a_bad(void) { char a[2]; a[3]=0; }\nvoid a_good(void) { char a[4]; a[3]=0; }\n'
        with zipfile.ZipFile(juliet, 'w') as z:
            z.writestr(member, src)
            z.writestr('C/manifest.xml', '<container><testcase><file path="CWE121_hand_01.c"><flaw line="999"/></file></testcase><testcase><file path="' + missing + '"/></testcase></container>')
        official = root / 'official.json'
        official.write_text(json.dumps({'juliet_1_3': {'metadata_member_crosscheck': {'stale_testcases': [
            {'xml_testcase_ordinal_1_based': 2, 'scenario_id': 'CWE121_missing_21', 'missing_file_paths': [missing]}]}}}))
        config = {'archive': str(juliet), 'archive_sha256': file_sha(juliet), 'metadata_member': 'C/manifest.xml',
                  'official_metadata': str(official), 'target_cwes': [121, 125], 'arithmetic_diagnostic_cwes': [190, 191]}
        summary = {}
        records = list(juliet_records(config, root / 'out-juliet', summary))
        assert len(records) == 3 and sum(r['record_kind'] == 'scene' for r in records) == 2
        assert records[-1]['status'] == 'official_known_removed_metadata_remnant'
        assert summary['candidate_cwes_absent'] == [125]
        # Flaw line 999 was neither a selected key nor a source-line ownership hint.
        assert records[0]['source_line_spans'][0][0] == 1
        mit, patch = root / 'mit.zip', root / 'patch.zip'
        variants = {'min': 1, 'med': 1, 'large': 1, 'ok': 0}
        identifiers = {'min': '833-v1.0.0', 'med': '832-v1.0.0', 'large': '831-v1.0.0', 'ok': '2073-v1.0.0'}
        def manifest(variant, source, identifier, status='candidate'):
            return {'runs': [{'properties': {'id': 834 if status == 'deprecated' else 2073,
                'state': 'good' if variant == 'ok' else 'bad', 'status': status,
                'version': '2.0.0' if identifier == '2073-v2.0.0' else '1.0.0'},
                'artifacts': [{'location': {'uri': 'src/basic-00180-' + variant + '.c'},
                               'hashes': {'sha-256': sha(source)}}],
                'results': [{'kind': 'pass' if variant == 'ok' else 'fail'}]}]}
        original = b'int main(void) { char a[4]; a[1]=0; return 0; }\n'
        replacement = b'int main(void) { char a[5]; a[1]=0; return 0; }\n'
        with zipfile.ZipFile(mit, 'w') as z:
            for variant, identifier in identifiers.items():
                z.writestr(identifier + '/manifest.sarif', json.dumps(manifest(variant, original, identifier, 'deprecated' if variant == 'ok' else 'candidate')))
                z.writestr(identifier + '/src/basic-00180-' + variant + '.c', original)
        with zipfile.ZipFile(patch, 'w') as z:
            z.writestr('manifest.sarif', json.dumps(manifest('ok', replacement, '2073-v2.0.0')))
            z.writestr('src/basic-00180-ok.c', replacement)
        config = {'archive': str(mit), 'archive_sha256': file_sha(mit), 'variants': variants,
                  'supplement': {'archive': str(patch), 'archive_sha256': file_sha(patch),
                      'original_identifier': '2073-v1.0.0', 'effective_identifier': '2073-v2.0.0',
                      'base_case': 'basic-00180', 'variant': 'ok', 'replacement_reason': 'hand-fixture official replacement'}}
        summary = {}
        records = list(mit_records(config, root / 'out-mit', summary))
        assert len(records) == 4 and summary['base_case_count'] == 1
        assert not summary['incomplete_or_duplicate_variant_groups']
        revised = next(r for r in records if r['variant'] == 'ok')
        assert revised['original_identifier'] == '2073-v1.0.0' and revised['effective_identifier'] == '2073-v2.0.0'
        assert revised['raw_sha256'] == sha(original) and revised['effective_sha256'] == sha(replacement)
        assert revised['original_properties']['id'] == 834 and revised['label'] == revised['original_label'] == 0
        assert all(r['primary_processable'] is False for r in records)
        assert file_sha(mit) == config['archive_sha256']
        resume_output = root / 'resume-handfixture'
        resume_output.mkdir()
        run_config = {'protocol_version': 'handfixture', 'datasets': {'mit': config}}
        request = {'dataset': 'mit', 'handfixture': True}
        original_transform = globals()['source_scenes']
        calls = []
        def interrupted_transform(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise RuntimeError('intentional handfixture interruption')
            return original_transform(*args, **kwargs)
        globals()['source_scenes'] = interrupted_transform
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    run_manifest('mit', run_config, resume_output, request)
                    raise AssertionError('interruption fixture did not interrupt')
                except RuntimeError as error:
                    assert str(error) == 'intentional handfixture interruption'
                assert json.loads((resume_output / 'manifest_receipt.json').read_text())['status'] == 'RUNNING'
                first_completed = calls[0]
                calls.clear()
                def counted_transform(*args, **kwargs):
                    calls.append(args[1])
                    return original_transform(*args, **kwargs)
                globals()['source_scenes'] = counted_transform
                run_manifest('mit', run_config, resume_output, request)
                assert len(calls) == 3 and first_completed not in calls
                calls.clear()
                run_manifest('mit', run_config, resume_output, request)
                assert not calls
                try:
                    run_manifest('mit', run_config, resume_output, dict(request, changed=True))
                    raise AssertionError('changed request was accepted')
                except ValueError as error:
                    assert 'request differs' in str(error)
        finally:
            globals()['source_scenes'] = original_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--dataset', choices=['juliet', 'mit'])
    parser.add_argument('--config', default=str(HERE / 'dataset_configs.json'))
    parser.add_argument('--acceptance-receipt')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dataset or not args.output or not args.acceptance_receipt:
        parser.error('external generation requires --dataset --output --acceptance-receipt')
    profile = gate(args.acceptance_receipt, args.config)
    config = json.loads(Path(args.config).read_text())
    request = {'dataset': args.dataset, 'config_sha256': file_sha(args.config),
               'implementation_sha256': file_sha(__file__), 'profile_sha256': file_sha(args.acceptance_receipt),
               'profile_scope': profile['status'], 'legacy_fixture_overall_status': profile['legacy_fixture_overall_status']}
    if args.dataset == 'juliet':
        request['official_metadata_sha256'] = file_sha(ROOT / config['datasets']['juliet']['official_metadata'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Advisory lock prevents two writers from racing or duplicating successful work.
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_manifest(args.dataset, config, output, request)


def run_manifest(dataset, config, output, request):
    selected = config['datasets'][dataset]
    verify_archive(ROOT / selected['archive'], selected['archive_sha256'])
    if dataset == 'mit':
        verify_archive(ROOT / selected['supplement']['archive'], selected['supplement']['archive_sha256'])
    request_hash = digest(request)
    request_path = output / 'request.json'
    if request_path.exists():
        saved = json.loads(request_path.read_text())
        if saved != {'request': request, 'request_sha256': request_hash}:
            raise ValueError('existing output request differs; refusing overwrite')
    else:
        if any(p.name != 'run.lock' for p in output.iterdir()):
            raise ValueError('existing output without matching request; refusing overwrite')
        atomic_json(request_path, {'request': request, 'request_sha256': request_hash})
    cache = MemberCache(output)
    receipt_path = output / 'manifest_receipt.json'
    try:
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('request_sha256') != request_hash:
                raise ValueError('receipt request mismatch')
            if receipt.get('status') == 'PASS':
                expected = receipt.pop('receipt_sha256', None)
                if expected != digest(receipt):
                    raise ValueError('completed receipt self hash mismatch')
                if file_sha(output / 'scenes.jsonl') != receipt['scenes_sha256']:
                    raise ValueError('completed scene manifest changed')
                cache.verify_all()
                print(json.dumps({'status': 'SKIP_VERIFIED_COMPLETE', 'output': str(output)}))
                return receipt
        summary = {'dataset': dataset, 'protocol_version': config['protocol_version'],
                   'request_sha256': request_hash, 'request': request, 'status': 'RUNNING',
                   'source_text_logged': False, 'primary_processable_count': 0, 'diagnostic_families': [],
                   'scene_count_is_not_source_file_count': True}
        atomic_json(receipt_path, summary)
        records = juliet_records if dataset == 'juliet' else mit_records
        counts, statuses, record_kinds, seen = Counter(), Counter(), Counter(), set()
        unresolved_sources = set()
        temporary = output / 'scenes.jsonl.tmp'
        with temporary.open('w') as stream:
            for scene in records(config['datasets'][dataset], output, summary, cache):
                scene = qualified_scene_identity(scene, dataset)
                if scene['scene_id'] in seen:
                    raise ValueError('duplicate scene identity; no manifest accepted')
                seen.add(scene['scene_id'])
                counts[str(scene['label'])] += 1
                statuses[scene['status']] += 1
                record_kinds[scene.get('record_kind', 'scene')] += 1
                if scene.get('unresolved_source_group'):
                    unresolved_sources.add(scene['unresolved_source_group'])
                stream.write(json.dumps(scene, ensure_ascii=True) + '\n')
        temporary.replace(output / 'scenes.jsonl')
        summary.update(manifest_records=len(seen), resolved_scene_records=record_kinds['scene'],
                       unresolved_entry_source_groups=len(unresolved_sources),
                       unresolved_entry_scene_multiplicity='unknown_not_record_count',
                       record_kind_counts=dict(record_kinds), label_counts=dict(counts), status_counts=dict(statuses),
                       status='PASS', completion='static_manifest_only_not_model_acceptance',
                       committed_source_members=cache.verify_all(), scenes_sha256=file_sha(output / 'scenes.jsonl'))
        summary['receipt_sha256'] = digest(summary)
        atomic_json(receipt_path, summary)
        print(json.dumps({'status': summary['completion'], 'records': len(seen), 'output': str(output)}))
        return summary
    finally:
        cache.close()


if __name__ == '__main__':
    main()
