"""Source-position obligations and pure external-scene aggregation.

The numerical core, line-unique linker and trace reader are imported unchanged.
No labels, model scores, or official vulnerability locations are consulted.
"""
import collections
import hashlib

UNKNOWN = 'UNKNOWN'
SCHEMA = 'external_source_address_obligations_v1'


def exact_decision(v, allowed):
    return (type(v) is int and v in allowed) or (type(v) is str and v == UNKNOWN)


def aggregate_scene(obligations, inventory_complete):
    out = {'schema': SCHEMA, 'D_pos': UNKNOWN, 'D_trace': UNKNOWN,
           'positive_obligations': [], 'unresolved_obligations': [], 'reasons': [],
           'negative_scope': 'all enumerated necessary accesses on completed finite source-address model traces only',
           'full_C_program_safety': UNKNOWN, 'reachability': 'NOT_PROVED'}
    if not isinstance(obligations, list) or type(inventory_complete) is not bool:
        out['reasons'] = ['INVALID_INVENTORY_CONTAINER_OR_COMPLETENESS']
        return out
    if not obligations:
        out['reasons'] = ['EMPTY_NECESSARY_ACCESS_INVENTORY']
        return out
    counts = collections.Counter(a.get('id') for a in obligations if isinstance(a, dict) and type(a.get('id')) is str)
    complete = []
    required = {'id', 'identity_valid', 'task_scope_valid', 'D_pos', 'D_trace',
                'complete_trace', 'query_visited', 'failure'}
    for j, a in enumerate(obligations):
        sid = a.get('id', 'invalid:' + str(j)) if isinstance(a, dict) else 'invalid:' + str(j)
        reason = None
        if not isinstance(a, dict) or not required.issubset(a):
            reason = 'MISSING_REQUIRED_EVIDENCE'
        elif type(a['id']) is not str or not a['id'] or counts[a['id']] != 1:
            reason = 'DUPLICATE_OR_INVALID_OBLIGATION_ID'
        elif any(type(a[k]) is not bool for k in ['identity_valid', 'task_scope_valid', 'complete_trace', 'query_visited']):
            reason = 'INVALID_BOOLEAN_CERTIFICATE'
        elif not exact_decision(a['D_pos'], {1}) or not exact_decision(a['D_trace'], {0, 1}):
            reason = 'INVALID_DECISION_TYPE_OR_VALUE'
        elif a['D_pos'] == 1 and a['D_trace'] == 0:
            reason = 'POSITIVE_NEGATIVE_READOUT_CONFLICT'
        elif a['D_trace'] == 1 and a['D_pos'] != 1:
            reason = 'TRACE_POSITIVE_NOT_SUPPORTED_BY_D_POS'
        elif a['failure'] not in (None, '', False):
            reason = 'FAILED_OBLIGATION'
        elif not a['identity_valid'] or not a['task_scope_valid']:
            reason = 'IDENTITY_OR_TASK_SCOPE_UNVERIFIED'
        elif not a['query_visited']:
            reason = 'ACCESS_NOT_OBSERVED'
        if reason:
            out['unresolved_obligations'].append({'id': sid, 'reason': reason})
            complete.append(False)
            continue
        if a['D_pos'] == 1:
            out['positive_obligations'].append(sid)
        okay0 = a['D_trace'] == 0 and a['complete_trace']
        complete.append(okay0)
        if not okay0 and a['D_pos'] != 1:
            out['unresolved_obligations'].append({'id': sid, 'reason': 'NO_COMPLETE_CONDITIONAL_NEGATIVE'})
    if out['positive_obligations']:
        out.update(D_pos=1, D_trace=1)
        out['reasons'] = ['BOUND_TASK_POTENTIAL_DOMINATES_UNKNOWN']
    elif inventory_complete and all(complete):
        out['D_trace'] = 0
        out['reasons'] = ['NONEMPTY_ALL_NECESSARY_ACCESSES_COMPLETED_CONDITIONAL_TRACE_ZERO']
    else:
        out['reasons'] = ['NECESSARY_ACCESS_OR_CONTEXT_INCOMPLETE' if not inventory_complete else 'UNRESOLVED_NECESSARY_OBLIGATION']
    return out


def enumerate_scene(source, registration):
    # Parser imports stay in isolated workers. Pure aggregator needs only stdlib.
    from query_access_linkage import (Parser, Language, tree_sitter_c, walk, text,
                                     span, context, role, function_name, MEMORY_NAMES, CALL_ROLES, link_query)
    raw = source.encode('utf-8')
    digest = hashlib.sha256(raw).hexdigest()
    sid = registration['unit_id']; entry = registration.get('entry', 'main')
    out = {'schema': SCHEMA, 'unit_id': sid, 'source_sha256': digest, 'entry': entry,
           'obligations': [], 'excluded_syntax': [], 'context_blockers': [],
           'inventory_complete': False, 'classification': UNKNOWN,
           'scope': 'original sanitized source, entry and syntactically necessary called helpers; all source accesses, not model slice subset',
           'labels_or_scores_used': False, 'full_C_program_safety': UNKNOWN}
    if digest != registration['source_sha256']:
        out['context_blockers'] = ['SOURCE_HASH_MISMATCH']
        return out
    tree = Parser(Language(tree_sitter_c.language())).parse(raw)
    out['whole_translation_unit_has_diagnostics'] = tree.root_node.has_error
    functions = collections.defaultdict(list)
    for f in tree.root_node.named_children:
        if f.type == 'function_definition':
            functions[function_name(raw, f)].append(f)
    if len(functions.get(entry, [])) != 1:
        out['context_blockers'] = ['ENTRY_NOT_UNIQUE']
        return out
    necessary = {entry}; pending = [entry]
    # These are only context names; actual trust remains source-specific factory
    # evidence in the unchanged worker. No name alone grants a negative.
    core_context_names = {'malloc', 'free', 'assert'}
    while pending:
        name = pending.pop()
        if len(functions[name]) != 1:
            out['context_blockers'].append('CALLED_FUNCTION_NOT_UNIQUE:' + name)
            continue
        fn = functions[name][0]
        if fn.has_error:
            out['context_blockers'].append('NECESSARY_FUNCTION_AST_ERROR:' + name)
        for n in walk(fn.child_by_field_name('body')):
            if n.type != 'call_expression':
                continue
            callee = n.child_by_field_name('function')
            cname = text(raw, callee)
            if callee is None or callee.type != 'identifier':
                out['context_blockers'].append('INDIRECT_OR_UNRESOLVED_CALL')
            elif cname in functions:
                if cname not in necessary:
                    necessary.add(cname); pending.append(cname)
                out['context_blockers'].append('NECESSARY_NONENTRY_CALL:' + cname)
            elif cname not in core_context_names:
                out['context_blockers'].append('UNSUPPORTED_CONTEXT_CALL:' + cname)
    for n in walk(tree.root_node):
        if n.type in ('preproc_def', 'preproc_function_def', 'preproc_if', 'preproc_ifdef', 'preproc_elif'):
            out['context_blockers'].append('SOURCE_MACRO_OR_CONDITIONAL_PREPROCESSING')
    for fname in sorted(necessary):
        if len(functions[fname]) != 1:
            continue
        fn = functions[fname][0]
        for n in walk(fn.child_by_field_name('body')):
            kind = None; roles = []
            if n.type == 'subscript_expression':
                kind = 'array_index'; roles = [role(n, raw)]
            elif n.type == 'pointer_expression' and text(raw, n.child_by_field_name('operator')) == '*':
                kind = 'pointer_dereference'; roles = [role(n, raw)]
            elif n.type == 'field_expression':
                kind = 'field_access'; roles = [role(n, raw)]
            elif n.type == 'call_expression' and text(raw, n.child_by_field_name('function')) in MEMORY_NAMES:
                kind = 'memory_call'; roles = CALL_ROLES.get(text(raw, n.child_by_field_name('function')), ['unresolved_memory_call_role'])
            if kind is None:
                continue
            s = span(n); excluded = context(n, raw)
            if excluded:
                out['excluded_syntax'].append({'source_span': s, 'kind': kind, 'reasons': excluded})
                continue
            for access_role in roles:
                oid = sid + ':access:' + str(s['start_byte']) + ':' + str(s['end_byte']) + ':' + access_role
                # A separate task obligation preserves original source coordinates;
                # same-line ambiguity is rejected by the unchanged linker.
                q = {'scene_id': oid, 'source_case': sid, 'query_line': s['start_line'],
                     'row': len(out['obligations']), 'family_id': registration['group']}
                link = link_query(source, q, {'source_sha256': digest, 'entry': entry})
                unique = link.get('unique_access')
                bound = bool(fname == entry and link['linkage_status'] == 'DIRECT_SYNTAX_UNIQUE'
                             and unique and unique['source_span'] == s
                             and kind in ('array_index', 'pointer_dereference') and len(roles) == 1)
                out['obligations'].append({'id': oid, 'unit_id': oid, 'scene_id': sid,
                    'entry': entry, 'function': fname, 'source_sha256': digest,
                    'source_span': s, 'source_text': text(raw, n), 'kind': kind, 'role': access_role,
                    'query_registration': q, 'identity_valid': bound, 'task_scope_valid': bound,
                    'source_access_id': link.get('source_access_id') if bound else None,
                    'linkage_status': link['linkage_status'], 'linkage_reasons': link['reasons'],
                    'original_model_slice_required': False})
    out['context_blockers'] = sorted(set(out['context_blockers']))
    out['necessary_functions'] = sorted(necessary)
    out['inventory_complete'] = bool(out['obligations'] and not out['context_blockers']
        and all(a['identity_valid'] for a in out['obligations']))
    return out
