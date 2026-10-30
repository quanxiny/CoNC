"""Finite source-fact sidecar over installed tree-sitter C syntax AST.

No regex alias inference, source execution, model calls, or original-field writes.
Only the specified entry's straight-line fixed objects are interpreted as facts.
"""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tree_sitter
import tree_sitter_c
from evidence_numeric import CType, typed_op, convert, check_bounds

SCHEMA = 'evidence_access_v1'
MEMORY = {'memset': [(0, 'copy_destination')],
          'memcpy': [(0, 'copy_destination'), (1, 'copy_source')],
          'memmove': [(0, 'copy_destination'), (1, 'copy_source')]}


def walk(node):
    yield node
    for c in node.named_children:
        yield from walk(c)


def analyze_source(source, unit_id, source_path='', abi=None, entry='main', memory_models=None):
    return Analyzer(source, unit_id, source_path, abi, entry, memory_models).run()


class Analyzer:
    def __init__(self, source, uid, path, abi, entry, models):
        self.src = source.encode(); self.uid = uid; self.path = path
        self.abi = abi or {}; self.entry = entry; self.models = models or {}
        self.parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_c.language()))
        self.tree = self.parser.parse(self.src); self.env = {}; self.slots = {}; self.records = {}
        self.flags = []; self.blocked = []; self.objects = {}; self.declared = set()
        self.mac = self.abi.get('selected_macros', {})
        self.abi_ok = bool(self.abi.get('verified_x86_64')) and self.mac.get('__CHAR_BIT__') == '8' and self.mac.get('__SIZEOF_INT__') == '4'
        abi_body = {k:v for k,v in self.abi.items() if k != 'sha256'}
        self.abi_ok &= self.abi.get('sha256') == hashlib.sha256(json.dumps(abi_body, sort_keys=True).encode()).hexdigest()

    def text(self, n):
        return self.src[n.start_byte:n.end_byte].decode() if n else ''

    def field(self, n, k):
        return n.child_by_field_name(k) if n else None

    def span(self, n):
        return {'start_byte': n.start_byte, 'end_byte': n.end_byte,
                'start_line': n.start_point.row+1, 'start_column': n.start_point.column+1,
                'end_line': n.end_point.row+1, 'end_column': n.end_point.column+1}

    def unwrap(self, n):
        while n is not None and n.type == 'parenthesized_expression':
            n = n.named_children[0]
        return n

    def size(self, typ):
        if not typ or not self.abi_ok:
            return None
        if typ['kind'] == 'array':
            s = self.size(typ['element']); count = typ['count']
            return s*count if s is not None and count is not None and count >= 0 else None
        if typ['kind'] == 'pointer':
            return int(self.mac['__SIZEOF_POINTER__'])
        return typ.get('bytes')

    def base_type(self, node):
        name = self.text(node)
        words = name.split()
        if not words or any(w not in ('char', 'short', 'int', 'long', 'signed', 'unsigned', 'void') for w in words):
            return {'kind': 'unknown', 'source_type': name}
        if 'void' in words:
            return {'kind': 'unknown', 'source_type': name}
        macro = '__SIZEOF_LONG_LONG__' if words.count('long') == 2 else '__SIZEOF_LONG__' if 'long' in words else '__SIZEOF_SHORT__' if 'short' in words else None if 'char' in words else '__SIZEOF_INT__'
        width = 1 if 'char' in words else int(self.mac[macro]) if self.abi_ok and self.mac.get(macro) else None
        signed = False if 'unsigned' in words else (self.mac.get('__CHAR_UNSIGNED__') is None if name == 'char' else True)
        return {'kind': 'integer', 'source_type': name, 'bytes': width, 'bits': width*8 if width else None, 'signed': signed}

    def declarator(self, n, base):
        if n is None:
            return None, {'kind': 'unknown'}
        if n.type == 'identifier':
            return self.text(n), base
        if n.type == 'pointer_declarator':
            return self.declarator(self.field(n, 'declarator'), {'kind': 'pointer', 'element': base})
        if n.type == 'array_declarator':
            dim = self.field(n, 'size')
            val = self.number(dim)[0] if dim is not None and not any(c.type == 'identifier' for c in walk(dim)) else None
            return self.declarator(self.field(n, 'declarator'), {'kind': 'array', 'element': base, 'count': val})
        return None, {'kind': 'unknown'}

    def ct(self, typ):
        return CType(typ['bits'], typ['signed']) if typ and typ.get('kind') == 'integer' and typ.get('bits') else None

    def number(self, node):
        n = self.unwrap(node)
        if n is None:
            return None, None
        if n.type == 'number_literal':
            raw = self.text(n); suffix = ''
            while raw and raw[-1] in 'uUlL':
                suffix = raw[-1] + suffix; raw = raw[:-1]
            if 'l' in suffix.lower():
                return None, None
            try:
                value = int(raw, 16 if raw.lower().startswith('0x') else 2 if raw.lower().startswith('0b') else 8 if len(raw)>1 and raw.startswith('0') else 10)
            except ValueError:
                return None, None
            typ = CType(32, 'u' not in suffix.lower())
            return (value, typ) if typ.bounds[0] <= value <= typ.bounds[1] else (None, None)
        if n.type == 'identifier':
            v = self.env.get(self.text(n), {})
            if v.get('type', {}).get('kind') == 'integer' and v.get('value') is None:
                self.blocked.append('UNVERIFIED_SCALAR_READ')
            return v.get('value'), self.ct(v.get('type'))
        if n.type == 'sizeof_expression':
            arg = self.unwrap(self.field(n, 'value'))
            if arg is not None and arg.type == 'identifier':
                typ = self.env.get(self.text(arg), {}).get('type')
                return self.size(typ), CType(64, False) if self.abi_ok else None
            return None, None
        if n.type == 'binary_expression':
            a, ta = self.number(self.field(n, 'left')); b, tb = self.number(self.field(n, 'right'))
            op = {'+': 'ADD', '-': 'SUB'}.get(self.text(self.field(n, 'operator')))
            r = typed_op(op, a, b, ta, tb)
            if r['reason'] == 'signed_overflow':
                self.blocked.append('PRECEDING_SIGNED_OVERFLOW')
            return r['value'], CType(**r['type']) if r['type'] else None
        if n.type == 'unary_expression' and self.text(self.field(n, 'operator')) == '-':
            a, ta = self.number(self.field(n, 'argument'))
            r = typed_op('SUB', 0, a, CType(32, True), ta)
            if r['reason'] == 'signed_overflow':
                self.blocked.append('PRECEDING_SIGNED_OVERFLOW')
            return r['value'], CType(**r['type']) if r['type'] else None
        return None, None

    def pointer(self, node):
        n = self.unwrap(node)
        if n is None:
            return None
        if n.type == 'identifier':
            return copy.deepcopy(self.env.get(self.text(n), {}).get('pointer'))
        if n.type == 'pointer_expression' and self.text(self.field(n, 'operator')) == '&':
            target = self.unwrap(self.field(n, 'argument'))
            if target.type == 'subscript_expression':
                return self.location(target)
            if target.type == 'identifier':
                v = self.env.get(self.text(target), {})
                return copy.deepcopy(v.get('storage_pointer'))
        if n.type == 'binary_expression' and self.text(self.field(n, 'operator')) in ('+', '-'):
            p = self.pointer(self.field(n, 'left')); idx, _ = self.number(self.field(n, 'right'))
            if p and idx is not None and self.size(p['pointee']) is not None:
                p['offset'] += idx*self.size(p['pointee'])*(1 if self.text(self.field(n, 'operator')) == '+' else -1)
                if check_bounds(p['offset'], 0, p['capacity'], formation=True)['relation'] != 'BOUND_SATISFIED':
                    self.blocked.append('POINTER_ARITHMETIC_FORMATION_UNVERIFIED')
                    return None
                return p
        if n.type == 'subscript_expression':
            p = self.location(n)
            if p and p['pointee']['kind'] == 'array':
                if check_bounds(p['offset'], self.size(p['pointee']), p['capacity'])['relation'] != 'BOUND_SATISFIED':
                    return None
                # Inner row is its own subobject; prevent crossing row bounds.
                p['object_id'] += ':subobject@' + str(p['offset'])
                p['capacity'] = self.size(p['pointee']); p['offset'] = 0
                p['pointee'] = p['pointee']['element']
                return p
            if p and p['pointee']['kind'] == 'pointer':
                return copy.deepcopy(self.slots.get((p['object_id'], p['offset'])))
        return None

    def location(self, n):
        p = self.pointer(self.field(n, 'argument')); idx, _ = self.number(self.field(n, 'index'))
        if p and idx is not None and self.size(p['pointee']) is not None:
            p['offset'] += idx*self.size(p['pointee'])
            return p
        return None

    def formation(self, n):
        cur = n
        while cur.parent and cur.parent.type == 'parenthesized_expression':
            cur = cur.parent
        return cur.parent is not None and cur.parent.type == 'pointer_expression' and self.text(self.field(cur.parent, 'operator')) == '&'

    def unevaluated(self, n):
        p = n.parent
        while p:
            if p.type == 'sizeof_expression':
                return True
            p = p.parent
        return False

    def access_mode(self, n, formation=False):
        if formation or self.unevaluated(n):
            return 'no_value_read'
        cur = n
        while cur.parent and cur.parent.type == 'parenthesized_expression':
            cur = cur.parent
        if cur.parent and cur.parent.type == 'assignment_expression' and self.field(cur.parent, 'left') == cur:
            return 'write'
        return 'read'

    def record(self, n, p, role='index', extra=None, width=None, formation=False):
        key = (n.start_byte, n.end_byte, role)
        if key in self.records:
            return
        missing = list(self.blocked)
        if not self.abi_ok:
            missing.append('ABI_NOT_VERIFIED')
        if p is None:
            missing.append('OBJECT_OR_INDEX_BINDING_UNKNOWN')
        if width is None and p:
            width = self.size(p['pointee'])
        if extra:
            missing.extend(extra)
        if self.unevaluated(n):
            relation = 'UNEVALUATED'; missing = ['sizeof_unevaluated_no_access']
        else:
            relation = check_bounds(p['offset'] if p else None, 0 if formation else width,
                                    p['capacity'] if p else None, path_known=not missing,
                                    type_known=bool(p and width is not None), formation=formation)['relation']
        self.records[key] = {'access_id': self.uid+':'+str(n.start_byte)+':'+str(n.end_byte)+':'+role,
            'span': self.span(n), 'source_text': self.text(n), 'role': 'pointer_formation' if formation else role,
            'object_id': p['object_id'] if p else None, 'offset_bytes': p['offset'] if p else None,
            'width_bytes': 0 if formation else width, 'capacity_bytes': p['capacity'] if p else None,
            'unit': 'bytes', 'type': p['pointee'] if p else None,
            'provenance': p['provenance'] if p else [], 'relation': relation,
            'reasons': sorted(set(missing or ([] if relation != 'UNKNOWN' else ['NUMERIC_OR_TYPE_FACT_UNKNOWN']))),
            'access_mode':self.access_mode(n, formation),
            'path': 'straight_line_entry' if not self.blocked else 'UNKNOWN',
            'reachability': 'NOT_CERTIFIED', 'full_C_program_safety': 'UNKNOWN'}

    def compatible_pointer(self, target_type, value):
        p = self.pointer(value)
        if not p or target_type.get('kind') != 'pointer' or target_type['element'] != p['pointee']:
            return None
        return p

    def expression(self, n):
        if n is None:
            return
        if n.type in ('conditional_expression', 'binary_expression'):
            op = self.text(self.field(n, 'operator'))
            if n.type == 'conditional_expression' or op in ('&&', '||'):
                self.blocked.append('CONDITIONAL_EXPRESSION_EVALUATION_UNKNOWN')
            elif op not in ('+', '-', '<', '<=', '>', '>=', '==', '!='):
                self.blocked.append('UNSUPPORTED_OPERATOR_PRECONDITIONS:'+op)
            elif op in ('+', '-'):
                # Establish overflow/operand-read preconditions even if the value
                # is discarded rather than used in a subsequent index.
                if self.pointer(self.field(n, 'left')) is not None:
                    if self.pointer(n) is None:
                        self.blocked.append('POINTER_ARITHMETIC_PRECONDITIONS_UNKNOWN')
                elif self.number(n)[0] is None:
                    self.blocked.append('ARITHMETIC_TYPE_OR_VALUE_PRECONDITIONS_UNKNOWN')
        if n.type == 'number_literal' and self.number(n)[0] is None:
            self.blocked.append('LITERAL_TYPE_OR_VALUE_UNSUPPORTED')
        if n.type == 'identifier':
            cur = n
            while cur.parent and cur.parent.type == 'parenthesized_expression':
                cur = cur.parent
            lhs = cur.parent and cur.parent.type == 'assignment_expression' and self.field(cur.parent, 'left') == cur
            addr = cur.parent and cur.parent.type == 'pointer_expression' and self.text(self.field(cur.parent, 'operator')) == '&'
            v = self.env.get(self.text(n), {})
            if not lhs and not addr:
                if self.text(n) not in self.env:
                    self.blocked.append('IDENTIFIER_FACTS_NOT_IN_LOCAL_ENVIRONMENT')
                elif v.get('type', {}).get('kind') == 'integer' and v.get('value') is None:
                    self.blocked.append('UNVERIFIED_SCALAR_READ')
                elif v.get('type', {}).get('kind') == 'pointer' and not v.get('pointer'):
                    self.blocked.append('UNVERIFIED_POINTER_READ')
            return
        if n.type == 'sizeof_expression':
            # sizeof operands are inventoried, never applied as assignments/calls.
            for c in walk(n):
                if c.type == 'subscript_expression':
                    self.record(c, None, formation=self.formation(c))
                elif c.type == 'pointer_expression' and self.text(self.field(c, 'operator')) == '*':
                    self.record(c, None, role='dereference', formation=self.formation(c))
                elif c.type == 'field_expression':
                    self.record(c, None, role='member_access')
            if self.field(n, 'type') is not None:
                self.blocked.append('SIZEOF_TYPE_OR_VLA_UNSUPPORTED')
            return
        if n.type == 'assignment_expression':
            if n.parent and n.parent.type != 'expression_statement':
                self.blocked.append('NESTED_ASSIGNMENT_SEQUENCE_UNSUPPORTED')
            lhs = self.field(n, 'left'); rhs = self.field(n, 'right')
            self.expression(lhs); self.expression(rhs)
            if self.text(self.field(n, 'operator')) != '=':
                self.blocked.append('COMPOUND_ASSIGNMENT_UNSUPPORTED'); return
            if lhs.type == 'identifier':
                name = self.text(lhs); old = self.env.get(name, {})
                if name not in self.env:
                    self.blocked.append('ASSIGNMENT_TARGET_NOT_IN_LOCAL_ENVIRONMENT')
                if old.get('type', {}).get('kind') == 'pointer':
                    old['pointer'] = self.compatible_pointer(old['type'], rhs)
                else:
                    val, typ = self.number(rhs); target = self.ct(old.get('type'))
                    old['value'] = convert(val, typ, target)['value']
                self.env[name] = old
            elif lhs.type == 'subscript_expression':
                loc = self.location(lhs)
                if loc and loc['pointee']['kind'] == 'pointer':
                    within = check_bounds(loc['offset'], self.size(loc['pointee']), loc['capacity'])['relation']
                    if within == 'BOUND_SATISFIED' and not self.blocked:
                        self.slots[(loc['object_id'], loc['offset'])] = self.compatible_pointer(loc['pointee'], rhs)
            return
        if n.type == 'subscript_expression':
            for c in n.named_children:
                self.expression(c)
            loc = self.location(n)
            form = self.formation(n)
            self.record(n, loc, formation=form)
            if loc and self.access_mode(n, form) == 'read':
                typ = loc['pointee']
                if typ['kind'] != 'array' and not (typ['kind'] == 'pointer' and (loc['object_id'],loc['offset']) in self.slots):
                    # Bounds are a separate conditional fact; content-read preconditions
                    # are not silently discharged by a known offset/capacity.
                    self.blocked.append('VALUE_READ_INITIALIZATION_NOT_VERIFIED')
            return
        if n.type == 'pointer_expression' and self.text(self.field(n, 'operator')) == '*':
            self.expression(self.field(n, 'argument'))
            self.record(n, self.pointer(self.field(n, 'argument')), role='dereference', formation=self.formation(n))
            if self.access_mode(n, self.formation(n)) == 'read':
                self.blocked.append('VALUE_READ_INITIALIZATION_NOT_VERIFIED')
            return
        if n.type == 'field_expression':
            for c in n.named_children:
                self.expression(c)
            self.record(n, None, role='member_access', extra=['STRUCT_MEMBER_TYPE_UNSUPPORTED'])
            return
        if n.type == 'call_expression':
            fn = self.text(self.field(n, 'function')); args = self.field(n, 'arguments').named_children
            for c in args:
                self.expression(c)
            if fn in MEMORY:
                permitted = fn in self.models and fn not in self.declared
                count, _ = self.number(args[2]) if len(args) == 3 else (None, None)
                for pos, role in MEMORY[fn]:
                    self.record(n, self.pointer(args[pos]) if len(args)>pos else None, role,
                                [] if permitted else ['LIBRARY_CONTRACT_NOT_BOUND'], width=count)
                dest = self.pointer(args[0]) if args else None
                if dest:
                    self.slots = {key:val for key,val in self.slots.items() if key[0] != dest['object_id']}
                if not permitted:
                    self.blocked.append('UNBOUND_LIBRARY_EFFECTS')
            else:
                self.blocked.append('UNMODELED_CALL_EFFECTS:'+fn)
                self.env = {k: {**v, 'value': None, 'pointer': None} for k, v in self.env.items()}
                self.slots.clear()
            return
        if n.type == 'update_expression':
            self.blocked.append('UPDATE_EFFECT_UNSUPPORTED')
            self.env = {k: {**v, 'value': None} for k, v in self.env.items()}
        allowed = {'number_literal','string_literal','string_content','escape_sequence','char_literal',
                   'parenthesized_expression','binary_expression','unary_expression','pointer_expression',
                   'expression_statement','return_statement','argument_list','concatenated_string'}
        if n.type not in allowed:
            self.blocked.append('EXPRESSION_SEMANTICS_UNSUPPORTED:'+n.type)
        if n.type == 'unary_expression':
            op = self.text(self.field(n, 'operator'))
            if op != '-':
                self.blocked.append('UNARY_OPERATOR_UNSUPPORTED:'+op)
            else:
                self.number(n)
        for c in n.named_children:
            self.expression(c)
        if n.type == 'pointer_expression' and self.text(self.field(n, 'operator')) == '&':
            if any(a['role'] == 'pointer_formation' and a['relation'] == 'BOUND_VIOLATED'
                   and n.start_byte <= a['span']['start_byte'] < n.end_byte for a in self.records.values()):
                self.blocked.append('PRECEDING_INVALID_POINTER_FORMATION')

    def declaration(self, node):
        if any(c.type == 'type_qualifier' for c in walk(node)):
            self.blocked.append('QUALIFIED_DECLARATION_UNSUPPORTED')
        if any(c.type == 'storage_class_specifier' and self.text(c) == 'extern' for c in node.named_children):
            self.blocked.append('EXTERN_STORAGE_UNVERIFIED')
        base = self.base_type(self.field(node, 'type'))
        for i, n in enumerate(node.children):
            if node.field_name_for_child(i) != 'declarator':
                continue
            init = self.field(n, 'value') if n.type == 'init_declarator' else None
            d = self.field(n, 'declarator') if n.type == 'init_declarator' else n
            name, typ = self.declarator(d, base)
            if not name:
                self.blocked.append('DECLARATION_TYPE_UNSUPPORTED'); continue
            if name in self.env:
                self.blocked.append('SHADOW_OR_REDECLARATION_UNSUPPORTED')
            self.declared.add(name)
            item = {'type': typ, 'value': None, 'pointer': None}
            self.expression(init)
            if typ['kind'] == 'array':
                oid = self.uid+':object:'+str(d.start_byte)
                p = {'object_id': oid, 'offset': 0, 'capacity': self.size(typ), 'pointee': typ['element'],
                     'provenance': [{'kind': 'fixed_local_array_declaration', 'span': self.span(node),
                                     'ABI_sha256': self.abi.get('sha256')}]}
                item['pointer'] = p; item['storage_pointer'] = {**p, 'pointee': typ}
                self.objects[oid] = {'type': typ, 'span': self.span(node), 'capacity_bytes': p['capacity']}
                if init is not None:
                    self.blocked.append('ARRAY_INITIALIZER_CONTENTS_UNSUPPORTED')
            elif typ['kind'] == 'pointer':
                item['pointer'] = self.compatible_pointer(typ, init)
            else:
                value, ct = self.number(init)
                item['value'] = convert(value, ct, self.ct(typ))['value']
            self.env[name] = item

    def statements(self, node, depth=0):
        if node.type == 'compound_statement':
            if depth > 0:
                self.blocked.append('NESTED_SCOPE_LIFETIME_UNSUPPORTED')
            for child in node.named_children:
                self.statements(child, depth+1)
        elif node.type == 'declaration':
            self.declaration(node)
        elif node.type in ('expression_statement', 'return_statement'):
            self.expression(node)
            if node.type == 'return_statement':
                self.blocked.append('AFTER_RETURN_UNREACHABLE_UNVERIFIED')
        elif node.type in ('if_statement', 'for_statement', 'while_statement', 'do_statement', 'switch_statement', 'goto_statement', 'labeled_statement'):
            self.blocked.append('DYNAMIC_CONTROL_FLOW_UNSUPPORTED')
            # Inventory is total; facts under/after branches are deliberately not propagated.
            old = copy.deepcopy(self.env)
            for c in node.named_children:
                self.expression(c)
            self.env = {k: {**v, 'value': None, 'pointer': None} for k, v in old.items()}
            self.slots.clear()
        else:
            self.blocked.append('STATEMENT_UNSUPPORTED:'+node.type)
            self.expression(node)

    def run(self):
        root = self.tree.root_node
        if root.has_error:
            self.flags.append('SYNTAX_PARSE_ERROR')
        if any(n.type.startswith('preproc_') for n in walk(root)):
            self.flags.append('PREPROCESSOR_OR_INCLUDE_NOT_RESOLVED')
        defs = []
        for n in root.named_children:
            if n.type == 'function_definition':
                decl = self.field(n, 'declarator'); name = self.field(decl, 'declarator')
                if name and self.text(name) in MEMORY:
                    self.declared.add(self.text(name))
                if name and self.text(name) == self.entry:
                    defs.append(n)
        if len(defs) != 1:
            self.flags.append('ENTRY_NOT_UNIQUE_OR_MISSING')
        self.blocked.extend(self.flags)
        body = self.field(defs[0], 'body') if len(defs) == 1 else None
        if body:
            self.statements(body)
            # Fall back for every syntax access not handled by finite semantics.
            for n in walk(body):
                if n.type == 'subscript_expression':
                    self.record(n, None, formation=self.formation(n), extra=['UNHANDLED_ACCESS'])
                elif n.type == 'pointer_expression' and self.text(self.field(n, 'operator')) == '*':
                    self.record(n, None, role='dereference', formation=self.formation(n), extra=['UNHANDLED_ACCESS'])
                elif n.type == 'field_expression':
                    self.record(n, None, role='member_access', extra=['STRUCT_MEMBER_TYPE_UNSUPPORTED'])
        accesses = sorted(self.records.values(), key=lambda a: (a['span']['start_byte'], a['span']['end_byte'], a['role']))
        # Source order for nested expressions is inner-before-outer to match evaluated selections.
        accesses.sort(key=lambda a: (a['span']['end_byte'], -a['span']['start_byte'], a['role']))
        relations = [a for a in accesses if a['relation'] != 'UNEVALUATED']
        applicable = [a for a in relations if a['role'] != 'pointer_formation']
        complete = bool(body and applicable) and not self.flags and all(a['relation'] != 'UNKNOWN' for a in relations)
        # A return at the end does not invalidate facts already established before it.
        residual = [r for r in self.blocked if r != 'AFTER_RETURN_UNREACHABLE_UNVERIFIED']
        complete = complete and not residual and self.abi_ok
        decision = 1 if any(a['relation'] == 'BOUND_VIOLATED' for a in applicable) else 0 if complete else None
        return {'schema_version': SCHEMA, 'unit_id': self.uid, 'source_path': self.path,
            'source_sha256': hashlib.sha256(self.src).hexdigest(), 'entry': self.entry,
            'source_accesses': accesses, 'objects': self.objects,
            'coverage_certificate': {'complete': complete, 'entry': self.entry, 'scope': 'entry syntactic accesses; no interprocedural or full-C proof',
                'ABI_sha256': self.abi.get('sha256'), 'abi_hash':self.abi.get('sha256'),
                'source_sha256':hashlib.sha256(self.src).hexdigest(),
                'all_source_targets_enumerated':bool(body) and not self.flags and not any('CALL_EFFECTS' in r or 'STATEMENT_UNSUPPORTED' in r for r in residual),
                'all_targets_bound':bool(applicable) and all(a['object_id'] is not None for a in applicable),
                'related_preconditions_verified':complete,
                'enumeration': 'tree_sitter_c_entry_AST',
                'source_access_count': len(accesses), 'applicable_access_count': len(applicable),
                'unknown_reasons': sorted(set(self.flags+residual)),
                'support': 'finite straight-line fixed local arrays, unique aliases, known pointer slots; ADD/SUB integers <=32bits',
                'library_model_assumptions': self.models,
                'parser_version': importlib.metadata.version('tree-sitter-c'),
                'path_reachability': 'NOT_CERTIFIED'},
            'decision': decision, 'status': 'CONDITIONAL_TARGET_ALARM' if decision == 1 else 'CONDITIONAL_TARGETS_SATISFIED' if decision == 0 else 'UNKNOWN',
            'positive_applicable':decision == 1,
            'positive_applicability_scope':'conditional bounds violation at an entry-scope evaluated access, no preceding unsupported/known UB in finite scan; entry invocation and complete C-definedness remain assumptions, NOT a reachable counterexample',
            'conditional_relation_decision':1 if any(a['relation']=='BOUND_VIOLATED' for a in relations) else 0 if relations and all(a['relation']=='BOUND_SATISFIED' for a in relations) else None,
            'full_C_program_safety': 'UNKNOWN', 'dataset_program_execution': False}
