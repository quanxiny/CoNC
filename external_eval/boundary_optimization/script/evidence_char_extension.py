"""Small character-literal extension of the frozen finite Analyzer.

Encoding/target values require an independently generated compiler certificate.
No model, old field, library policy, guard, numeric, or alias semantics changed.
"""
import hashlib
import json
import tree_sitter
import tree_sitter_c
from evidence_access import Analyzer, walk
from evidence_numeric import CType

VERSION = 'verified_character_extension_v1'
BASIC = {'a':7, 'b':8, 'f':12, 'n':10, 'r':13, 't':9, 'v':11,
         '\\':92, "'":39, '"':34, '?':63}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def dependency_affected(source, entry='main'):
    """False is an exact no-override dependency proof, not a lexical heuristic.

    Conservatively includes literals outside entry and in unevaluated operands.
    """
    raw = source.encode()
    parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_c.language()))
    nodes = [n for n in walk(parser.parse(raw).root_node) if n.type == 'char_literal']
    return {'affected':bool(nodes), 'reason':'character_literal_AST_present' if nodes else 'no_character_literal_AST_override_can_execute',
            'character_count':len(nodes), 'character_spans':[{'start_byte':n.start_byte,'end_byte':n.end_byte,
                'start_line':n.start_point.row+1,'start_column':n.start_point.column+1} for n in nodes],
            'contract_version':VERSION, 'entry':entry,
            'unaffected_equivalence':'all inherited fact/decision fields identical; extension metadata only'}


class CharacterAnalyzer(Analyzer):
    def __init__(self, source, uid, path, abi, entry, models, config=None):
        super().__init__(source, uid, path, abi, entry, models)
        self.config = config or {}; self.contract = self.config.get('character_contract') or {}
        self.character_facts = {}
        body = {k:v for k,v in self.contract.items() if k != 'sha256'}
        self.character_contract_ok = self.abi_ok and self.contract.get('sha256') == digest(body)
        self.character_contract_ok &= self.contract.get('status') == 'PASS'
        self.character_contract_ok &= self.contract.get('ABI_sha256') == self.abi.get('sha256')
        self.character_contract_ok &= self.contract.get('compiler_sha256') == self.abi.get('compiler_sha256')
        self.character_contract_ok &= self.contract.get('input_encoding') == 'UTF-8' and self.contract.get('execution_encoding') == 'UTF-8'
        self.character_contract_ok &= self.contract.get('standard') == 'c11' and self.contract.get('ordinary_literal_type') == {'bits':32,'signed':True}

    def character(self, n):
        raw = self.text(n); reason = None; byte = None; form = None
        if not self.character_contract_ok:
            reason = 'CHARACTER_ENCODING_OR_COMPILER_CONTRACT_NOT_VERIFIED'
        elif not raw.startswith("'") or not raw.endswith("'"):
            reason = 'PREFIXED_CHARACTER_UNSUPPORTED'
        else:
            text = raw[1:-1]
            if len(text) == 1 and 32 <= ord(text) <= 126 and text not in ("'", '\\'):
                byte = ord(text); form = 'ordinary_ascii'
            elif len(text) == 2 and text[0] == '\\' and text[1] in BASIC:
                byte = BASIC[text[1]]; form = 'basic_escape'
            elif text.startswith('\\') and 1 <= len(text[1:]) <= 3 and all(c in '01234567' for c in text[1:]):
                byte = int(text[1:], 8); form = 'octal_escape'
            elif text.startswith('\\x') and len(text) > 2 and all(c in '0123456789abcdefABCDEF' for c in text[2:]):
                byte = int(text[2:], 16); form = 'hex_escape'
            else:
                reason = 'MULTICHAR_NONASCII_OR_NONSTANDARD_ESCAPE_UNSUPPORTED'
            if byte is not None and not 0 <= byte <= 255:
                reason = 'NUMERIC_ESCAPE_OUTSIDE_VERIFIED_BYTE_DOMAIN'
        value = None
        if reason is None:
            table = self.contract.get('verified_byte_values', [])
            if len(table) != 256 or not isinstance(table[byte], int):
                reason = 'VERIFIED_CHARACTER_VALUE_MISSING'
            else:
                value = table[byte]
                if not -(1<<31) <= value < (1<<31):
                    reason = 'CHARACTER_INT_VALUE_OUTSIDE_CONTRACT'; value = None
        self.character_facts[(n.start_byte,n.end_byte)] = {
            'span':self.span(n),'literal':raw,'form':form,'byte_value':byte,'int_value':value,
            'C_type':{'bits':32,'signed':True} if value is not None else None,
            'status':'KNOWN' if value is not None else 'UNKNOWN','reason':reason,
            'encoding_contract_sha256':self.contract.get('sha256'),
            'provenance':'independent compiler static_assert certificate; no C execution'}
        return value, reason

    def number(self, node):
        n = self.unwrap(node)
        if n is not None and n.type == 'char_literal':
            value, reason = self.character(n)
            if reason:
                self.blocked.append(reason)
            return value, CType(32,True) if value is not None else None
        return super().number(node)

    def expression(self, n):
        if n is not None and n.type == 'char_literal':
            _, reason = self.character(n)
            if reason:
                self.blocked.append(reason)
            # A supported literal is a single int-valued atom; its lexical child
            # is not a separate C expression. Unsupported literals remain U.
            return
        return super().expression(n)

    def run(self):
        r = super().run()
        r['character_extension'] = {'version':VERSION, 'dependency':dependency_affected(self.src.decode(),self.entry),
            'contract_sha256':self.contract.get('sha256'), 'contract_verified':bool(self.character_contract_ok),
            'facts':[self.character_facts[k] for k in sorted(self.character_facts)],
            'scope':'ordinary ASCII/basic/octal/hex literals under verified compiler/encoding contract; inherited contexts/UNKNOWN unchanged',
            'old_schema_and_Q_changed':False}
        return r


def analyze_source(source, unit_id, source_path='', abi=None, entry='main', memory_models=None, config=None):
    return CharacterAnalyzer(source,unit_id,source_path,abi,entry,memory_models,config).run()
