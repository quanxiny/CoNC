"""Deterministic sidecar integers. Never reads or changes learned Q/15 fields."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = 'evidence_numeric_v1'


@dataclass(frozen=True)
class CType:
    bits: int
    signed: bool

    @property
    def supported(self):
        return self.bits in (8, 16, 32)

    @property
    def bounds(self):
        return (-(1 << (self.bits-1)), (1 << (self.bits-1))-1) if self.signed else (0, (1 << self.bits)-1)


def result(value=None, ctype=None, reason=None):
    return {'value': value, 'type': asdict(ctype) if ctype else None,
            'status': 'KNOWN' if value is not None else 'UNKNOWN', 'reason': reason}


def math_op(op, a, b):
    if a is None or b is None:
        return None
    return a+b if op == 'ADD' else a-b if op == 'SUB' else None


def bitvector(op, a, b, width):
    if width not in (8, 16, 32) or a is None or b is None or op not in ('ADD', 'SUB'):
        return result(reason='unsupported_bitvector_or_unknown')
    if not (0 <= a < 1 << width and 0 <= b < 1 << width):
        return result(reason='operand_out_of_range')
    return result(math_op(op, a, b) & ((1 << width)-1), CType(width, False))


def convert(value, source, target):
    if value is None or source is None or target is None or not source.supported or not target.supported:
        return result(reason='unknown_or_unsupported_conversion')
    if not source.bounds[0] <= value <= source.bounds[1]:
        return result(reason='source_out_of_range')
    if not target.signed:
        return result(value % (1 << target.bits), target)
    if not target.bounds[0] <= value <= target.bounds[1]:
        return result(reason='implementation_defined_signed_conversion')
    return result(value, target)


def typed_op(op, a, b, ta, tb, int_bits=32, promote=True):
    if a is None or b is None or ta is None or tb is None or not ta.supported or not tb.supported:
        return result(reason='unknown_operand_or_type')
    if op not in ('ADD', 'SUB') or int_bits != 32:
        return result(reason='unsupported_operation_or_int_abi')
    if not ta.bounds[0] <= a <= ta.bounds[1] or not tb.bounds[0] <= b <= tb.bounds[1]:
        return result(reason='operand_out_of_range')
    pa = CType(32, True) if promote and ta.bits < 32 else ta
    pb = CType(32, True) if promote and tb.bits < 32 else tb
    if pa.signed == pb.signed:
        common = CType(max(pa.bits, pb.bits), pa.signed)
    else:
        u, s = (pa, pb) if not pa.signed else (pb, pa)
        common = u if u.bits >= s.bits else s if s.bounds[1] >= u.bounds[1] else CType(s.bits, False)
    aa = convert(a, ta, common)['value']; bb = convert(b, tb, common)['value']
    val = math_op(op, aa, bb)
    if val is None:
        return result(reason='conversion_unknown')
    if common.signed and not common.bounds[0] <= val <= common.bounds[1]:
        return result(reason='signed_overflow')
    return result(val if common.signed else val % (1 << common.bits), common)


def check_bounds(offset, width, capacity, *, path_known=True, type_known=True,
                 units_match=True, lifetime_known=True, formation=False):
    facts = {'path_known': path_known, 'type_known': type_known,
             'units_match': units_match, 'lifetime_known': lifetime_known}
    missing = [k for k, v in facts.items() if not v]
    if missing or any(v is None for v in (offset, width, capacity)):
        return {'relation': 'UNKNOWN', 'reasons': missing or ['numeric_fact_unknown']}
    if capacity < 0 or width < 0:
        return {'relation': 'UNKNOWN', 'reasons': ['invalid_extent']}
    valid = 0 <= offset <= capacity and (formation or offset + width <= capacity)
    return {'relation': 'BOUND_SATISFIED' if valid else 'BOUND_VIOLATED', 'reasons': []}


def probe_abi():
    """Preprocess /dev/null only; never compiles/runs dataset C."""
    compiler = Path('/usr/bin/gcc').resolve()
    raw = subprocess.check_output([str(compiler), '-dM', '-E', '-x', 'c', '/dev/null'], text=True)
    macros = {}
    for line in raw.splitlines():
        _, key, value = line.split(' ', 2)
        macros[key] = value
    selected = {k: macros.get(k) for k in ('__CHAR_BIT__', '__SIZEOF_SHORT__', '__SIZEOF_INT__',
                '__SIZEOF_LONG__', '__SIZEOF_LONG_LONG__', '__SIZEOF_POINTER__', '__CHAR_UNSIGNED__', '__x86_64__')}
    target = subprocess.check_output([str(compiler), '-dumpmachine'], text=True).strip()
    valid = target.startswith('x86_64') and selected['__CHAR_BIT__'] == '8' and selected['__SIZEOF_INT__'] == '4' and selected['__SIZEOF_POINTER__'] == '8'
    body = {'schema': SCHEMA, 'compiler': str(compiler), 'compiler_sha256': hashlib.sha256(compiler.read_bytes()).hexdigest(),
            'target': target, 'selected_macros': selected, 'predefined_macros_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'verified_x86_64': valid, 'scope': 'GCC actual predefined ABI; caller must bind analysis to this explicit ABI, not infer Frama-C compatibility',
            'dataset_program_execution': False}
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body
