"""Frozen exact-integer evidence backend; explicitly NEW relative to v38 neural arithmetic.

No C execution, signedness inference or program proof. Only already-public finite
operand values and explicitly matching units are compared. Unknown stays unknown.
"""
from dataclasses import dataclass
from .grouping import digest_json

BACKEND = "exact_public_integer_relations_v1"

def integer_bits(value, width=64, signed=True):
    if type(value) is not int: raise TypeError("integers must not pass through float")
    lo, hi = (-(1 << (width-1)), (1 << (width-1))-1) if signed else (0, (1 << width)-1)
    if not lo <= value <= hi: raise OverflowError("out of declared bit representation")
    encoded = value & ((1 << width)-1)
    return tuple((encoded >> j) & 1 for j in range(width))

@dataclass(frozen=True)
class ObjectVersion:
    object_id: str
    version: int
    capacity: int | None
    unit: str
    node: int

@dataclass(frozen=True)
class Access:
    node: int
    operand: int | None
    kind: str
    unit: str
    candidate_versions: tuple[int, ...]
    path_literal: bool | None = None

def relation(access, obj):
    if access.kind not in {"index", "count"}: raise ValueError("unsupported relation")
    if access.operand is None or obj.capacity is None or access.unit == "unknown" or access.unit != obj.unit:
        return -1
    if type(access.operand) is not int or type(obj.capacity) is not int:
        raise TypeError("exact operand and capacity required")
    return int(access.operand < 0 or obj.capacity < 0 or
               (access.operand >= obj.capacity if access.kind == "index" else access.operand > obj.capacity))

def evidence_signature(objects, accesses):
    return digest_json({"backend": BACKEND, "objects": [vars(o) for o in objects], "accesses": [vars(a) for a in accesses]})

def validate_versions(objects):
    keys = [(o.object_id, o.version) for o in objects]
    if len(set(keys)) != len(keys): raise ValueError("duplicate object/version identity")
    for obj in objects:
        if obj.capacity is not None and type(obj.capacity) is not int:
            raise TypeError("capacity float is forbidden")
