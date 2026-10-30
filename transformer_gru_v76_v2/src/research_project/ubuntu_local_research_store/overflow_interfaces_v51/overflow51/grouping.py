"""Common manifests and train-only statistics; no source analysis."""
from collections import Counter
from hashlib import sha256
import json

def digest_json(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def components(records, project_holdout=False):
    parent = list(range(len(records)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, rec in enumerate(records):
        g = rec.group
        keys = [("exact", g.exact_hash)]
        if g.project and g.commit: keys.append(("commit", g.project, g.commit))
        keys.extend(("cve", x) for x in g.cves)
        if g.verified_pair: keys.append(("pair", g.verified_pair))
        if g.near_duplicate_component: keys.append(("near", g.near_duplicate_component))
        if project_holdout and g.project: keys.append(("project", g.project))
        for key in keys:
            if key in seen: parent[root(i)] = root(seen[key])
            else: seen[key] = i
    groups = {}
    for i, rec in enumerate(records): groups.setdefault(root(i), []).append(rec.group.record_id)
    return [tuple(sorted(x)) for x in groups.values()]

def frozen_split(records, *, salt, project_holdout=False):
    assignments = {}
    by_id={r.group.record_id:r for r in records}
    for group in components(records, project_holdout):
        value = int(sha256((salt + "|" + "|".join(group)).encode()).hexdigest()[:16], 16) / 2**64
        split = "train" if value < .8 else "val" if value < .9 else "test"
        if any(not by_id[rid].group.project or not by_id[rid].group.commit for rid in group):
            split='quarantine'
        assignments.update(dict.fromkeys(group, split))
    return {"assignments": assignments, "sha256": digest_json(assignments), "salt": salt}

def temporal_split(records, train_end, val_end):
    """ISO-8601 dates normalized upstream; components straddling boundaries quarantine."""
    by_id = {r.group.record_id: r for r in records}
    result = {}
    for group in components(records):
        periods = set()
        for rid in group:
            date = by_id[rid].group.timestamp
            periods.add("quarantine" if not date else "train" if date < train_end else "val" if date < val_end else "test")
        split = next(iter(periods)) if len(periods) == 1 else "quarantine"
        result.update(dict.fromkeys(group, split))
    return result

def fit_class_weights(ids, labels, manifest):
    if len(ids) != len(labels) or not ids or len(set(ids)) != len(ids):
        raise ValueError("unique nonempty aligned training IDs required")
    if any(manifest[i] != "train" for i in ids):
        raise ValueError("statistics may be fit on training rows only")
    counts = Counter(labels)
    if set(counts) != {0, 1}: raise ValueError("both classes required")
    return {"counts": dict(counts), "positive_weight": counts[0] / counts[1],
            "fit_ids_sha256": digest_json(sorted(ids))}
