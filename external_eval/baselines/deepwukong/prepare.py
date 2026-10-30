#!/usr/bin/env python3
"""Freeze matched-slice DeepWukong inputs without reading any model predictions."""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import adapter as a


def source_table(case):
    number = f"{case:09d}"
    root = a.PACKAGE / "external/overflowgen/author_data/data/overflow_gen"
    relative = Path("source-code") / number[:3] / number[3:6] / number[6:] / f"{case}.c"
    source = root / relative
    table = root / "csv/mnt/data/block/project/vdtransformer/data/overflow_gen" / relative
    return source, table


@lru_cache(maxsize=4096)
def sanitized_source(path, already_sanitized=False):
    import make_manifest as front
    source = Path(path).read_bytes()
    if already_sanitized:
        digest = hashlib.sha256(source).hexdigest()
        return source.decode("utf-8", errors="strict").splitlines(keepends=True), digest, digest
    tree, functions = front.function_info(source, ".c")
    clean, names, reasons = front.neutralize(source, tree, functions)
    if reasons or source.count(b"\n") != clean.count(b"\n"):
        raise ValueError("source sanitization unresolved")
    return clean.decode("utf-8", errors="strict").splitlines(keepends=True), hashlib.sha256(source).hexdigest(), hashlib.sha256(clean).hexdigest()


def tokens_for(path, lines, original, already_sanitized=False):
    source, raw_hash, clean_hash = sanitized_source(str(path), already_sanitized)
    code = [source[n - 1] for n in lines]
    words = [original["tokenize_code_line"](s, False) for s in original["clean_gadget"](code)]
    if any(not w for w in words):
        raise ValueError("empty token sequence in a retained benchmark node")
    # Comments are removed before official symbolization; no label annotations.
    if any("Tag." in s for s in code):
        raise ValueError("source annotation reached tokenizer")
    return words, raw_hash, clean_hash, "".join(code)


def verify_edges(table, source, lines, cached, original, api):
    pdg, _ = original["build_PDG"](str(table), str(api), str(source))
    if pdg is None:
        raise ValueError("native PDG missing in regression fixture")
    expected = {(u, v) for u, v in pdg.edges if u in lines and v in lines}
    actual = {(lines[int(u)], lines[int(v)]) for u, v in np.argwhere(np.asarray(cached) > 0)}
    if expected != actual:
        raise ValueError("cached Joern line projection is not the native induced PDG")
    return {"source": str(source), "source_sha256": a.sha(source),
            "nodes_sha256": a.sha(table / "nodes.csv"), "edges_sha256": a.sha(table / "edges.csv"),
            "source_lines": lines, "native_induced_edges": len(expected), "edge_sets_exact": True}


def source_records(original, api, fixtures, limit=0):
    cache = a.PACKAGE / "external/overflowgen/joint_graph_study_v1"
    receipt = a.read(cache / "CACHE_RECEIPT.json")
    if a.sha(cache / "edges.npy") != receipt["cache_hashes"]["edges.npy"]:
        raise ValueError("original Joern line-edge cache changed")
    edges = np.load(cache / "edges.npy", mmap_mode="r", allow_pickle=False)
    metadata_path = a.PACKAGE / "assets/comparisons/overflowgen/overflow_interfaces_v51/output/row_metadata.jsonl.gz"
    split_receipt = a.read(a.WORKSPACE / "reproducibility/data_split_check.json")
    if a.sha(metadata_path) != split_receipt["metadata_sha256"]:
        raise ValueError("source labels/splits changed")
    with gzip.open(metadata_path, "rt") as stream:
        metadata = [json.loads(line) for line in stream]
    row = 0
    for split in ("train", "val", "test"):
        path = a.PACKAGE / "external/overflowgen/author_data/data/overflow_gen" / (split + "_data.json")
        if a.sha(path) != split_receipt["author_json_sha256"][split]:
            raise ValueError("author slice JSON changed")
        data = a.read(path)
        for sample_id, sample in data.items():
            meta = metadata[row]
            if meta["original_split"] != split or str(meta["original_sample_id"]) != sample_id or meta["label"] != int(sample["label"]):
                raise ValueError("original slice identity mismatch")
            lines = [int(n) for n in sample["line_num"]]
            source, table = source_table(meta["source_case"])
            words, raw_hash, clean_hash, text = tokens_for(source, lines, original)
            matrix = np.asarray(edges[row, :len(lines), :len(lines)])
            if row < 128:
                fixtures.append({"dataset": "overflowgen", "row": row,
                                 **verify_edges(table, source, lines, matrix, original, api)})
            yield {"sample_id": f"og24455320:{split}:{sample_id}", "row": row,
                   "split": meta["family_split"], "scene_id": f"og24455320:{split}:{sample_id}",
                   "family_id": meta["family_id"], "source_case": meta["source_case"], "label": meta["label"],
                   "source_lines": lines, "key_source_line": lines[sample["key_line"]],
                   "token_lines": words, "edges": matrix.tolist(), "slice_text": text,
                   "source_path": str(source.resolve()), "native_pdg_path": str(table.resolve()),
                   "raw_source_sha256": raw_hash, "sanitized_source_sha256": clean_hash}
            row += 1
            if limit and row >= limit:
                return
        del data
    if row != 117696:
        raise ValueError("source population changed")


def external_records(dataset, original, api, fixtures):
    import paper_frozen_evaluate as paper
    import mit_baseline_frozen_evaluate as mit
    context = paper.external_inputs(dataset)
    # Signed prior preparation identifies every tensor and source; no frontend rerun.
    if dataset == "mit":
        source_root = a.EXTERNAL / "manifests/mit"
        old = a.read(a.EXTERNAL / "prepared_mit_condition_overlay/preparation_receipt.json")
    else:
        source_root = a.EXTERNAL / "formai_pool/prepared"
        old = a.read(a.EXTERNAL / "formai_pool/adaptation/prepared/preparation_receipt.json")
        # Adaptation reuses the base Joern tables and receipts only its new cache.
        base = a.read(source_root / "preparation_receipt.json")
    tables, scene_tables = {}, {}
    for unit_id, unit in old["completed"].items():
        candidates = [Path(p).parent for p in unit.get("artifact_sha256", {}) if Path(p).name == "nodes.csv"]
        if not candidates and dataset == "formai":
            base_artifacts = base["completed"][unit_id]["artifact_sha256"]
            candidates = [Path(p).parent for p in base_artifacts if Path(p).name == "nodes.csv"]
            for table in candidates:
                for filename in ("nodes.csv", "edges.csv"):
                    if a.sha(table / filename) != base_artifacts[str(table / filename)]:
                        raise ValueError("base FormAI Joern table changed")
        if len(candidates) == 1:
            for row in unit.get("slices", []):
                tables[row["slice_id"]] = candidates[0]
                for scene_id in row.get("owner_scene_ids", []):
                    scene_tables[scene_id] = candidates[0]
    grouped = {}
    for index, row in enumerate(context["records"]):
        scene = context["scenes"][row["owner_scene_ids"][0]]
        source = source_root / scene["sanitized_path"]
        if a.sha(source) != scene["sanitized_sha256"]:
            raise ValueError("external sanitized source changed")
        path = row["tensor_cache"]
        if path not in grouped:
            with np.load(path, allow_pickle=False) as arrays:
                grouped[path] = np.array(arrays["edges"], copy=True)
        lines = row["source_line_numbers"]
        matrix = grouped[path][row["tensor_row"], :len(lines), :len(lines)]
        words, raw_hash, clean_hash, text = tokens_for(source, lines, original, already_sanitized=True)
        table = tables.get(row["slice_id"], scene_tables.get(scene["scene_id"]))
        if table is None:
            raise ValueError("external original PDG table is unresolved")
        if index < 16 and table is not None:
            fixtures.append({"dataset": dataset, "slice_id": row["slice_id"],
                             **verify_edges(table, source, lines, matrix, original, api)})
        yield {"sample_id": row["slice_id"], "row": None, "split": "external", "scene_id": scene["scene_id"],
               "family_id": scene["family_id"], "variant": scene.get("variant", ""), "label": scene["label"],
               "source_lines": lines, "key_source_line": row["key_source_line"],
               "token_lines": words, "edges": matrix.tolist(), "slice_text": text,
               "source_path": str(source.resolve()), "native_pdg_path": str(table.resolve()) if table else None,
               "in_common_native": scene["scene_id"] in context["common"],
               "raw_source_sha256": raw_hash, "sanitized_source_sha256": clean_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="only first128 source samples; separate pilot artifact")
    args = parser.parse_args()
    out = a.OUTPUT / ("data_pilot" if args.pilot else "data")
    out.mkdir(parents=True, exist_ok=True)
    final = out / "preparation.json"
    if final.exists():
        receipt = a.frozen.signed_read(final, "result_sha256")
        for name, expected in receipt["artifacts"].items():
            if a.sha(out / name) != expected:
                raise ValueError("completed prepared baseline data changed")
        print(json.dumps({"status": "SKIP_VERIFIED", "directory": str(out)}))
        return
    original, api = a.original_functions()
    fixtures, summaries, artifacts = [], {}, {}
    started = time.monotonic()
    scopes = ["source"] if args.pilot else ["source", "mit", "formai"]
    for name in scopes:
        path = out / (name + ".jsonl.gz")
        if path.exists():
            raise ValueError("unreceipted preparation output exists; inspect before replacing")
        counts = Counter()
        iterator = source_records(original, api, fixtures, 128 if args.pilot else 0) if name == "source" else external_records(name, original, api, fixtures)
        text_path = out / (name + "_text.jsonl.gz")
        with gzip.open(path, "wt") as stream, gzip.open(text_path, "wt") as text_stream:
            for record in iterator:
                counts["samples"] += 1
                counts[record["split"]] += 1
                counts[record["split"] + "_positive"] += record["label"]
                counts["statements"] += len(record["token_lines"])
                counts["truncated_statements"] += sum(len(words) > 16 for words in record["token_lines"])
                counts["samples_with_truncation"] += any(len(words) > 16 for words in record["token_lines"])
                counts["isolated_graphs"] += not np.asarray(record["edges"]).any()
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                text_record = {k: v for k, v in record.items() if k not in {"token_lines", "edges"}}
                text_record["dataset"] = "overflowgen" if name == "source" else name
                text_stream.write(json.dumps(text_record, separators=(",", ":")) + "\n")
                if counts["samples"] % 10000 == 0:
                    print(json.dumps({"dataset": name, "samples": counts["samples"], "seconds": time.monotonic() - started}), flush=True)
        summaries[name] = dict(counts)
        artifacts[path.name] = a.sha(path)
        artifacts[text_path.name] = a.sha(text_path)
    if not args.pilot and (summaries["source"]["train"], summaries["source"]["val"], summaries["source"]["test"]) != (94179, 11940, 11577):
        raise ValueError("frozen family split counts differ")
    a.signed(out / "native_edge_regression.json", {"status": "PASS", "fixtures": fixtures,
              "comparison": "native build_PDG CONTROLS/REACHES projected to retained source lines versus original typed line-edge cache"})
    artifacts["native_edge_regression.json"] = a.sha(out / "native_edge_regression.json")
    a.signed(final, {"status": "COMPLETE", "scope": "PILOT_ONLY" if args.pilot else "FROZEN_ALL_COHORTS",
                    "vendor": a.verify_vendor(), "adapter_sha256": a.sha(a.__file__), "prepare_sha256": a.sha(__file__),
                    "api_sha256": a.sha(api), "sanitizer_sha256": a.sha(a.EXTERNAL / "make_manifest.py"),
                    "numeric_candidate_graph_used": False, "source_labels_used_for_feature_selection": False,
                    "embedding_trained": False, "counts": summaries, "artifacts": artifacts,
                    "wall_s": time.monotonic() - started})
    print(json.dumps({"status": "COMPLETE", "counts": summaries, "wall_s": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
