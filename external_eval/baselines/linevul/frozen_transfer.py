#!/usr/bin/env python3
"""Official Big-Vul task checkpoint: a separate frozen cross-domain panel.

True neutral source snippets are shared with DeepWukong; no tensor-to-text,
new fitting, threshold tuning, or equal-source-training claim is involved.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent
EXTERNAL = BASE.parents[1]
WORKSPACE = EXTERNAL.parent
OUT = EXTERNAL / "paper_baselines/linevul"
SHARED = Path(os.environ.get("OVERFLOW_DATA_DIR", WORKSPACE / "data")) / "baselines/deepwukong"
OFFICIAL = WORKSPACE / "transformer_gru_v76_v2/third_party/LineVul/linevul/linevul_model.py"
NATIVE_MAIN = OFFICIAL.with_name("linevul_main.py")
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXTERNAL))
import frozen_evaluate as frozen
sha = frozen.sha


def read(path):
    return json.loads(Path(path).read_text())


def signed(path, obj):
    frozen.atomic_json(path, {**obj, "result_sha256": frozen.digest(obj)})


def csv_write(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def identity():
    manifest = read(BASE / "OFFICIAL_CHECKPOINT_MANIFEST.json")
    checkpoint = BASE / manifest["file"]
    if checkpoint.stat().st_size != manifest["bytes"] or sha(checkpoint) != manifest["sha256"]:
        raise ValueError("official task checkpoint changed")
    assets = read(BASE / "pretrained/SOURCE_MANIFEST.json")
    for item in assets["files"]:
        if sha(BASE / "pretrained" / item["path"]) != item["sha256"]:
            raise ValueError("official tokenizer/config assets changed")
    return {"checkpoint_sha256": manifest["sha256"], "checkpoint_manifest_sha256": sha(BASE / "OFFICIAL_CHECKPOINT_MANIFEST.json"),
            "official_class_sha256": sha(OFFICIAL), "official_main_sha256": sha(NATIVE_MAIN),
            "pretrained_manifest_sha256": sha(BASE / "pretrained/SOURCE_MANIFEST.json"),
            "runner_sha256": sha(__file__), "training_domain": "official Big-Vul task-trained checkpoint",
            "panel": "single-checkpoint cross-domain transfer; separate from equal-source-training comparisons"}


def load_inputs():
    prepared = read(SHARED / "manifest.json")
    data, hashes = {}, {}
    available = [name for name in ("source", "mit", "formai") if (SHARED / (name + ".jsonl.gz")).exists()]
    for name in available:
        filename = name + ".jsonl.gz"
        path = SHARED / filename
        hashes[filename] = sha(path)
        if hashes[filename] != prepared["files"][filename]:
            raise ValueError("shared neutral text changed")
        with gzip.open(path, "rt") as stream:
            rows = [r for line in stream if (r := json.loads(line))["split"] == "test" or name != "source"]
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate sample IDs")
        data["test" if name == "source" else name] = rows
    if any(len(data[name]) != expected for name, expected in (("test", 11577), ("mit", 1578), ("formai", 2709)) if name in data):
        raise ValueError("frozen slice populations differ")
    for name, scenes, common in (("mit", 1059, 796), ("formai", 167, 161)):
        if name not in data:
            continue
        if len({r["scene_id"] for r in data[name]}) != scenes or len({r["scene_id"] for r in data[name] if r["in_common_native"]}) != common:
            raise ValueError("frozen external scene populations differ")
    return data, {"preparation_sha256": sha(SHARED / "manifest.json"), "manifests": hashes}


def model_and_tokenizer(threads):
    import torch
    import transformers
    from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = RobertaConfig.from_pretrained(BASE / "pretrained", local_files_only=True)
    config.num_labels = 1
    config.num_attention_heads = 12
    config._attn_implementation = "eager"
    tokenizer = RobertaTokenizer.from_pretrained(BASE / "pretrained", local_files_only=True)
    spec = importlib.util.spec_from_file_location("official_linevul_model", OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Every parameter is replaced by the official task state under strict=True.
    encoder = RobertaForSequenceClassification(config)
    model = module.Model(encoder, config, tokenizer, SimpleNamespace(block_size=512))
    for embeddings in (model.roberta.embeddings, model.encoder.roberta.embeddings):
        embeddings._non_persistent_buffers_set.discard("position_ids")
    weights = torch.load(BASE / "official_12heads_linevul_model.bin", map_location="cpu", weights_only=True, mmap=True)
    if len(weights) != 404:
        raise ValueError("unexpected official task state")
    result = model.load_state_dict(weights, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("strict official state mismatch")
    for embeddings in (model.roberta.embeddings, model.encoder.roberta.embeddings):
        if not torch.equal(embeddings.position_ids, torch.arange(514).expand((1, -1))):
            raise ValueError("position buffer differs from native deterministic values")
    del weights
    model.eval()
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__, "CPU_threads": threads,
               "precision": "FP32", "attention": "eager", "load_weights_only": True, "strict": True,
               "missing_keys": [], "unexpected_keys": [], "state_keys": 404,
               "position_ids": "existing deterministic buffers restored as persistent; no key dropped",
               "model_class": "official linevul_model.Model", "num_attention_heads": 12}
    return torch, model, tokenizer, runtime


def tokenize(rows, tokenizer):
    encoded = []
    for row in rows:
        native = tokenizer.tokenize(str(row["slice_text"]))
        tokens = [tokenizer.cls_token] + native[:510] + [tokenizer.sep_token]
        encoded.append((tokenizer.convert_tokens_to_ids(tokens), len(native)))
    return encoded


def forward(torch, model, encoded, fixed=False):
    length = 512 if fixed else max(len(ids) for ids, _ in encoded)
    ids = torch.tensor([ids + [1] * (length - len(ids)) for ids, _ in encoded], dtype=torch.long)
    with torch.inference_mode():
        probabilities = model(input_ids=ids)
    if probabilities.shape != (len(encoded), 2) or not torch.isfinite(probabilities).all():
        raise ValueError("invalid official classifier output")
    return probabilities


def accept(threads):
    ident = identity()
    data, data_identity = load_inputs()
    torch, model, tokenizer, runtime = model_and_tokenizer(threads)
    fixtures, maximum, timings = [], 0., []
    # Quantiles plus maximum cover variable-length and truncated native inputs.
    for name, rows in data.items():
        encoded = tokenize(rows, tokenizer)
        ordered = sorted(range(len(rows)), key=lambda i: encoded[i][1])
        indexes = [ordered[round((len(rows) - 1) * q)] for q in (0, .1, .25, .5, .75, .9, .99, 1)]
        selected = [encoded[i] for i in indexes]
        started = time.monotonic()
        fixed = forward(torch, model, selected, fixed=True)
        fixed_s = time.monotonic() - started
        # Compare each sample individually so true dynamic lengths are exercised.
        started = time.monotonic()
        dynamic = torch.cat([forward(torch, model, [sample]) for sample in selected])
        dynamic_s = time.monotonic() - started
        error = float((fixed - dynamic).abs().max())
        maximum = max(maximum, error)
        parity = torch.equal(fixed[:, 1] >= .5, dynamic[:, 1] >= .5)
        if error > 1e-5 or not parity:
            raise ValueError("dynamic padding differs from native fixed512")
        fixtures += [{"dataset": name, "sample_id": rows[i]["sample_id"], "raw_bpe_tokens": encoded[i][1],
                      "retained_bpe_tokens": len(encoded[i][0]) - 2, "fixed_probability": fixed[j].tolist(),
                      "dynamic_probability": dynamic[j].tolist()} for j, i in enumerate(indexes)]
        timings.append({"dataset": name, "fixtures": len(indexes), "fixed512_batch_seconds": fixed_s, "dynamic_single_seconds": dynamic_s})
        print(json.dumps({"event": "padding_acceptance", "dataset": name, "max_error": error, "label_parity": parity}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "PASS_STRICT_OFFICIAL_CHECKPOINT_AND_DYNAMIC_PADDING", "identity": ident,
               "data": data_identity, "runtime": runtime, "fixtures": fixtures, "timings": timings,
               "max_probability_error": maximum, "probability_atol": 1e-5, "threshold_label_parity": True,
               "native_tokenization": "raw neutral slice_text; RobertaTokenizer.tokenize; first510 BPE + CLS/SEP; pad ID1 to512",
               "acceleration": "right-pad each inference batch only to longest retained token sequence <=512",
               "task_weights_used": True, "parameters_updated": False}
    signed(OUT / "acceptance.json", receipt)
    protocol = {"status": "SEALED_FROZEN_TRANSFER", "identity": ident, "data": data_identity,
                "acceptance_sha256": sha(OUT / "acceptance.json"), "threshold": .5,
                "selection": "official published Big-Vul task checkpoint fixed before target evaluation",
                "source_training": False, "target_tuning": False, "seeds": "single official checkpoint; no three-seed standard deviation",
                "input_unit": "fixed benchmark backward slices", "max_tokens": 512,
                "external_aggregation": "maximum slice risk per scene; common scenes filtered from same scores",
                "cohorts": {"test": {"slices": 11577}, "mit": {"slices": 1578, "scenes": 1059, "common_scenes": 796},
                            "formai": {"slices": 2709, "scenes": 167, "common_scenes": 161}},
                "coverage_scope": "all samples in frozen eligible manifest; earlier frontend exclusions outside these cohorts are not scored"}
    signed(OUT / "protocol.json", protocol)
    print(json.dumps({"status": receipt["status"], "max_error": maximum, "protocol_sha256": sha(OUT / "protocol.json")}), flush=True)


def metrics(rows):
    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in rows)
    tn = sum(r["label"] == 0 and r["prediction"] == 0 for r in rows)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in rows)
    return {"n": len(rows), "positive": tp + fn, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0., "accuracy": (tp + tn) / len(rows)}


def infer_dataset(name, rows, encoded, torch, model, request, batch_size):
    where = OUT / name
    where.mkdir(parents=True, exist_ok=True)
    request = {**request, "dataset": name, "samples": len(rows), "batch_size": batch_size,
               "ordered_sample_ids_sha256": frozen.digest([r["sample_id"] for r in rows])}
    if (where / "receipt.json").exists():
        previous = frozen.signed_read(where / "receipt.json", "result_sha256")
        if previous["request"] != request:
            raise ValueError("completed inference request differs")
        for filename, expected in previous["artifacts"].items():
            if sha(where / filename) != expected:
                raise ValueError("completed prediction artifact changed")
        print(json.dumps({"event": "skip_verified", "dataset": name}), flush=True)
        return
    connection = sqlite3.connect(where / "scores.sqlite3")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS scores (sample_id TEXT PRIMARY KEY,risk REAL NOT NULL,p0 REAL NOT NULL,raw_tokens INTEGER NOT NULL)")
    wanted = json.dumps(request, sort_keys=True)
    old = connection.execute("SELECT value FROM metadata WHERE key='request'").fetchone()
    if old is not None and old[0] != wanted:
        raise ValueError("SQLite resume identity differs")
    connection.execute("INSERT OR IGNORE INTO metadata VALUES ('request',?)", (wanted,))
    connection.commit()
    done = {r[0] for r in connection.execute("SELECT sample_id FROM scores")}
    all_ids = {r["sample_id"] for r in rows}
    if not done <= all_ids:
        raise ValueError("SQLite has scores outside the frozen manifest")
    pending = sorted([i for i, row in enumerate(rows) if row["sample_id"] not in done], key=lambda i: len(encoded[i][0]))
    started = time.monotonic()
    for offset in range(0, len(pending), batch_size):
        indexes = pending[offset:offset + batch_size]
        result = forward(torch, model, [encoded[i] for i in indexes]).tolist()
        connection.executemany("INSERT INTO scores VALUES (?,?,?,?)", [(rows[i]["sample_id"], float(p[1]), float(p[0]), encoded[i][1]) for i, p in zip(indexes, result)])
        connection.commit()
        if offset % (batch_size * 25) == 0:
            print(json.dumps({"event": "inference", "dataset": name, "scored": len(done) + min(offset + batch_size, len(pending)),
                              "total": len(rows), "wall_s": time.monotonic() - started}), flush=True)
    scores = {sid: (risk, p0, tokens) for sid, risk, p0, tokens in connection.execute("SELECT * FROM scores")}
    if set(scores) != all_ids:
        raise ValueError("incomplete final sample coverage")
    output = []
    for row in rows:
        risk, p0, tokens = scores[row["sample_id"]]
        output.append({"sample_id": row["sample_id"], "row": row["row"], "scene_id": row["scene_id"], "family_id": row["family_id"],
                       "label": row["label"], "risk": risk, "prediction": int(risk >= .5), "probability0": p0,
                       "raw_bpe_tokens": tokens, "truncated": int(tokens > 510), "in_common_native": row.get("in_common_native", False)})
    csv_write(where / "slice_predictions.csv", output)
    artifacts = ["slice_predictions.csv", "metrics.json", "scores.sqlite3"]
    summary = {"slices": metrics(output)}
    if name != "test":
        scenes = {}
        for row in output:
            sid = row["scene_id"]
            if sid in scenes and scenes[sid]["label"] != row["label"]:
                raise ValueError("scene label inconsistency")
            if sid not in scenes or row["risk"] > scenes[sid]["risk"]:
                scenes[sid] = {k: row[k] for k in ("scene_id", "family_id", "label", "risk", "prediction", "in_common_native")}
        scene_rows = list(scenes.values())
        common = [r for r in scene_rows if r["in_common_native"]]
        csv_write(where / "scene_predictions.csv", scene_rows)
        csv_write(where / "common_scene_predictions.csv", common)
        summary.update({"scenes": metrics(scene_rows), "common_scenes": metrics(common)})
        artifacts += ["scene_predictions.csv", "common_scene_predictions.csv"]
    summary["coverage"] = {"eligible_manifest_slices": len(rows), "scored": len(output), "unscored": 0,
                           "truncated_slices": sum(r["truncated"] for r in output), "threshold": .5}
    signed(where / "metrics.json", summary)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    signed(where / "receipt.json", {"status": "COMPLETE", "request": request, "coverage": summary["coverage"],
           "artifacts": {n: sha(where / n) for n in artifacts}})
    print(json.dumps({"event": "inference_complete", "dataset": name, "metrics": summary}), flush=True)


def infer(threads, batch_size):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = frozen.signed_read(OUT / "protocol.json", "result_sha256")
        accepted = frozen.signed_read(OUT / "acceptance.json", "result_sha256")
        ident = identity()
        data, data_identity = load_inputs()
        if protocol["identity"] != ident or accepted["identity"] != ident or protocol["data"] != data_identity:
            raise ValueError("implementation/data changed after acceptance")
        if sha(OUT / "acceptance.json") != protocol["acceptance_sha256"]:
            raise ValueError("acceptance changed after sealing")
        torch, model, tokenizer, runtime = model_and_tokenizer(threads)
        request = {"protocol_sha256": sha(OUT / "protocol.json"), "identity": ident, "data": data_identity, "runtime": runtime, "threshold": .5}
        for name, rows in data.items():
            infer_dataset(name, rows, tokenize(rows, tokenizer), torch, model, request, batch_size)
        signed(OUT / "complete.json", {"status": "COMPLETE", "request": request,
               "receipt_hashes": {n: sha(OUT / n / "receipt.json") for n in data}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["accept", "infer"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "accept":
        if (OUT / "protocol.json").exists():
            raise ValueError("protocol already sealed; inspect before replacing")
        accept(args.threads)
    else:
        infer(args.threads, args.batch_size)
