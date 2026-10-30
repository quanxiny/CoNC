#!/usr/bin/env python3
"""Train the pinned official GCN on frozen source slices and evaluate once.

Upstream model/optimizer/scheduler bodies are unchanged. The loop preserves
batch64, Adam0.002, gamma0.95, clip5, max50, patience10 and DEV CE selection.
No external loss, prediction, label or metric participates in training choices.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time
import adapter as a
import numpy as np


def atomic_torch(path, value, torch):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def embedding(records, directory, seed, pilot=False):
    from gensim.models import Word2Vec
    from src.vocabulary import Vocabulary
    path = directory / "w2v.wv"
    receipt_path = directory / "embedding.json"
    selected = [r for r in records if r["split"] == "train"]
    request = {"scope": "PILOT_ONLY" if pilot else "SOURCE_TRAIN_ONLY",
               "train_ids_sha256": a.frozen.digest([r["sample_id"] for r in selected]),
               "n_train": len(selected), "seed": seed, "vector_size": 256, "min_count": 3,
               "max_vocab_size": 190000, "workers": 1, "sg": 1,
               "window": 5, "negative": 5, "epochs": 5,
               "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
               "source_recipe_sha256": a.sha(a.VENDOR / "src/preprocess/word_embedding.py")}
    if receipt_path.exists():
        previous = a.frozen.signed_read(receipt_path, "result_sha256")
        if previous["request"] != request:
            raise ValueError("embedding request changed")
        for name, expected in previous["artifacts"].items():
            if a.sha(directory / name) != expected:
                raise ValueError("TRAIN-only embedding changed")
    else:
        sentences = [words for row in selected for words in row["token_lines"]]
        started = time.monotonic()
        model = Word2Vec(sentences=sentences, min_count=3, vector_size=256,
                         max_vocab_size=190000, workers=1, sg=1, seed=seed,
                         window=5, negative=5, epochs=5)
        model.wv.save(str(path))
        files = list(directory.glob("w2v.wv*"))
        a.signed(receipt_path, {"request": request, "status": "COMPLETE", "vocabulary_size": len(model.wv),
                               "wall_s": time.monotonic() - started,
                               "artifacts": {p.name: a.sha(p) for p in files}})
        del model, sentences
    return path, Vocabulary.build_from_w2v(str(path))


def batch(records, indexes, graphs, torch):
    from torch_geometric.data import Batch
    return Batch.from_data_list([graphs[i] for i in indexes]), torch.tensor([records[i]["label"] for i in indexes], dtype=torch.long)


def evaluate(model, records, indexes, graphs, torch):
    model.eval()
    losses, rows, total_loss = [], [], 0.0
    with torch.inference_mode():
        for offset in range(0, len(indexes), 64):
            ix = indexes[offset:offset + 64]
            x, labels = batch(records, ix, graphs, torch)
            logits = model(x)
            if logits.shape != (len(ix), 2) or not torch.isfinite(logits).all():
                raise ValueError("nonfinite/invalid official logits")
            loss = torch.nn.functional.cross_entropy(logits, labels)
            losses.append(float(loss))
            total_loss += float(loss) * len(ix)
            for i, score, logit in zip(ix, logits.softmax(-1)[:, 1].tolist(), logits.tolist()):
                r = records[i]
                rows.append({"sample_id": r["sample_id"], "row": r["row"], "scene_id": r["scene_id"],
                             "family_id": r["family_id"], "label": r["label"], "risk": score,
                             "prediction": int(score >= .5), "margin": logit[1] - logit[0],
                             "logit0": logit[0], "logit1": logit[1]})
    return float(np.mean(losses)), total_loss / len(indexes), rows


def acceptance(seed, threads):
    """Input conversion, upstream model and optimizer smoke; never eligible weights."""
    torch = a.runtime(seed, threads)
    directory = a.OUTPUT / "acceptance"
    directory.mkdir(parents=True, exist_ok=True)
    records = [r for r in a.load_records("source") if r["split"] == "train"][:128]
    if len(records) != 128:
        raise ValueError("fixed128 input fixtures required")
    w2v, vocab = embedding(records, directory, seed, pilot=True)
    model = a.make_model(seed, w2v, vocab)
    graphs = []
    for r in records:
        native, fast = a.official_data(r, vocab), a.encoded_data(r, vocab)
        if not torch.equal(native.x, fast.x) or not torch.equal(native.edge_index, fast.edge_index):
            raise ValueError("fast converter differs from official XFG.to_torch")
        graphs.append(fast)
    from torch_geometric.data import Batch
    model.eval()
    with torch.inference_mode():
        first = model(Batch.from_data_list([a.official_data(r, vocab) for r in records[:64]]))
        second = model(Batch.from_data_list(graphs[:64]))
    if not torch.equal(first, second):
        raise ValueError("native/adapter model forward regression failed")
    opt = model.configure_optimizers()
    optimizer = opt["optimizer"]
    selected = [i for i, r in enumerate(records) if r["split"] == "train"]
    timings = []
    model.train()
    for step in range(8):
        ix = [selected[(step * 64 + j) % len(selected)] for j in range(64)]
        x, y = batch(records, ix, graphs, torch)
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if not math.isfinite(float(loss.detach())) or not math.isfinite(float(norm)):
            raise ValueError("official optimization smoke failed")
        timings.append(time.monotonic() - started)
    receipt = {"status": "PASS_NATIVE_MODEL_AND_INPUT_PARITY", "pilot_only": True,
               "fixtures": 128, "token_tensor_exact": True, "edge_index_exact": True,
               "logits_bitwise_equal": True, "native_class": type(model).__module__ + "." + type(model).__name__,
               "parameters": sum(p.numel() for p in model.parameters()), "CPU_threads": threads,
               "training_batches": 8, "batch_size": 64, "batch_times_s": timings,
               "median_step_s": float(np.median(timings[1:])),
               "estimated_source_epoch_train_s": float(np.median(timings[1:])) * math.ceil(94179 / 64),
               "model_weights_saved": False, "adapter_sha256": a.sha(a.__file__),
               "runner_sha256": a.sha(__file__), "vendor": a.verify_vendor()}
    a.signed(directory / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def rng_state(torch):
    nr = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [nr[0], nr[1].tolist(), nr[2], nr[3], nr[4]]}


def restore_rng(state, torch):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    nr = state["numpy"]
    np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))


def frozen_infer(name, model, records, vocab, directory, checkpoint_sha, torch, seed):
    """One SQLite-ledger pass per chosen state; no threshold/model choices here."""
    where = directory / ("test" if name == "source" else name)
    where.mkdir(exist_ok=True)
    selected = [r for r in records if r["split"] == ("test" if name == "source" else "external")]
    request = {"dataset": name, "seed": seed, "method": "deepwukong_matched_slice",
               "checkpoint_sha256": checkpoint_sha, "threshold": .5, "batch_size": 64,
               "data_receipt_sha256": a.sha(a.data_manifest()),
               "sample_ids_sha256": a.frozen.digest([r["sample_id"] for r in selected]),
               "script_sha256": a.sha(__file__), "adapter_sha256": a.sha(a.__file__)}
    digest = a.frozen.digest(request)
    if (where / "receipt.json").exists():
        done = a.frozen.signed_read(where / "receipt.json", "result_sha256")
        if done["request"] != request:
            raise ValueError("completed inference request changed")
        for path, expected in done["artifacts"].items():
            if a.sha(where / path) != expected:
                raise ValueError("completed predictions changed")
        return
    c = sqlite3.connect(where / "scores.sqlite3")
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=FULL")
    c.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    c.execute("CREATE TABLE IF NOT EXISTS scores(sample_id TEXT PRIMARY KEY,risk REAL NOT NULL,logit0 REAL NOT NULL,logit1 REAL NOT NULL)")
    prior = c.execute("SELECT value FROM metadata WHERE key='request_sha256'").fetchone()
    if prior and prior[0] != digest:
        raise ValueError("inference ledger identity mismatch")
    c.execute("INSERT OR IGNORE INTO metadata VALUES('request_sha256',?)", (digest,))
    c.commit()
    completed = {r[0] for r in c.execute("SELECT sample_id FROM scores")}
    expected_ids = {r["sample_id"] for r in selected}
    if not completed <= expected_ids or len(expected_ids) != len(selected):
        raise ValueError("inference sample identity mismatch")
    pending = [r for r in selected if r["sample_id"] not in completed]
    from torch_geometric.data import Batch
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(pending), 64):
            subset = pending[offset:offset + 64]
            output = model(Batch.from_data_list([a.encoded_data(r, vocab) for r in subset]))
            if output.shape != (len(subset), 2) or not torch.isfinite(output).all():
                raise ValueError("invalid frozen predictions")
            rows = [(r["sample_id"], p, q[0], q[1]) for r, p, q in zip(subset, output.softmax(-1)[:, 1].tolist(), output.tolist())]
            with c:
                c.executemany("INSERT INTO scores VALUES(?,?,?,?)", rows)
    saved = {r[0]: r[1:] for r in c.execute("SELECT * FROM scores")}
    if set(saved) != expected_ids:
        raise ValueError("incomplete prediction ledger")
    predictions = []
    for r in selected:
        score, lo, hi = saved[r["sample_id"]]
        predictions.append({"method": "deepwukong_matched_slice", "seed": seed,
                            "sample_id": r["sample_id"], "row": r["row"], "scene_id": r["scene_id"],
                            "family_id": r["family_id"], "label": r["label"], "risk": score,
                            "prediction": int(score >= .5), "margin": hi - lo, "logit0": lo, "logit1": hi})
    write_csv(where / "slice_predictions.csv", predictions)
    files = ["slice_predictions.csv"]
    if name != "source":
        grouped, metadata = defaultdict(list), {}
        for r in predictions:
            grouped[r["scene_id"]].append(r)
        for r in selected:
            metadata[r["scene_id"]] = r
        scenes = []
        for scene_id, values in sorted(grouped.items()):
            best = max(sorted(values, key=lambda r: r["sample_id"]), key=lambda r: r["risk"])
            scenes.append({**{k: best[k] for k in ("method", "seed", "scene_id", "family_id", "label", "risk", "prediction", "margin")},
                           "max_slice_id": best["sample_id"], "slice_count": len(values),
                           "in_common_native": metadata[scene_id]["in_common_native"]})
        write_csv(where / "scene_predictions.csv", scenes)
        write_csv(where / "common_scene_predictions.csv", [r for r in scenes if r["in_common_native"]])
        files += ["scene_predictions.csv", "common_scene_predictions.csv"]
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    a.signed(where / "receipt.json", {"status": "COMPLETE", "request": request,
             "samples": len(selected), "artifacts": {p: a.sha(where / p) for p in files + ["scores.sqlite3"]}})
    print(json.dumps({"event": "inference_complete", "seed": seed, "dataset": name, "samples": len(selected)}), flush=True)


def train(seed, threads):
    torch = a.runtime(seed, threads)
    protocol = a.frozen.signed_read(a.OUTPUT / "protocol.json", "result_sha256")
    acceptance_receipt = a.frozen.signed_read(a.OUTPUT / "acceptance/receipt.json", "result_sha256")
    if acceptance_receipt["status"] != "PASS_NATIVE_MODEL_AND_INPUT_PARITY":
        raise ValueError("official model/input acceptance required")
    if acceptance_receipt["adapter_sha256"] != a.sha(a.__file__) or acceptance_receipt["runner_sha256"] != a.sha(__file__):
        raise ValueError("implementation changed after acceptance")
    records = a.load_records("source")
    indexes = {split: [i for i, r in enumerate(records) if r["split"] == split] for split in ("train", "val", "test")}
    if tuple(len(indexes[s]) for s in ("train", "val", "test")) != (94179, 11940, 11577):
        raise ValueError("frozen split mismatch")
    directory = a.OUTPUT / "runs" / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = {"seed": seed, "threads": threads, "precision": "FP32", "protocol_sha256": a.sha(a.OUTPUT / "protocol.json"),
                   "data_sha256": a.sha(a.data_manifest()), "vendor": a.verify_vendor(),
                   "adapter_sha256": a.sha(a.__file__), "runner_sha256": a.sha(__file__)}
        if (directory / "request.json").exists() and a.read(directory / "request.json") != request:
            raise ValueError("immutable training request changed")
        a.frozen.atomic_json(directory / "request.json", request)
        w2v, vocab = embedding(records, directory, seed)
        model = a.make_model(seed, w2v, vocab)
        optimizers = model.configure_optimizers()
        optimizer, scheduler = optimizers["optimizer"], optimizers["lr_scheduler"]
        start_epoch, best, early_best, bad_epochs, best_epoch = 0, float("inf"), float("inf"), 0, -1
        if (directory / "last.pt").exists():
            state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
            if state["request"] != request:
                raise ValueError("resume checkpoint request changed")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            restore_rng(state["rng"], torch)
            start_epoch, best, bad_epochs, best_epoch = state["epoch"] + 1, state["best"], state["bad_epochs"], state["best_epoch"]
            early_best = state["early_best"]
        if not (directory / "training_complete.json").exists():
            graphs = [a.encoded_data(r, vocab) for r in records]
            for epoch in range(start_epoch, 50):
                if bad_epochs >= 10:
                    break
                started = time.monotonic()
                shuffled = torch.randperm(len(indexes["train"])).tolist()
                order = [indexes["train"][i] for i in shuffled]
                model.train()
                train_loss = 0.0
                for offset in range(0, len(order), 64):
                    ix = order[offset:offset + 64]
                    x, labels = batch(records, ix, graphs, torch)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x), labels)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    train_loss += float(loss.detach()) * len(ix)
                    if offset and offset % (64 * 300) == 0:
                        print(json.dumps({"event": "training", "seed": seed, "epoch": epoch + 1,
                                          "samples": offset, "wall_s": time.monotonic() - started}), flush=True)
                dev_ce, weighted_ce, predictions = evaluate(model, records, indexes["val"], graphs, torch)
                improved = dev_ce < best
                if improved:
                    best, best_epoch = dev_ce, epoch
                    atomic_torch(directory / "best.pt", {"model": model.state_dict(), "request": request,
                                 "epoch": epoch, "dev_ce_batch_mean": dev_ce, "dev_ce_sample_mean": weighted_ce}, torch)
                    write_csv(directory / "DEV_predictions.csv", predictions)
                # Installed Lightning EarlyStopping's native default is sealed;
                # ModelCheckpoint itself still retains every strict CE improvement.
                if dev_ce < early_best - protocol["recipe"]["early_stopping_min_delta"]:
                    early_best, bad_epochs = dev_ce, 0
                else:
                    bad_epochs += 1
                scheduler.step()
                record = {"epoch": epoch + 1, "seed": seed, "train_ce": train_loss / len(order),
                          "dev_ce_batch_mean": dev_ce, "dev_ce_sample_mean": weighted_ce,
                          "selected": improved, "patience_used": bad_epochs, "wall_s": time.monotonic() - started}
                with (directory / "epochs.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                atomic_torch(directory / "last.pt", {"request": request, "model": model.state_dict(),
                             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(torch),
                             "epoch": epoch, "best": best, "early_best": early_best,
                             "bad_epochs": bad_epochs, "best_epoch": best_epoch}, torch)
                print(json.dumps({"event": "epoch_complete", **record}), flush=True)
            del graphs
            a.signed(directory / "training_complete.json", {"status": "COMPLETE", "request": request,
                     "best_epoch": best_epoch + 1, "best_dev_ce": best, "best_sha256": a.sha(directory / "best.pt"),
                     "DEV_predictions_sha256": a.sha(directory / "DEV_predictions.csv"),
                     "stop_rule": "source DEV CE patience10 or max50", "external_selection": False})
        done = a.frozen.signed_read(directory / "training_complete.json", "result_sha256")
        if done["request"] != request or a.sha(directory / "best.pt") != done["best_sha256"]:
            raise ValueError("selected baseline checkpoint identity changed")
        state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        frozen_infer("source", model, records, vocab, directory, done["best_sha256"], torch, seed)
        del records
        available = [name for name in ("mit", "formai") if (a.DATA / (name + ".jsonl.gz")).exists()]
        for dataset in available:
            frozen_infer(dataset, model, a.load_records(dataset), vocab, directory, done["best_sha256"], torch, seed)
        a.signed(directory / "complete.json", {"status": "COMPLETE", "seed": seed,
                 "training_sha256": a.sha(directory / "training_complete.json"),
                 "inference_receipts": {name: a.sha(directory / name / "receipt.json") for name in ["test"] + available}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["accept", "train"])
    parser.add_argument("--seed", type=int, choices=[11, 29, 47], default=11)
    parser.add_argument("--threads", type=int, choices=[4, 8], default=4)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("launch with PYTHONHASHSEED=0 for reproducible native Word2Vec initialization")
    a.OUTPUT.mkdir(parents=True, exist_ok=True)
    (acceptance if args.action == "accept" else train)(args.seed, args.threads)


if __name__ == "__main__":
    main()
