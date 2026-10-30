#!/usr/bin/env python3
"""Frozen paper inference; no training, preprocessing changes, or source mutation.

All runs first pass the same first 64 historical family DEV rows. Complete TEST
uses original cached inputs, fixed checkpoint selection and threshold 0.5.
SQLite primary keys and a process lock prevent duplicate committed scores.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import gzip
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics  # Import before historical modules add their paths.
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PACKAGE = WORKSPACE / "transformer_gru_v76_v2"
STORE = PACKAGE / "src/research_project/ubuntu_local_research_store"
OUT = HERE / "paper_frozen"
CACHE = PACKAGE / "external/overflowgen/joint_graph_study_v1"
SELECTED = STORE / "hybrid_numeric_semantic_v59/baseline_dev60"
REGISTRY = PACKAGE / "docs/comparison_recovery/asset_manifest.json"
ALIASES = {
    "v70": "liquid_transfer_frozen", "vd": "vd",
    "v71": "residual_transfer_frozen", "v76": "no_refiner_frozen",
    "v75": "fixed_encoder_no_dropout", "v52": "numeric_all_visible",
    "semantic": "semantic", "numeric": "numeric", "legacy_joint": "legacy_joint",
}
SEEDS = (11, 29, 47)
PARITY_ATOL = 2e-5  # Fixed before TEST: CPU/GPU FP32 drift, plus zero class flips.
sys.path.insert(0, str(HERE))
import frozen_evaluate as frozen

sha = frozen.sha
read = lambda path: json.loads(Path(path).read_text())


def signed_write(path, value):
    frozen.atomic_json(path, {**value, "result_sha256": frozen.digest(value)})


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def runtime(threads, device):
    sys.path.insert(0, str(PACKAGE / "scripts"))
    from portable_runtime import configure_imports
    configure_imports()
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA unavailable; select CPU explicitly")
    return np, torch, torch.device(device)


def code_identity():
    # Include dynamically executed historical models, even if not sys.modules.
    paths = {Path(__file__), PACKAGE / "scripts/check_comparison_assets.py",
             PACKAGE / "scripts/portable_runtime.py", HERE / "frozen_evaluate.py",
             HERE / "mit_baseline_frozen_evaluate.py", HERE / "mit_numeric_ood.py"}
    for group in ["adaptive_graph_liquid_v70", "adaptive_graph_frozen_control_v71",
                  "no_refiner_frozen_v76", "frozen_encoder_dropout_v75", "full_overflowgen_v52"]:
        paths.update((STORE / group).glob("*.py"))
    paths.update((STORE / "overflow_interfaces_v51/overflow51").glob("*.py"))
    paths.update((CACHE / "implementation_family_isolated_v2").glob("*.py"))
    return {str(p.relative_to(WORKSPACE)): sha(p) for p in sorted(paths)}


def registry_run(method, seed):
    runs = [r for r in read(REGISTRY)["runs"] if r["scope"] == "overflowgen_family"
            and r["arm"] == ALIASES[method] and r["seed"] == seed]
    if len(runs) != 1:
        raise ValueError("expected exactly one recovered source checkpoint")
    run = runs[0]
    if run["group"] not in {"adaptive_graph_liquid_v70", "full_overflowgen_v52",
                            "adaptive_graph_frozen_control_v71", "no_refiner_frozen_v76",
                            "frozen_encoder_dropout_v75"}:
        raise ValueError("out-of-scope model group")
    return run


def model_identity(run):
    result = {name + "_sha256": sha(PACKAGE / run[name])
              for name in ("checkpoint", "config", "completion")}
    if result["checkpoint_sha256"] != run["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    return {**result, "asset_id": run["id"], "checkpoint": run["checkpoint"],
            "best_epoch": read(PACKAGE / run["completion"])["best_epoch"]}


def build_model(run, torch, np, device):
    # Reuse the audited historical constructor and restricted checkpoint loader.
    # That audit's top-level CPU selection is restored without modifying its file.
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    import check_comparison_assets as assets
    if previous is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = previous
    cfg = read(PACKAGE / run["config"])
    model = assets.make_historical(run, cfg)
    state = assets.safe_load(PACKAGE / run["checkpoint"], torch, np)
    expected, actual = model.state_dict(), state["model"]
    if set(expected) != set(actual) or any(expected[k].shape != actual[k].shape
            or expected[k].dtype != actual[k].dtype for k in expected):
        raise ValueError("state schema/dtype mismatch")
    if state["config"] != cfg["config"] or state["manifest_sha256"] != cfg.get("manifest", cfg.get("manifest_sha256")):
        raise ValueError("checkpoint/config/data identity mismatch")
    offset = int(run["group"] == "full_overflowgen_v52")
    if state["earlystop"]["best_epoch"] + offset != read(PACKAGE / run["completion"])["best_epoch"]:
        raise ValueError("checkpoint epoch mismatch")
    model.load_state_dict(actual, strict=True)
    if sum(p.numel() for p in model.parameters()) != cfg["parameters"]:
        raise ValueError("parameter count mismatch")
    del state, actual
    return model.eval().to(device)


class SourceData:
    def __init__(self, np):
        self.np = np
        receipt = read(CACHE / "CACHE_RECEIPT.json")
        for name, expected in receipt["cache_hashes"].items():
            if sha(CACHE / name) != expected:
                raise ValueError("original full cache mismatch: " + name)
        self.arrays = {k: np.load(CACHE / (k + ".npy"), mmap_mode="r", allow_pickle=False)
                       for k in frozen.KEYS + ["labels", "family_split"]}
        metadata = PACKAGE / "assets/comparisons/overflowgen/overflow_interfaces_v51/output/row_metadata.jsonl.gz"
        audit = read(WORKSPACE / "reproducibility/data_split_check.json")
        if sha(metadata) != audit["metadata_sha256"]:
            raise ValueError("source metadata identity mismatch")
        with gzip.open(metadata, "rt") as stream:
            self.metadata = [json.loads(line) for line in stream]
        if len(self.metadata) != 117696 or [m["row"] for m in self.metadata] != list(range(117696)):
            raise ValueError("source row identity mismatch")
        self.rows = {}
        for split in ("val", "test"):
            reference = audit["frozen_row_lists"][split]
            path = WORKSPACE / reference["path"]
            if sha(path) != reference["sha256"]:
                raise ValueError("frozen row list changed")
            rows = [int(v) for v in path.read_text().splitlines()]
            if rows != np.flatnonzero(self.arrays["family_split"] == (1 if split == "val" else 2)).tolist():
                raise ValueError("frozen split/cache mismatch")
            if any(self.metadata[r]["family_split"] != split or self.metadata[r]["label"] != int(self.arrays["labels"][r]) for r in rows):
                raise ValueError("frozen metadata/label mismatch")
            self.rows[split] = rows
        if (len(self.rows["test"]), sum(self.metadata[r]["label"] for r in self.rows["test"])) != (11577, 5015):
            raise ValueError("TEST denominator changed")
        self.identity = {"cache_receipt_sha256": sha(CACHE / "CACHE_RECEIPT.json"),
                         "array_sha256": receipt["cache_hashes"], "metadata_sha256": sha(metadata),
                         "frozen_row_lists": audit["frozen_row_lists"],
                         "split_audit_sha256": sha(WORKSPACE / "reproducibility/data_split_check.json")}
        # Verify identical original selected/full inputs for the fixed parity batch.
        selected = read(SELECTED / "MANIFEST60.json")
        selected_rows = np.load(SELECTED / "row.npy", allow_pickle=False)
        if sha(SELECTED / "row.npy") != selected["array_sha256"]["row.npy"]:
            raise ValueError("selected row map changed")
        lookup = {int(r): i for i, r in enumerate(selected_rows)}
        dev = self.rows["val"][:64]
        indices = [lookup[r] for r in dev]
        for key in frozen.KEYS:
            path = SELECTED / (key + ".npy")
            if sha(path) != selected["array_sha256"][path.name]:
                raise ValueError("selected cache changed: " + key)
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            if a.dtype != self.arrays[key].dtype or not np.array_equal(a[indices], self.arrays[key][dev]):
                raise ValueError("selected/full DEV tensor parity failed: " + key)
        self.identity["selected_manifest_sha256"] = sha(SELECTED / "MANIFEST60.json")

    def batch(self, rows, arm, torch, device):
        if arm == "vd":
            from overflow51.overflowgen import original_vd_batch
            return {k: v.to(device) for k, v in original_vd_batch(self.arrays, rows).items()}
        x = {k: torch.from_numpy(self.np.array(self.arrays[k][rows], copy=True)).to(device) for k in frozen.KEYS}
        length, nodes = int(x["line_count"].max()), int(x["node_count"].max())
        for key in frozen.LINE_KEYS:
            x[key] = x[key][:, :length]
        for key in frozen.NODE_KEYS:
            x[key] = x[key][:, :nodes]
        x["edges"] = x["edges"][:, :length, :length]
        # Match historical Selected.batch: only bit-input models cast numbers.
        if arm not in {"semantic", "numeric", "legacy_joint"}:
            x["numbers"] = x["numbers"].long()
        return x


def forward(model, x, torch):
    with torch.inference_mode():
        result = model(x)
    logits = result["logits"]
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("nonfinite/invalid logits")
    risk = logits.softmax(-1)[:, 1]
    return risk.cpu().tolist(), logits.cpu().tolist()


def parity(model, method, run, source, context, np, torch, device, extended=False):
    reference = (PACKAGE / run["config"]).parent / "DEV_predictions.csv.gz"
    request = {"schema": "paper-fixed-dev-parity-v1", "method": method, "seed": run["seed"],
               "model": context["model"], "code": context["code"], "runtime": context["runtime"],
               "source": source.identity, "reference_sha256": sha(reference),
               "rows": source.rows["val"][:64], "absolute_probability_tolerance": PARITY_ATOL,
               "required_class_flips": 0, "extended_numeric_assertion": extended}
    run_id = frozen.digest(request)
    directory = OUT / "parity" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "receipt.json"
    if receipt_path.exists():
        receipt = frozen.signed_read(receipt_path, "result_sha256")
        if receipt["request"] != request or receipt["status"] != "PASS":
            raise ValueError("parity receipt mismatch")
        return {"path": str(receipt_path), "sha256": sha(receipt_path)}
    with gzip.open(reference, "rt") as stream:
        rows = list(csv.DictReader(stream))[:64]
    if [int(r["row"]) for r in rows] != request["rows"]:
        raise ValueError("DEV fixed rows do not match historical saved predictions")
    x = source.batch(request["rows"], run["arm"], torch, device)
    started = time.monotonic()
    scores, logits = forward(model, x, torch)
    comparisons = []
    for row, value, logit in zip(rows, scores, logits):
        original = float(row["score"])
        if int(row["label"]) != source.metadata[int(row["row"])]["label"]:
            raise ValueError("DEV label mismatch")
        comparisons.append({"row": int(row["row"]), "historical_probability": original,
                            "probability": value, "absolute_difference": abs(original - value),
                            "class_flip": int((original >= .5) != (value >= .5)),
                            "logit0": logit[0], "logit1": logit[1]})
    elapsed = time.monotonic() - started
    maximum = max(r["absolute_difference"] for r in comparisons)
    flips = sum(r["class_flip"] for r in comparisons)
    passed = maximum <= PARITY_ATOL and flips == 0
    write_csv(directory / "predictions.csv", comparisons)
    signed_write(receipt_path, {"status": "PASS" if passed else "FAIL", "request": request,
                               "request_sha256": run_id, "max_absolute_difference": maximum,
                               "class_flips": flips, "forward_wall_s": elapsed,
                               "samples_per_second": 64 / elapsed,
                               "predictions_sha256": sha(directory / "predictions.csv")})
    print(json.dumps({"event": "parity", "method": method, "seed": run["seed"],
                      "extended": extended, "pass": passed, "max_difference": maximum,
                      "class_flips": flips, "seconds": elapsed}), flush=True)
    if not passed:
        raise ValueError("fixed DEV parity failed; TEST is prohibited")
    return {"path": str(receipt_path), "sha256": sha(receipt_path)}


def external_inputs(dataset):
    import mit_baseline_frozen_evaluate as mit
    if dataset == "mit":
        from external_eval.portable_inputs import load_mit
        data = load_mit()
        return {"records": sorted(data["extended_slices"], key=lambda r: r["slice_id"]),
                "ids": data["extended_ids"], "scenes": data["scenes"],
                "common": data["strict_ids"], "identity": data["strict"]["identity"]}
    membership = HERE / "static_baselines/results/cohort_membership.csv"
    with membership.open() as stream:
        common = {r["scene_id"] for r in csv.DictReader(stream)
                  if r["dataset"] == dataset and r["cohort"] == "common_native_all9"}
    if dataset == "mit":
        data = mit.load_inputs()
        records, ids, scenes = data["extended_slices"], data["extended_ids"], data["scenes"]
        if common != data["strict_ids"]:
            raise ValueError("MIT796 membership changed")
        identity = {"protocol_sha256": sha(mit.FINAL_PROTOCOL),
                    "strict_identity": data["strict"]["identity"],
                    "recovered_receipt": data["recovered"]["receipt"]}
        expected = (1059, 1578, 796)
    else:
        directory = HERE / "formai_pool/adaptation/test_inputs"
        data = frozen.load_prepared(directory)
        scenes = {r["scene_id"]: r for r in data["scenes"]}
        ids = {s for s, r in scenes.items() if mit.processable(r)}
        records = [r for r in data["slices"] if r["owner_scene_ids"][0] in ids]
        identity = data["identity"]
        expected = (167, 2709, 161)
    if (len(ids), len(records), len(common)) != expected or not common <= ids:
        raise ValueError("external frozen denominator changed")
    if {r["owner_scene_ids"][0] for r in records} != ids or len({r["slice_id"] for r in records}) != len(records):
        raise ValueError("external ownership/coverage mismatch")
    return {"records": sorted(records, key=lambda r: r["slice_id"]), "ids": ids,
            "scenes": scenes, "common": common,
            "identity": {"prepared": identity, "cohort_membership_sha256": sha(membership),
                         "scene_ids_sha256": frozen.digest(sorted(ids)),
                         "common_ids_sha256": frozen.digest(sorted(common))}}


def ledger(path, request_hash):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute("""CREATE TABLE IF NOT EXISTS scores(
        sample_id TEXT PRIMARY KEY, row INTEGER, scene_id TEXT NOT NULL, risk REAL NOT NULL,
        logit0 REAL NOT NULL, logit1 REAL NOT NULL)""")
    prior = connection.execute("SELECT value FROM metadata WHERE key='request_sha256'").fetchone()
    if prior and prior[0] != request_hash:
        raise ValueError("ledger belongs to different request")
    connection.execute("INSERT OR IGNORE INTO metadata VALUES('request_sha256',?)", (request_hash,))
    connection.commit()
    return connection


def execute(args, method, seed, source, np, torch, device):
    run = registry_run(method, seed)
    context = {"model": model_identity(run), "code": code_identity(),
               "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "device": str(device), "threads": args.threads,
                           "precision": "FP32", "deterministic_algorithms": True, "tf32": False,
                           "mkldnn_enabled": torch.backends.mkldnn.enabled}}
    model = build_model(run, torch, np, device)
    gate = parity(model, method, run, source, context, np, torch, device)
    if args.action == "parity":
        return
    ext = None
    if args.dataset != "overflowgen":
        if method != "v70":
            raise ValueError("external evaluation supports the CoNC v70 model")
        ext = external_inputs(args.dataset)
        if args.dataset == "mit":
            from mit_numeric_ood import patch_numeric_range
            patch_numeric_range(model, torch)
            gate = parity(model, method, run, source, context, np, torch, device, extended=True)
        records = ext["records"]
        ids = [r["slice_id"] for r in records]
        input_identity = ext["identity"]
    else:
        records = source.rows["test"]
        ids = ["og24455320:" + source.metadata[r]["original_split"] + ":" + str(source.metadata[r]["original_sample_id"]) for r in records]
        input_identity = source.identity
    request = {"schema": "paper-frozen-inference-v1", "dataset": args.dataset,
               "method": method, "historical_arm": run["arm"], "seed": seed, **context,
               "asset_registry_sha256": sha(REGISTRY), "inputs": input_identity,
               "parity": gate, "sample_count": len(records), "sample_ids_sha256": frozen.digest(ids),
               "batch_size": args.batch_size, "threshold": .5,
               "aggregation": "slice" if ext is None else "max_slice_risk",
               "numeric_range": [0, 32767 if args.dataset == "mit" else 999],
               "training_steps": 0, "optimizer_constructed": False}
    run_id = frozen.digest(request)
    directory = OUT / "runs" / f"{args.dataset}_{method}_s{seed}_{run_id[:16]}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / "request.json"
        if path.exists() and read(path) != request:
            raise ValueError("immutable request changed")
        if not path.exists():
            frozen.atomic_json(path, request)
        complete = directory / "run_receipt.json"
        if complete.exists():
            receipt = frozen.signed_read(complete, "result_sha256")
            if receipt["request_sha256"] != run_id or receipt["status"] != "COMPLETE":
                raise ValueError("invalid completion")
            for name, expected in receipt["artifacts"].items():
                if sha(directory / name) != expected:
                    raise ValueError("completed artifact changed")
            print(json.dumps({"event": "SKIP_VERIFIED", "directory": str(directory)}), flush=True)
            return
        connection = ledger(directory / "scores.sqlite3", run_id)
        committed = {r[0] for r in connection.execute("SELECT sample_id FROM scores")}
        if not committed <= set(ids):
            raise ValueError("ledger contains unexpected sample")
        pending = [(sample_id, record) for sample_id, record in zip(ids, records) if sample_id not in committed]
        started, batches = time.monotonic(), 0
        print(json.dumps({"event": "start", "pid": os.getpid(), "run_id": run_id,
                          "directory": str(directory), "committed": len(committed), "pending": len(pending)}), flush=True)
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset:offset + args.batch_size]
            if ext is None:
                x = source.batch([r for _, r in batch], run["arm"], torch, device)
            else:
                x = frozen.tensor_batch([r for _, r in batch], device, np, torch)
            scores, logits = forward(model, x, torch)
            if len(scores) != len(batch):
                raise ValueError("batch coverage mismatch")
            entries = [(sample_id, record if ext is None else None,
                        sample_id if ext is None else record["owner_scene_ids"][0],
                        probability, logit[0], logit[1])
                       for (sample_id, record), probability, logit in zip(batch, scores, logits)]
            with connection:
                connection.executemany("INSERT INTO scores VALUES(?,?,?,?,?,?)", entries)
            batches += 1
            done = len(committed) + min(offset + args.batch_size, len(pending))
            if batches == 1 or batches % 10 == 0:
                print(json.dumps({"event": "progress", "method": method, "seed": seed,
                                  "dataset": args.dataset, "committed": done,
                                  "total": len(ids), "wall_s": time.monotonic() - started}), flush=True)
            if args.max_batches and batches >= args.max_batches:
                break
        connection.row_factory = sqlite3.Row
        stored = [dict(r) for r in connection.execute("SELECT * FROM scores ORDER BY sample_id")]
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
        progress = {"status": "COMPLETE" if len(stored) == len(records) else "IN_PROGRESS",
                    "run_id": run_id, "pid": os.getpid(), "committed": len(stored),
                    "total": len(records), "invocation_wall_s": time.monotonic() - started,
                    "new_samples": len(stored) - len(committed)}
        frozen.atomic_json(directory / "progress.json", progress)
        if len(stored) != len(records):
            print(json.dumps(progress), flush=True)
            return
        predictions = []
        if ext is None:
            for row in sorted(stored, key=lambda r: r["row"]):
                metadata = source.metadata[row["row"]]
                predictions.append({"method": method, "seed": seed, "sample_id": row["sample_id"],
                                    "row": row["row"], "family_id": metadata["family_id"],
                                    "source_case": metadata["source_case"], "label": metadata["label"],
                                    "risk": row["risk"], "prediction": int(row["risk"] >= .5),
                                    "margin": row["logit1"] - row["logit0"],
                                    "logit0": row["logit0"], "logit1": row["logit1"]})
            write_csv(directory / "predictions.csv", predictions)
            names = ["predictions.csv"]
        else:
            import mit_baseline_frozen_evaluate as mit
            slices = [{"slice_id": r["sample_id"], "scene_id": r["scene_id"], "risk": r["risk"],
                       "logit0": r["logit0"], "logit1": r["logit1"]} for r in stored]
            predictions = mit.aggregate_scenes(slices, ext["ids"], ext["scenes"], method, seed)
            for row in predictions:
                row["in_common_native"] = row["scene_id"] in ext["common"]
            write_csv(directory / "slice_scores.csv", slices)
            write_csv(directory / "scene_predictions.csv", predictions)
            write_csv(directory / "common_scene_predictions.csv", [r for r in predictions if r["in_common_native"]])
            names = ["slice_scores.csv", "scene_predictions.csv", "common_scene_predictions.csv"]
        signed_write(complete, {"status": "COMPLETE", "request_sha256": run_id,
                                "method": method, "seed": seed, "dataset": args.dataset,
                                "sample_count": len(stored), "prediction_units": len(predictions),
                                "training_steps": 0, "optimizer_constructed": False,
                                "artifacts": {name: sha(directory / name) for name in names + ["request.json", "scores.sqlite3"]}})
        print(json.dumps({"event": "COMPLETE", "directory": str(directory), **progress}), flush=True)
    del model
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["parity", "run"])
    parser.add_argument("--dataset", choices=["overflowgen", "mit", "formai"], default="overflowgen")
    parser.add_argument("--method", choices=["all", *ALIASES], default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Commit only this many new batches; resume same run without the limit")
    args = parser.parse_args()
    if args.batch_size != 64 or not 1 <= args.threads <= 8 or args.max_batches < 0:
        raise ValueError("fixed batch size64, CPU threads1..8, nonnegative batch limit required")
    if args.dataset != "overflowgen" and args.method != "v70":
        raise ValueError("external scope requires --method v70")
    OUT.mkdir(parents=True, exist_ok=True)
    np, torch, device = runtime(args.threads, args.device)
    source = SourceData(np)
    for method in ALIASES if args.method == "all" else [args.method]:
        for seed in SEEDS if args.seed is None else [args.seed]:
            execute(args, method, seed, source, np, torch, device)


if __name__ == "__main__":
    main()
