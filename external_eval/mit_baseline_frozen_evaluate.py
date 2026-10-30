#!/usr/bin/env python3
"""Evaluate recovered OverflowGen baseline checkpoints on the frozen MIT protocol.

The formal common comparison uses the unchanged native 0..999 input domain
(796 MIT scenes).  The numeric-only graph model is additionally evaluated on
the frozen 0..32767 extension (1,059 scenes), because its forward path consumes
the existing 32-bit node channel and does not index the line-level decimal
table.  Semantic/joint and VD models are never silently extended beyond their
recorded 0..999 tables.

No optimizer is constructed.  Per-run SQLite ledgers make interrupted
inference resumable without repeating committed slice scores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PACKAGE = WORKSPACE / "transformer_gru_v76_v2"
OUTPUT = HERE / "mit_baselines"
FINAL_PROTOCOL = HERE / "mit_final_external_protocol.json"
ASSET_MANIFEST = PACKAGE / "docs/comparison_recovery/asset_manifest.json"
ASSET_VERIFICATION = PACKAGE / "docs/comparison_recovery/verification.json"
OVERLAY = HERE / "prepared_mit_condition_overlay"
NUMERIC = HERE / "mit_numeric_ood"
MAIN_PREDICTIONS = NUMERIC / "combined_scene_predictions.csv"
ALL_STATUS = NUMERIC / "all_samples_protocol_predictions.csv"
SEEDS = (11, 29, 47)
BASELINES = ("semantic", "numeric", "legacy_joint", "vd")
THRESHOLD = 0.5
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261018
# These completed ledgers were produced before two statistics-only edits
# (vectorized bootstrap and margin-based ranking metrics).  The inference
# functions and run-request fields were unchanged.  Compatibility is limited
# to this exact predecessor hash and still requires every other request field
# and every artifact hash to match.
STATISTICS_ONLY_PREDECESSOR = "8f82042b539fbef91336aaa88acafe95ed88935feeda18dd7ab3239f704506d7"

sys.path.insert(0, str(HERE))
import frozen_evaluate as frozen


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def signed_write(path: Path, value: dict) -> None:
    value = dict(value)
    value.pop("result_sha256", None)
    value["result_sha256"] = frozen.digest(value)
    atomic_json(path, value)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else ["status"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def processable(scene: dict) -> bool:
    inventory = scene.get("slice_inventory") or []
    return bool(inventory and scene.get("slice_inventory_complete")
                and scene.get("frontend_processable")
                and all(row.get("status") == "encoded" for row in inventory))


def load_inputs() -> dict:
    """Load the portable frozen MIT tensors and verify their release hashes."""
    from external_eval.portable_inputs import load_mit
    return load_mit()


def configure_runtime():
    sys.path.insert(0, str(PACKAGE / "scripts"))
    from portable_runtime import configure_imports
    configure_imports()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    return torch


def safe_load(path: Path, torch):
    from numpy.core.multiarray import _reconstruct
    allow = [_reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
    with torch.serialization.safe_globals(allow):
        return torch.load(path, map_location="cpu", weights_only=True)


def select_runs() -> list[dict]:
    manifest = json.loads(ASSET_MANIFEST.read_text())
    selected = [row for row in manifest["runs"]
                if row["scope"] == "overflowgen_family" and row["group"] == "full_overflowgen_v52"
                and row["arm"] in BASELINES]
    if {(row["arm"], row["seed"]) for row in selected} != {(m, s) for m in BASELINES for s in SEEDS}:
        raise ValueError("expected 12 complete OverflowGen baseline checkpoints")
    verification = json.loads(ASSET_VERIFICATION.read_text())
    passed = {row["id"] for row in verification["results"] if row.get("status") == "PASS"}
    if any(row["id"] not in passed for row in selected):
        raise ValueError("a selected baseline lacks prior safe-load verification")
    return sorted(selected, key=lambda row: (BASELINES.index(row["arm"]), row["seed"]))


def build_model(run: dict, torch, device):
    checkpoint = PACKAGE / run["checkpoint"]
    config_path = PACKAGE / run["config"]
    completion_path = PACKAGE / run["completion"]
    if sha(checkpoint) != run["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    config = json.loads(config_path.read_text())
    completion = json.loads(completion_path.read_text())
    if config["arm"] != run["arm"] or config["track"] != "family":
        raise ValueError("baseline config identity mismatch")
    from overflow51.full_data import make_model
    model = make_model(run["arm"], "family")
    state = safe_load(checkpoint, torch)
    if state["config"] != config["config"] or state["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("checkpoint/config/data identity mismatch")
    if state["earlystop"]["best_epoch"] + 1 != completion["best_epoch"]:
        raise ValueError("selected epoch mismatch")
    expected = model.state_dict()
    actual = state["model"]
    if set(expected) != set(actual) or any(expected[key].shape != actual[key].shape for key in expected):
        raise ValueError("checkpoint state schema mismatch")
    model.load_state_dict(actual, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != config["parameters"]:
        raise ValueError("parameter count mismatch")
    model.eval().to(device)
    return model, {"checkpoint_sha256": run["sha256"], "config_sha256": sha(config_path),
                   "completion_sha256": sha(completion_path), "best_epoch": completion["best_epoch"]}


def vd_batch(x: dict, torch) -> dict:
    counts = x["line_count"].long()
    length = x["tokens"].shape[1]
    valid = torch.arange(length, device=counts.device)[None, :] < counts[:, None]
    return {"token_ids": x["tokens"].long()[valid], "number_ids": x["numbers"].long()[valid],
            "number_mark": x["marks"].long()[valid], "line_counts": counts,
            "key_padding_mask": ~valid, "key_lines": x["key"].long()}


def open_ledger(path: Path, request_sha256: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("""CREATE TABLE IF NOT EXISTS scores (
        slice_id TEXT PRIMARY KEY, scene_id TEXT NOT NULL, risk REAL NOT NULL,
        logit0 REAL NOT NULL, logit1 REAL NOT NULL)""")
    existing = connection.execute("SELECT value FROM metadata WHERE key='request_sha256'").fetchone()
    if existing and existing[0] != request_sha256:
        raise ValueError("existing ledger belongs to a different frozen request")
    connection.execute("INSERT OR IGNORE INTO metadata VALUES ('request_sha256', ?)", (request_sha256,))
    connection.commit()
    return connection


def aggregate_scenes(rows: list[dict], scene_ids: set[str], scenes: dict, method: str, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    if set(grouped) != scene_ids:
        raise ValueError("scene aggregation coverage mismatch")
    result = []
    for scene_id in sorted(scene_ids):
        best = max(grouped[scene_id], key=lambda row: row["risk"])
        scene = scenes[scene_id]
        result.append({"method": method, "seed": seed, "scene_id": scene_id,
                       "family_id": scene["family_id"], "variant": scene["variant"],
                       "label": scene["label"], "risk": best["risk"],
                       "margin": best["logit1"] - best["logit0"],
                       "prediction": int(best["risk"] >= THRESHOLD),
                       "max_slice_id": best["slice_id"], "slice_count": len(grouped[scene_id])})
    return result


def run_one(run: dict, context: dict, torch, device, smoke: bool = False) -> dict:
    method, seed = run["arm"], run["seed"]
    scope = "extended_0_32767" if method == "numeric" else "strict_0_999"
    records = context["extended_slices"] if method == "numeric" else context["strict_slices"]
    scene_ids = context["extended_ids"] if method == "numeric" else context["strict_ids"]
    model, identity = build_model(run, torch, device)
    if smoke:
        batch = records[:2]
        x = frozen.tensor_batch(batch, device, np, torch)
        with torch.inference_mode():
            output = model(vd_batch(x, torch) if method == "vd" else x)
        logits = output["logits"]
        if logits.shape != (2, 2) or not torch.isfinite(logits).all():
            raise ValueError("baseline smoke output invalid")
        return {"method": method, "seed": seed, "scope": scope, "smoke": "PASS"}
    run_request = {"schema": "mit-overflowgen-baseline-inference-v1", "script_sha256": sha(__file__),
        "final_protocol_sha256": sha(FINAL_PROTOCOL), "asset_manifest_sha256": sha(ASSET_MANIFEST),
        "asset_verification_sha256": sha(ASSET_VERIFICATION), "method": method, "seed": seed,
        "scope": scope, "threshold": THRESHOLD, "scene_aggregation": "maximum slice risk",
        "input_receipts": {"strict": sha(OVERLAY / "preparation_receipt.json"),
                           "numeric_extension": sha(NUMERIC / "preparation_receipt.json") if method == "numeric" else None},
        "input_counts": {"scenes": len(scene_ids), "slices": len(records)}, **identity,
        "model_parameters_changed": False, "training_steps": 0,
        "numeric_contract": ("existing 32-bit node bits; line decimal channel unused by numeric arm"
                             if method == "numeric" else "native recorded 0..999 tables unchanged")}
    request_hash = frozen.digest(run_request)
    for prior_directory in sorted((OUTPUT / "runs").glob(f"{method}_s{seed}_*")):
        prior_request_path = prior_directory / "request.json"
        prior_receipt_path = prior_directory / "run_receipt.json"
        if not prior_request_path.exists() or not prior_receipt_path.exists():
            continue
        prior_request = json.loads(prior_request_path.read_text())
        comparable_prior = {key: value for key, value in prior_request.items() if key != "script_sha256"}
        comparable_now = {key: value for key, value in run_request.items() if key != "script_sha256"}
        if prior_request.get("script_sha256") == STATISTICS_ONLY_PREDECESSOR and comparable_prior == comparable_now:
            receipt = frozen.signed_read(prior_receipt_path, "result_sha256")
            if receipt["request_sha256"] != frozen.digest(prior_request):
                raise ValueError("predecessor run receipt mismatch")
            for name, expected in receipt["artifacts"].items():
                if sha(prior_directory / name) != expected:
                    raise ValueError("predecessor baseline artifact changed")
            del model
            gc.collect(); torch.cuda.empty_cache()
            return {"method": method, "seed": seed, "scope": scope, "directory": str(prior_directory),
                    "request_sha256": receipt["request_sha256"], "status": "SKIP_VERIFIED_PREDECESSOR"}
    directory = OUTPUT / "runs" / f"{method}_s{seed}_{request_hash[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != run_request:
        raise ValueError("existing run request mismatch")
    if not request_path.exists():
        atomic_json(request_path, run_request)
    receipt_path = directory / "run_receipt.json"
    if receipt_path.exists():
        receipt = frozen.signed_read(receipt_path, "result_sha256")
        if receipt["request_sha256"] != request_hash:
            raise ValueError("existing run receipt mismatch")
        for name, expected in receipt["artifacts"].items():
            if sha(directory / name) != expected:
                raise ValueError("completed baseline artifact changed")
        del model
        gc.collect(); torch.cuda.empty_cache()
        return {"method": method, "seed": seed, "scope": scope, "directory": str(directory),
                "request_sha256": request_hash, "status": "SKIP_VERIFIED"}
    connection = open_ledger(directory / "scores.sqlite3", request_hash)
    committed = {row[0] for row in connection.execute("SELECT slice_id FROM scores")}
    missing = [row for row in records if row["slice_id"] not in committed]
    started = time.monotonic()
    for start in range(0, len(missing), 64):
        batch = missing[start:start + 64]
        x = frozen.tensor_batch(batch, device, np, torch)
        with torch.inference_mode():
            output = model(vd_batch(x, torch) if method == "vd" else x)
        logits = output["logits"]
        risks = logits.softmax(-1)[:, 1]
        if logits.shape != (len(batch), 2) or not torch.isfinite(logits).all() or not torch.isfinite(risks).all():
            raise ValueError("invalid baseline output")
        values = [(row["slice_id"], row["owner_scene_ids"][0], float(risk), float(pair[0]), float(pair[1]))
                  for row, pair, risk in zip(batch, logits.cpu(), risks.cpu())]
        connection.executemany("INSERT INTO scores VALUES (?, ?, ?, ?, ?)", values)
        connection.commit()
        print(json.dumps({"method": method, "seed": seed, "committed": len(committed) + start + len(batch),
                          "total": len(records)}), flush=True)
    stored = [{"slice_id": row[0], "scene_id": row[1], "risk": row[2], "logit0": row[3], "logit1": row[4]}
              for row in connection.execute("SELECT slice_id,scene_id,risk,logit0,logit1 FROM scores ORDER BY slice_id")]
    connection.close()
    if len(stored) != len(records) or {row["slice_id"] for row in stored} != {row["slice_id"] for row in records}:
        raise ValueError("baseline ledger is incomplete")
    predictions = aggregate_scenes(stored, scene_ids, context["scenes"], method, seed)
    write_csv(directory / "slice_scores.csv", stored)
    write_csv(directory / "scene_predictions.csv", predictions)
    receipt = {"status": "COMPLETE", "request": run_request, "request_sha256": request_hash,
               "counts": {"slice_scores": len(stored), "scene_predictions": len(predictions)},
               "artifacts": {name: sha(directory / name) for name in ["slice_scores.csv", "scene_predictions.csv"]},
               "elapsed_seconds_this_invocation": time.monotonic() - started,
               "optimizer_constructed": False, "training_steps": 0}
    signed_write(receipt_path, receipt)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return {"method": method, "seed": seed, "scope": scope, "directory": str(directory),
            "request_sha256": request_hash, "status": "COMPLETE"}


def metric_values(rows: list[dict]) -> dict:
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 matthews_corrcoef, precision_score, recall_score, roc_auc_score)
    y = np.asarray([int(row["label"]) for row in rows])
    # Use the two-logit margin for ranking metrics.  It is monotone with the
    # positive softmax risk but retains ordering when float32 probabilities
    # saturate at exactly 0 or 1; this matches the frozen MIT implementation.
    score = np.asarray([float(row["margin"]) for row in rows])
    prediction = np.asarray([int(row["prediction"]) for row in rows])
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"n": len(rows), "positive": int(y.sum()), "negative": int((1-y).sum()),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "ap": float(average_precision_score(y, score)) if len(set(y)) == 2 else None,
            "mcc": float(matthews_corrcoef(y, prediction)) if len(set(y)) == 2 else None,
            "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def group_bootstrap(method_rows: dict[tuple[str, int], list[dict]], methods: list[str], reference: str | None = None) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    first = next(iter(method_rows.values()))
    groups = sorted({row["family_id"] for row in first})
    group_index = {group: index for index, group in enumerate(groups)}
    # One shared draw matrix preserves the paired comparison while avoiding
    # repeated general-purpose metric-library calls inside the bootstrap loop.
    draws = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    weights = np.zeros((BOOTSTRAP_REPLICATES, len(groups)), dtype=np.int16)
    for replicate, draw in enumerate(draws):
        weights[replicate] = np.bincount(draw, minlength=len(groups))
    distributions = {method: [] for method in methods}
    differences = {method: [] for method in methods if reference and method != reference}
    for method in methods:
        seed_f1 = []
        for seed in SEEDS:
            counts = np.zeros((len(groups), 3), dtype=np.int32)  # TP, FP, FN
            for row in method_rows[(method, seed)]:
                target, prediction = int(row["label"]), int(row["prediction"])
                at = group_index[row["family_id"]]
                counts[at, 0] += target == 1 and prediction == 1
                counts[at, 1] += target == 0 and prediction == 1
                counts[at, 2] += target == 1 and prediction == 0
            sampled = weights @ counts
            denominator = 2 * sampled[:, 0] + sampled[:, 1] + sampled[:, 2]
            seed_f1.append(np.divide(2 * sampled[:, 0], denominator,
                                     out=np.zeros(BOOTSTRAP_REPLICATES), where=denominator != 0))
        distributions[method] = np.mean(np.stack(seed_f1), axis=0)
    if reference:
        for method in differences:
            differences[method] = distributions[method] - distributions[reference]
    output = []
    for method in methods:
        values = distributions[method]
        output.append({"method": method, "metric": "f1", "comparison": "point",
                       "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                       "ci95_low": float(np.quantile(values, .025)),
                       "ci95_high": float(np.quantile(values, .975))})
    for method, values in differences.items():
        output.append({"method": method, "metric": "f1", "comparison": f"{method}-{reference}",
                       "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                       "ci95_low": float(np.quantile(values, .025)),
                       "ci95_high": float(np.quantile(values, .975))})
    return output


def paired_rows(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family_id"]].append(row)
    paired = [values for values in grouped.values()
              if {int(row["label"]) for row in values} == {0, 1}]
    full4 = [values for values in paired if len(values) == 4 and len({row["variant"] for row in values}) == 4]
    return {"paired_groups": len(paired),
            "paired_all_correct": sum(all(int(row["prediction"]) == int(row["label"]) for row in values) for values in paired),
            "paired_all_correct_rate": (sum(all(int(row["prediction"]) == int(row["label"]) for row in values) for values in paired) / len(paired)) if paired else None,
            "full4_groups": len(full4),
            "full4_all_correct": sum(all(int(row["prediction"]) == int(row["label"]) for row in values) for values in full4),
            "full4_all_correct_rate": (sum(all(int(row["prediction"]) == int(row["label"]) for row in values) for values in full4) / len(full4)) if full4 else None}


def summarize(context: dict, completed: list[dict]) -> dict:
    output_rows = []
    for run in completed:
        rows = read_csv(Path(run["directory"]) / "scene_predictions.csv")
        for row in rows:
            row["label"] = int(row["label"]); row["seed"] = int(row["seed"])
            row["risk"] = float(row["risk"]); row["prediction"] = int(row["prediction"])
        output_rows.extend(rows)
    main = read_csv(MAIN_PREDICTIONS)
    for row in main:
        row["label"] = int(row["label"]); row["seed"] = int(row["seed"])
        row["risk"] = float(row["risk"]); row["prediction"] = int(row["prediction"])
    strict_ids, extended_ids = context["strict_ids"], context["extended_ids"]
    all_rows = output_rows + main
    write_csv(OUTPUT / "scene_predictions.csv", output_rows)

    comparisons = []
    scope_methods = {"common_strict_796": ["v76", "v52", *BASELINES],
                     "extended_1059": ["v76", "v52", "numeric"]}
    ids_by_scope = {"common_strict_796": strict_ids, "extended_1059": extended_ids}
    lookup_rows = {}
    for scope, methods in scope_methods.items():
        wanted = ids_by_scope[scope]
        for method in methods:
            for seed in SEEDS:
                rows = [row for row in all_rows if row["method"] == method and row["seed"] == seed
                        and row["scene_id"] in wanted]
                if len(rows) != len(wanted) or len({row["scene_id"] for row in rows}) != len(wanted):
                    raise ValueError(f"missing comparison rows for {scope}/{method}/{seed}")
                values = metric_values(rows)
                comparisons.append({"scope": scope, "method": method, "seed": seed, **values,
                                    **paired_rows(rows)})
                lookup_rows[(scope, method, seed)] = rows
    write_csv(OUTPUT / "metrics_by_seed.csv", comparisons)

    summary = []
    metric_names = ["precision", "recall", "f1", "ap", "mcc", "auroc", "specificity"]
    for scope, methods in scope_methods.items():
        for method in methods:
            rows = [row for row in comparisons if row["scope"] == scope and row["method"] == method]
            item = {"scope": scope, "method": method, "seeds": len(rows), "n": rows[0]["n"],
                    "positive": rows[0]["positive"], "negative": rows[0]["negative"],
                    "all_positive_f1": 2 * rows[0]["positive"] / (rows[0]["n"] + rows[0]["positive"])}
            for name in metric_names:
                values = [row[name] for row in rows]
                item[name + "_mean"] = statistics.mean(values)
                item[name + "_sd"] = statistics.stdev(values)
            summary.append(item)
    write_csv(OUTPUT / "summary.csv", summary)

    deltas = []
    for baseline in BASELINES:
        for seed in SEEDS:
            left = next(row for row in comparisons if row["scope"] == "common_strict_796" and row["method"] == baseline and row["seed"] == seed)
            right = next(row for row in comparisons if row["scope"] == "common_strict_796" and row["method"] == "v76" and row["seed"] == seed)
            deltas.append({"scope": "common_strict_796", "method": baseline, "reference": "v76", "seed": seed,
                           **{name + "_difference": left[name] - right[name] for name in metric_names}})
    write_csv(OUTPUT / "method_differences.csv", deltas)

    bootstrap = []
    for scope, methods in scope_methods.items():
        method_rows = {(method, seed): lookup_rows[(scope, method, seed)] for method in methods for seed in SEEDS}
        for row in group_bootstrap(method_rows, methods, "v76"):
            bootstrap.append({"scope": scope, "bootstrap_unit": "MIT base case/family",
                              "bootstrap_seed": BOOTSTRAP_SEED, **row})
    write_csv(OUTPUT / "bootstrap_ci.csv", bootstrap)

    final_status = [row for row in read_csv(ALL_STATUS)
                    if row["protocol"] == "extended_0_32767" and row["method"] == "v76" and int(row["seed"]) == 11]
    if len(final_status) != 1164:
        raise ValueError("final status reference changed")
    final_by_id = {row["scene_id"]: row for row in final_status}
    coverage_rows, status_rows = [], []
    for method in BASELINES:
        passed = extended_ids if method == "numeric" else strict_ids
        for scene_id, scene in sorted(context["scenes"].items()):
            if scene_id in passed:
                status, stage, reason = "PASS", "", ""
            elif scene_id in context["recovered_ids"]:
                status, stage, reason = "UNKNOWN", "input_domain", "NUMBER_OUTSIDE_NATIVE_0_999"
            else:
                ref = final_by_id[scene_id]
                status, stage, reason = "UNKNOWN", ref["failure_stage"], ref["failure_reason"]
            status_rows.append({"method": method, "scene_id": scene_id, "family_id": scene["family_id"],
                                "variant": scene["variant"], "label": scene["label"],
                                "evaluation_status": status, "failure_stage": stage, "failure_reason": reason})
        coverage_rows.append({"method": method, "registered": 1164, "processable": len(passed),
                              "unknown": 1164-len(passed),
                              "processable_positive": sum(context["scenes"][scene_id]["label"] for scene_id in passed),
                              "processable_negative": sum(1-context["scenes"][scene_id]["label"] for scene_id in passed),
                              "input_domain": "32-bit node bits; line decimal unused" if method == "numeric" else "native 0..999"})
    write_csv(OUTPUT / "coverage.csv", coverage_rows)
    write_csv(OUTPUT / "all_scene_status.csv", status_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    strict_summary = [row for row in summary if row["scope"] == "common_strict_796"]
    order = ["v76", "v52", *BASELINES]
    rows = [next(row for row in strict_summary if row["method"] == method) for method in order]
    x = np.arange(len(rows)); width = .37
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x-width/2, [row["f1_mean"] for row in rows], width,
             yerr=[row["f1_sd"] for row in rows], label="F1", capsize=3)
    axis.bar(x+width/2, [row["auroc_mean"] for row in rows], width,
             yerr=[row["auroc_sd"] for row in rows], label="AUROC", capsize=3)
    axis.set_xticks(x, order, rotation=20, ha="right"); axis.set_ylim(0, 1)
    axis.set_ylabel("score (mean ± seed sample SD)")
    axis.set_title("MIT common native-domain comparison (796 scenes)")
    axis.legend(); axis.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(OUTPUT / "mit_baseline_common_metrics.png", dpi=180)
    plt.close(fig)

    artifacts = ["scene_predictions.csv", "metrics_by_seed.csv", "summary.csv", "method_differences.csv",
                 "bootstrap_ci.csv", "coverage.csv", "all_scene_status.csv", "mit_baseline_common_metrics.png"]
    result = {"status": "COMPLETE", "protocol_id": context["protocol"]["protocol_id"],
              "request": {"script_sha256": sha(__file__), "final_protocol_sha256": sha(FINAL_PROTOCOL),
                          "asset_manifest_sha256": sha(ASSET_MANIFEST), "asset_verification_sha256": sha(ASSET_VERIFICATION),
                          "main_predictions_sha256": sha(MAIN_PREDICTIONS), "threshold": THRESHOLD,
                          "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED},
              "counts": {"registered_scenes": 1164, "strict_common_scenes": 796,
                         "extended_numeric_scenes": 1059, "baseline_runs": len(completed)},
              "scope_rules": {"semantic": "native 0..999 only", "legacy_joint": "native 0..999 only",
                              "vd": "native DICE 0..999 only", "numeric": "1,059 scenes; existing 32-bit node channel"},
              "inference_receipts": {f"{run['method']}_s{run['seed']}": sha(Path(run["directory"]) / "run_receipt.json")
                                     for run in completed},
              "artifacts": {name: sha(OUTPUT / name) for name in artifacts}, "training": False,
              "model_parameters_changed": False}
    signed_write(OUTPUT / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "all", "summarize"], default="all", nargs="?")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    context = load_inputs()
    runs = select_runs()
    torch = configure_runtime()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    device = torch.device(args.device)
    if args.action == "smoke":
        results = [run_one(run, context, torch, device, smoke=True) for run in runs if run["seed"] == 11]
        print(json.dumps({"status": "SMOKE_PASS", "results": results}, indent=2))
        return
    completed = []
    if args.action == "summarize":
        for run in runs:
            method, seed = run["arm"], run["seed"]
            candidates = sorted((OUTPUT / "runs").glob(f"{method}_s{seed}_*"))
            matched = []
            for directory in candidates:
                receipt_path = directory / "run_receipt.json"
                if receipt_path.exists():
                    receipt = frozen.signed_read(receipt_path, "result_sha256")
                    if receipt["status"] == "COMPLETE":
                        matched.append(directory)
            if len(matched) != 1:
                raise ValueError(f"expected one completed run for {method}/{seed}, found {len(matched)}")
            completed.append({"method": method, "seed": seed,
                              "scope": "extended_0_32767" if method == "numeric" else "strict_0_999",
                              "directory": str(matched[0])})
    else:
        for run in runs:
            completed.append(run_one(run, context, torch, device))
    result = summarize(context, completed)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "result_sha256": json.loads((OUTPUT / "result.json").read_text())["result_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
