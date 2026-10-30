#!/usr/bin/env python3
"""Recompute classification and paired metrics from the released predictions."""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(rows, dataset):
    known = [r for r in rows if r["prediction"] != "UNKNOWN"]
    counts = collections.Counter((int(r["label"]), int(r["prediction"])) for r in known)
    tp, tn, fp, fn = [counts[p] for p in [(1, 1), (0, 0), (0, 1), (1, 0)]]
    div = lambda a, b: a / b if b else 0.0
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    out = dict(target=len(rows), known=len(known), unknown=len(rows)-len(known),
               positive=sum(int(r["label"]) for r in rows), tp=tp, tn=tn, fp=fp, fn=fn,
               Precision=div(tp, tp+fp), Recall=div(tp, tp+fn),
               Specificity=div(tn, tn+fp), F1=div(2*tp, 2*tp+fp+fn),
               MCC=(tp*tn-fp*fn)/math.sqrt(denom) if denom else "",
               coverage=div(len(known), len(rows)))
    if dataset in ("mit", "boundary"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r["pair_id"] if dataset == "boundary" else r["group"]].append(r)
        pairs = known_pairs = correct_pairs = full4 = known4 = correct4 = 0
        for group in groups.values():
            safe = [r for r in group if r["label"] == "0"]
            bad = [r for r in group if r["label"] == "1"]
            for a in safe:
                for b in bad:
                    pairs += 1
                    eligible = a["prediction"] != "UNKNOWN" and b["prediction"] != "UNKNOWN"
                    known_pairs += eligible
                    correct_pairs += eligible and a["prediction"] == "0" and b["prediction"] == "1"
            if dataset == "mit" and len(group) == 4:
                full4 += 1
                eligible = all(r["prediction"] != "UNKNOWN" for r in group)
                known4 += eligible
                correct4 += eligible and all(r["prediction"] == r["label"] for r in group)
        out.update(target_pairs=pairs, known_pairs=known_pairs, correct_pairs=correct_pairs,
                   pair_accuracy_known=div(correct_pairs, known_pairs), target_full4=full4,
                   known_full4=known4, correct_full4=correct4,
                   full4_accuracy_known=div(correct4, known4))
    return out


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def replay(out):
    names = json.loads((ROOT / "experiment_names.json").read_text())["prediction_method_names"]
    cohorts = collections.defaultdict(set)
    datasets = {}
    for r in read_csv(ROOT / "data/cohorts.csv"):
        cohorts[r["cohort"]].add(r["unit_id"])
        datasets[r["cohort"]] = r["dataset"]
    books = collections.defaultdict(dict)
    with gzip.open(ROOT / "data/predictions.csv.gz", "rt", newline="") as stream:
        for r in csv.DictReader(stream):
            key = (r["dataset"], r["method"], r["seed"])
            if r["unit_id"] in books[key]:
                raise ValueError("Duplicate prediction: " + str(key))
            books[key][r["unit_id"]] = r
    result = []
    for cohort, ids in sorted(cohorts.items()):
        dataset = datasets[cohort]
        for (ds, method, seed), rows in sorted(books.items()):
            if ds != dataset:
                continue
            if not ids.issubset(rows):
                raise ValueError(f"Missing registered IDs: {cohort}/{method}/{seed}")
            result.append(dict(cohort=cohort, dataset=dataset, method=method, paper_name=names.get(method, method), seed=seed,
                               **metrics([rows[u] for u in sorted(ids)], dataset)))
    lookup = {(r["cohort"],r["method"],r["seed"]):r for r in result}
    checks = 0
    for expected in read_csv(ROOT / "results/reference/requested_comparison_metrics_per_seed.csv"):
        key = (expected["cohort"],expected["method"],expected["seed"])
        if key not in lookup:
            continue  # References also contain additional sensitivity cohorts.
        actual = lookup[key]
        for metric in ["target", "known", "unknown", "tp", "tn", "fp", "fn", "F1", "Precision", "Recall", "Specificity", "pair_accuracy_known", "full4_accuracy_known"]:
            if expected.get(metric, "") == "" or metric not in actual:
                continue
            if not math.isclose(float(expected[metric]), float(actual[metric]), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Reference mismatch: {key} {metric}: {actual[metric]} != {expected[metric]}")
            checks += 1
    for expected in read_csv(ROOT / "results/reference/boundary_scores_metrics_corrected.csv"):
        if expected["cohort"] != "all448":
            continue
        method = {"Eva":"eva"}.get(expected["method"], expected["method"])
        seed = "deterministic" if expected["seed"] == "0" else expected["seed"]
        key = ("boundary448", method, seed)
        actual = lookup[key]
        for metric in ["target", "known", "unknown", "tp", "tn", "fp", "fn", "F1"]:
            if not math.isclose(float(expected[metric]),float(actual[metric]),rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError(f"Boundary reference mismatch: {key} {metric}")
            checks += 1
    internal_map = {"v70":"original_N", "v71":"residual_refiner", "v76":"no_refiner",
                    "v75":"fixed_encoder_no_dropout", "v52":"numeric_all_visible"}
    for expected in read_csv(ROOT / "results/reference/internal_metrics_by_seed.csv"):
        if expected["dataset"] != "overflowgen" or int(expected["cohort_target_n"]) != 11577:
            continue
        method = internal_map.get(expected["method"], expected["method"])
        key = ("source_registered11577",method,expected["seed"])
        if key not in lookup:
            continue
        actual = lookup[key]
        for metric in ["f1", "tp", "tn", "fp", "fn"]:
            if not math.isclose(float(expected[metric]),float(actual["F1" if metric == "f1" else metric]),rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError(f"Internal control reference mismatch: {key} {metric}")
            checks += 1
    grouped = collections.defaultdict(list)
    for r in result:
        grouped[r["cohort"],r["method"]].append(r)
    summary = []
    for (cohort, method), rr in sorted(grouped.items()):
        for metric in ["F1","Precision","Recall","Specificity","coverage","pair_accuracy_known","full4_accuracy_known"]:
            vals = [float(r[metric]) for r in rr if metric in r]
            if vals:
                summary.append(dict(cohort=cohort,method=method,paper_name=names.get(method, method),metric=metric,
                                    mean=statistics.mean(vals),sd=statistics.stdev(vals) if len(vals)>1 else "",n_seeds=len(vals)))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"metrics_per_seed.csv", result)
    write_csv(out/"summary.csv", summary)
    report = dict(status="PASS", reference_value_checks=checks, prediction_rows=sum(map(len,books.values())),
                  cohorts={k:len(v) for k,v in cohorts.items()},
                  scope="Saved-prediction replay; no training, model inference or analyzer execution.")
    (out/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/replay")
    args = parser.parse_args()
    replay(args.output.resolve())
