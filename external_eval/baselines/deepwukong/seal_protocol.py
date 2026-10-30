#!/usr/bin/env python3
"""Seal source-only recipe after data/model acceptance, before eligible fitting."""
from datetime import datetime, timezone
import inspect
import adapter as a


def main():
    path = a.OUTPUT / "protocol.json"
    if path.exists():
        raise ValueError("protocol already sealed; do not change a running experiment")
    prepared = a.frozen.signed_read(a.OUTPUT / "data/preparation.json", "result_sha256")
    accepted = a.frozen.signed_read(a.OUTPUT / "acceptance/receipt.json", "result_sha256")
    if prepared["status"] != "COMPLETE" or prepared["scope"] != "FROZEN_ALL_COHORTS":
        raise ValueError("full data acceptance required")
    for name, expected in prepared["artifacts"].items():
        if a.sha(a.OUTPUT / "data" / name) != expected:
            raise ValueError("prepared data changed before sealing")
    regression = a.frozen.signed_read(a.OUTPUT / "data/native_edge_regression.json", "result_sha256")
    from collections import Counter
    fixture_counts = Counter(r["dataset"] for r in regression["fixtures"])
    if any(fixture_counts[name] < count for name, count in {"overflowgen": 128, "mit": 16, "formai": 16}.items()):
        raise ValueError("native edge fixtures required for every cohort")
    if accepted["adapter_sha256"] != a.sha(a.__file__) or accepted["runner_sha256"] != a.sha(a.BASE / "train_evaluate.py"):
        raise ValueError("implementation changed since model acceptance")
    original, _ = a.original_functions()
    example = ["int capacity = 8;\n", "if (count < 4096 && count <= capacity) {\n",
               "  strcpy(buffer, \"abcdef\");\n", "}\n"]
    tokenized = [original["tokenize_code_line"](s, False) for s in original["clean_gadget"](example)]
    flattened = sum(tokenized, [])
    assert all(t in flattened for t in ["8", "4096", "if", "<", "<=", "&&", '""'])
    assert "abcdef" not in flattened
    a.signed(a.OUTPUT / "symbolizer_regression.json", {"status": "PASS", "input": example,
             "official_tokens": tokenized, "integer_tokens_and_guard_syntax_retained": True,
             "string_content_removed_by_official_clean_gadget": True,
             "learned_vocabulary_can_map_unseen_literals_to_UNK": True,
             "statement_token_limit": 16, "no_handcrafted_numeric_features": True})
    from pytorch_lightning.callbacks import EarlyStopping
    min_delta = inspect.signature(EarlyStopping).parameters["min_delta"].default
    if min_delta != 0.0:
        raise ValueError("native early-stopping default differs from accepted loop")
    import torch, torch_geometric, gensim, pytorch_lightning
    a.signed(path, {
        "status": "SEALED_BEFORE_ELIGIBLE_FITTING", "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "display_name": "DeepWukong-GCN (matched-slice adaptation)",
        "method_id": "deepwukong_matched_slice", "upstream": a.verify_vendor(),
        "license": "MIT", "model_class": "src.models.vd.DeepWuKong",
        "native_pipeline_claim": False,
        "adaptation": {
            "unit": "existing benchmark slice; external score is maximum slice risk per scene",
            "slicing": "fixed backward benchmark slices, not upstream backward-plus-forward XFG enumeration",
            "labels": "fixed query/scene benchmark labels, not native any-vulnerable-line XFG labeling",
            "graph": "original Joern CONTROLS/REACHES line projection induced on retained slice lines",
            "node_order": "existing ordered source-line list",
            "edge_types": "native GCN uses directed adjacency without separate relation channels",
            "empty_edges": "normalize upstream empty edge tensor rank to [2,0], no edges added",
            "sanitization": "source comments and label-bearing identifiers neutralized before symbolization; external canonical sanitized source reused unchanged",
            "model": "unmodified pinned official model, statement BiLSTM and GCN/TopK/attention/classifier bodies",
            "numeric_candidate_graph": False,
            "training_loop": "manual equivalent CE/Adam/LambdaLR/clip/DEV loop for modern execution and immutable receipts",
            "word2vec_reproducibility": "stable TRAIN row order, one worker and explicit seed/hash seed replace upstream unordered multicore defaults"
        },
        "architecture": {"trainable_embedding_dim": 256, "statement_token_limit": 16,
            "statement_encoder": "one-layer bidirectional LSTM hidden256, sum direction states, dropout0.5",
            "graph_encoder": "3 GCNConv256/TopKPooling0.8 stages; GlobalAttention per stage, sum readouts",
            "classifier": "256->512->512->2; ReLU/dropout0.5"},
        "recipe": {"seeds": [11, 29, 47], "source_train": 94179, "source_dev": 11940, "source_test": 11577,
            "batch_size": 64, "precision": "CPU FP32", "threads_per_seed": 4,
            "optimizer": "native Adam", "learning_rate": 0.002, "weight_decay": 0,
            "lr_multiplier_each_epoch": 0.95, "gradient_clip_L2": 5, "max_epochs": 50,
            "patience": 10, "early_stopping_min_delta": min_delta,
            "selection": "minimum source DEV mean-of-batch cross-entropy after completed epoch; sample-mean CE also recorded",
            "checkpoint_improvement_min_delta": 0, "eligible_epoch0": False,
            "threshold": 0.5, "TEST_or_external_selection": False,
            "word2vec": {"training_split": "family TRAIN94179 only", "vector_size": 256,
                "sg": 1, "min_count": 3, "max_vocab_size": 190000, "workers": 1,
                "window": 5, "negative": 5, "epochs": 5, "per_model_seed": True, "PYTHONHASHSEED": 0}},
        "external": {"mit": {"scenes": 1059, "slices": 1578, "common_scenes": 796},
                     "formai": {"scenes": 167, "slices": 2709, "common_scenes": 161},
                     "aggregation": "maximum slice risk; common cohorts filtered from same predictions"},
        "data_receipt_sha256": a.sha(a.OUTPUT / "data/preparation.json"),
        "acceptance_sha256": a.sha(a.OUTPUT / "acceptance/receipt.json"),
        "symbolizer_regression_sha256": a.sha(a.OUTPUT / "symbolizer_regression.json"),
        "code_sha256": {n: a.sha(a.BASE / n) for n in ["adapter.py", "prepare.py", "train_evaluate.py", "run_three_seeds.py", "seal_protocol.py"]},
        "runtime": {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                    "gensim": gensim.__version__, "lightning": pytorch_lightning.__version__},
        "truncation_and_input_counts": prepared["counts"],
        "pilot": {"fixtures": 128, "batch64_median_s": accepted["median_step_s"],
                  "small_slice_cost_estimate_only": True}
    })
    print(str(path), flush=True)


if __name__ == "__main__":
    main()
