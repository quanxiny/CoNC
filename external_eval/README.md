# Experimental code

Run commands from the repository root after unpacking the data and, for retained-checkpoint inference, the weights bundle. The main source experiment uses the fixed family split (94,179 TRAIN / 11,940 DEV / 11,577 TEST). The supplementary mechanism experiment uses the retrospective strict component split (94,157 / 11,770 / 11,769). These protocols must not be combined into one training comparison. Source DEV selects checkpoints; target labels do not select models or thresholds.

## CoNC and VDTransformer inference

Install `requirements-core.txt`. The portable entry point reads the released scene and tensor manifests and verifies their checksums:

```bash
python external_eval/evaluate_portable.py --dataset mit --model conc --seed 11 --output-dir outputs/mit_full
python external_eval/evaluate_portable.py --dataset mit --model vd --seed 11 --output-dir outputs/mit_vd
python external_eval/evaluate_portable.py --dataset boundary --model conc --seed 11 --output-dir outputs/boundary_full
```

The default processes two scenes on CPU. Add `--all-scenes` for complete inference. Use a fresh output directory for each model, seed and scope. MIT has 1,164 registered scenes, 796 strict scenes and 1,059 extended processable scenes; the boundary bank has 448 scenes and 224 pairs. CoNC extends its deterministic numeric encoding to 32,767; VD retains its native 0–999 domain and the strict MIT cohort. Scene risk is the maximum over required slices. A scene with any unsupported required slice remains `UNKNOWN`.

For source TEST inference, use `transformer_gru_v76_v2/scripts/evaluate_core.py`; its README documents the historical CoNC, independent VDTransformer and internal variants. `portable_inputs.py` also accepts `OVERFLOW_DATA_DIR` pointing to an extracted `data/` directory.

## Eight mechanism variants

The two-stage implementation and optimizer are in `tifs/models.py` and `tifs/train.py`. Variants are `conc` (historical ID `full`), `local_q`, `no_h_to_q`, `no_dynamic_q_to_h`, `symmetric_q`, `no_clamp`, `mlp_q`, and `no_both_dynamic`. Seeds are 11, 29 and 47.

```bash
python -m external_eval.tifs.validate --out outputs/mechanism_smoke --device cpu
python -m external_eval.tifs.run prepare
python -m external_eval.tifs.run train --variant conc --seed 11 --stage 1 --device cuda:0
python -m external_eval.tifs.run train --variant conc --seed 11 --stage 2 --device cuda:0
python -m external_eval.tifs.run score --variant conc --seed 11 --dataset source --device cuda:0
python -m external_eval.tifs.run score --variant conc --seed 11 --dataset mit --device cuda:0
```

Repeat the train and score commands for each variant and seed. Pass `--resume` to resume a training stage. Outputs default to `external_eval/tifs_runs/mechanism_release`; `--out` must remain under `external_eval/tifs_runs`. Preparation verifies the strict split and freezes code, input and split checksums. VD uses only stage 1 (`--variant vd`). The separately named `diagnostics.py` generates the supplementary controlled diagnostic design; it is distinct from the released 448-scene final boundary bank.

The CPU validation covers forward/backward computation for all variants, exact CoNC parity with the original implementation, stage-transition parity, dynamic-path removal, numeric invariants, and exact optimizer/cursor/RNG resume. It passed in the packaging environment. CoNC training was not repeated during packaging.

## Independent source-trained baselines

The released `data/baselines` inputs retain the original graph edges, token sequences, labels and train/dev/test membership. DeepDFA vocabulary and DeepWukong embeddings are fitted using source TRAIN only. Three independent seed runs use 11, 29 and 47. Do not install both graph stacks into the core environment.

### DeepDFA

Use a separate Python 3.11 environment and `requirements-deepdfa.txt`. The verified stack uses PyTorch 2.10.0+cu128 and DGL 1.1.2; DGL 2's GraphBolt binaries are not interchangeable with this stack. The documented experiment runs on CPU with eight threads and flush-to-zero enabled, as specified in `paper_baselines/deepdfa/cpu_runtime.json`.

```bash
python external_eval/baselines/deepdfa/training.py --help
python external_eval/baselines/deepdfa/training.py smoke
python external_eval/baselines/deepdfa/training.py train --seed 11 --out outputs/deepdfa
python external_eval/baselines/deepdfa/training.py evaluate --seed 11 --out outputs/deepdfa
```

The eight-TRAIN-graph forward/backward smoke passed with the unchanged official GGNN (375,938 parameters). The prepared inputs support the main family-split experiment. Re-fitting DeepDFA for the supplementary strict split requires rerunning the raw abstract-dataflow feature extraction before `tifs/deepdfa.py pack`; the large Joern extraction cache is not included. Reassigning already encoded main-experiment features to the strict split would not refit the vocabulary and is not an equivalent reproduction.

### DeepWukong

Use a separate Python 3.10 environment and `requirements-deepwukong.txt`. The exact upstream model and its license are under `baselines/deepwukong/vendor`. Set `PYTHONHASHSEED=0` before starting Python:

```bash
PYTHONHASHSEED=0 python external_eval/baselines/deepwukong/train_evaluate.py --help
PYTHONHASHSEED=0 python external_eval/baselines/deepwukong/train_evaluate.py accept --seed 11 --threads 4
PYTHONHASHSEED=0 python external_eval/baselines/deepwukong/train_evaluate.py train --seed 11 --threads 4
```

Acceptance passed on 128 TRAIN fixtures: token tensors, edges and model logits matched the upstream implementation exactly. Eight real training batches completed; acceptance saves no classifier weights. Formal runs save artifacts under `external_eval/paper_baselines/deepwukong`. The dependency pins describe the environment used for this verification; package installation from the public indexes was not repeated during packaging.

## Hybrid decisions and continuation

The final policy is: use a known address-rule result; otherwise use an Eva-positive result; otherwise use the neural result. Eva-negative or unknown results do not certify safety. All five policy variants share the same neural predictions and static results.

```bash
python -m external_eval.hybrid --self-test
python -m external_eval.hybrid --input data/predictions.csv.gz --method conc --output outputs/hybrid.csv
python -m external_eval.continuation --prepare-only
python -m external_eval.continuation --smoke --arm fallback_weighted --out outputs/continuation_smoke
python -m external_eval.continuation --arm ordinary_continue --seed 11 --device cuda:0
python -m external_eval.continuation --arm fallback_weighted --seed 11 --device cuda:0
python -m external_eval.continuation --arm class_prior_matched --seed 11 --device cuda:0
```

The 27 possible three-state decision combinations passed the policy self-test. The continuation smoke also passed a nonzero-loss forward/backward check on two TRAIN examples selected from 128 screened TRAIN rows; it saved no model weights. Continuation uses the released source gates and fixed 1,280-step schedules. Each seed starts from its historical CoNC checkpoint; ordinary continuation, fallback-weighted continuation and class-matched continuation use the same schedule. Class constants and residual normalization are derived from the complete source TRAIN partition. Preparation checks gate identity, consumed-input hashes, split membership and source array checksums. No target adaptation is performed.

## LineVul and native tools

LineVul is the official Big-Vul-trained checkpoint evaluated as a separate cross-domain comparator. It is not an equal-source-training baseline. Official model code and license are included under `transformer_gru_v76_v2/third_party/LineVul`. Tokenizer/config files are included; the approximately 997 MB task checkpoint must be obtained from the upstream [LineVul repository](https://github.com/awsm-research/LineVul) using the public download ID and exact SHA-256 in `baselines/linevul/OFFICIAL_CHECKPOINT_MANIFEST.json`. Place it at `external_eval/baselines/linevul/official_12heads_linevul_model.bin`, then install the optional dependencies and run:

```bash
python external_eval/baselines/linevul/frozen_transfer.py --help
python external_eval/baselines/linevul/frozen_transfer.py accept
python external_eval/baselines/linevul/frozen_transfer.py infer
```

Checkpoint loading checks the size, SHA-256 and exact parameter keys. LineVul inference was not rerun during packaging because the checkpoint is intentionally not included in the source bundle.

Raw frontend reconstruction and native static-analysis runs require the corresponding official source archives and external tools. `vendor/source_assets.json` records pinned download locations and hashes. Cppcheck 2.22, Clang 14 and Frama-C 33 can be supplied using `CPPCHECK_BIN`, `CLANG_BIN` and `FRAMAC_BIN` or installed on `PATH`. Their selectors and timeout/unknown handling are retained in `static_baselines.py` and `boundary_optimization/script`. Joern/Java binaries and dependency installations are not bundled. The prepared tensors and baseline graphs permit inference/training without rerunning extraction.

The supplementary FormAI dataset is not redistributed in this release because its local redistribution license was not established. Its saved supplementary summary tables are retained in results/reference; FormAI per-sample replay, new feature extraction and inference require obtaining the original dataset and its corresponding predictions separately. Optional native-tool and FormAI execution paths were not rerun during packaging. Historical run directories, task-control records and environment-specific receipts are not prerequisites for the portable commands above.
