# Source model and internal controls

The paper's **CoNC** model (historical ID `Full`) is `AdaptiveGraph70('liquid_transfer_frozen')`:
AllVisible numerical/context backbone followed by liquid graph refinement, with
partial parameter freezing during the second training stage. `NoRefiner76` is
the refiner-removal control. The directory name is retained for compatibility
with the data and evaluation paths.

The model equations and checkpoint parameter names are preserved. Only required
source modules are included. Historical campaign schedulers and development
reports are not needed by the release entry points.

Run commands from the repository root, after installing `requirements-core.txt`
and extracting the data and initialization/checkpoint attachments into that root.

## Check the installation

```bash
python transformer_gru_v76_v2/scripts/smoke_core.py
python transformer_gru_v76_v2/scripts/smoke_core.py --with-assets
python transformer_gru_v76_v2/scripts/train_full.py --help
python transformer_gru_v76_v2/scripts/train_full.py \
  --arm conc --seed 11 --output-dir outputs/full_s11
```

The last command prints the training configuration and exits. Training requires
`--execute`. The architecture check uses constructed inputs and performs zero
optimizer updates; `--with-assets` additionally checks the supplied checksums,
TRAIN/DEV tensor allocation and VDTransformer forward pass.

## Evaluate the retained models

```bash
python transformer_gru_v76_v2/scripts/evaluate_core.py \
  --model conc --seed 11 --split test --device cpu --limit-rows 2 \
  --output-dir outputs/full_s11_smoke
python transformer_gru_v76_v2/scripts/evaluate_core.py \
  --model conc --seed 11 --split test --device cuda:0 \
  --output-dir outputs/full_s11_test
python transformer_gru_v76_v2/scripts/evaluate_core.py \
  --model vd --seed 11 --split test --device cuda:0 \
  --output-dir outputs/vd_s11_test
```

Repeat for seeds 29 and 47. `--limit-rows` is explicitly partial inference;
omitting it evaluates the complete named split. The TEST set contains 11,577
slices. It differs from the 11,342-slice intersection with static analyzers in
the main comparison table. The evaluator writes row IDs, labels, scores,
predictions, metrics and checkpoint checksums. Use the external-evaluation
scripts for the aligned table cohorts, graph baselines, MIT and hybrid policy.

`--checkpoint PATH` evaluates a new checkpoint with its `.pt.sha256` sidecar.
Additional `--model` choices expose `source_backbone`, `no_refiner`,
`residual_refiner`, `lexical_context`, `numerical_structure`,
`joint_representation` and `fixed_encoder_no_dropout`. The internal semantic
and numeric models retain their historical widths, 512 and 576 respectively.

## Frozen external inference

```bash
python external_eval/evaluate_portable.py --dataset mit --model conc --seed 11 \
  --output-dir outputs/mit_full_smoke
python external_eval/evaluate_portable.py --dataset mit --model vd --seed 11 \
  --output-dir outputs/mit_vd_smoke
python external_eval/evaluate_portable.py --dataset boundary --model conc --seed 11 \
  --output-dir outputs/boundary_full_smoke
```

These commands default to two registered scenes on CPU. Add `--all-scenes` for
complete inference, optionally with `--device cuda:0`. CoNC uses the unchanged
0–32,767 literal adapter; VDTransformer retains its 0–999 domain and the 796-scene
MIT eligibility set. All registered scenes remain in a complete output, including
unsupported scenes marked `UNKNOWN`. Scene decisions use the maximum FP32
class-1 probability across the full required slice inventory; equal probabilities
keep the first slice in the archived order. The output includes slice logits,
scene scores, unknown reasons and metrics over known scenes.

## Train CoNC and refiner controls

```bash
python transformer_gru_v76_v2/scripts/train_full.py \
  --arm conc --seed 11 --output-dir outputs/full_s11 --execute
python transformer_gru_v76_v2/scripts/train_full.py \
  --arm residual_transfer_frozen --seed 11 --output-dir outputs/residual_s11 --execute
python transformer_gru_v76_v2/scripts/train_v76.py \
  --arm no_refiner_frozen --seed 11 --output-dir outputs/no_refiner_s11 --execute
```

The CoNC entry preserves the original v70 worker's forward path, random
initialization stream, transfer mapping, partial freeze, AdamW optimizer,
learning-rate schedule, sample order, epoch-zero candidate and source DEV
selection. The residual control uses the original v71 class. The no-refiner
entry retains the original v76 post-initialization RNG states and resume
implementation. Source labels are never added to the model inputs.

Training uses FP32, batch size 64, up to 100 epochs, patience 10, minimum DEV-loss
improvement 0.0001, learning rate 0.0001, weight decay 0.01, norm-5 gradient
clipping and epoch factor 0.95. The original run budget is six hours per second
stage, plus a separate final inference allowance. Use a CUDA device with about
32 GiB memory, at least 16 GiB host RAM and 8 GiB free output storage per run.
The CPU option belongs to inference and smoke checks. Training uses `cuda:0`;
set `CUDA_VISIBLE_DEVICES` to select a physical device. Output directories must
be new. `--resume` uses the same run directory and exact saved configuration.

Relative package paths, checksummed resources and independent output directories
replace the original machine paths and campaign-completion gates. These entry
changes do not constitute a new reproduction of the published numerical scores.
The release was checked on CPU; full GPU retraining has not been repeated during
packaging.

## Source stage and internal representation controls

The retained initialization checkpoints allow the preceding commands to start
at stage two. The source stage can also be trained from scratch:

```bash
python transformer_gru_v76_v2/scripts/train_source.py \
  --arm numeric_all_visible --seed 11 --output-dir outputs/source_s11 --execute
python transformer_gru_v76_v2/scripts/train_source.py \
  --arm vd --seed 11 --output-dir outputs/vd_s11 --execute
python transformer_gru_v76_v2/scripts/train_source.py \
  --arm semantic --seed 11 --output-dir outputs/semantic_s11 --execute
python transformer_gru_v76_v2/scripts/train_source.py \
  --arm numeric --seed 11 --output-dir outputs/numeric_s11 --execute
python transformer_gru_v76_v2/scripts/train_source.py \
  --arm legacy_joint --seed 11 --output-dir outputs/joint_s11 --execute
```

This entry keeps the source worker's convergence and optimizer rules and uses
the same selected rows in original row order. VDTransformer uses Adam with
learning rate 0.00002, zero weight decay and zero minimum improvement; its
family-split lexical embedding is randomly initialized. Other source arms use
the recipe above. The default source-stage budget is four hours (`--hours`).
Model modules, training functions and checkpoints retain their original schemas.

A completed `numeric_all_visible` run writes `transfer_sources.json` beside its
checkpoint. To initialize CoNC from that new source run, set
`OVERFLOW_TRANSFER_SOURCES` to this file when invoking `train_full.py`. Its
paths are resolved relative to the source configuration. For example:

```bash
OVERFLOW_TRANSFER_SOURCES=outputs/source_s11/runs/family_numeric_all_visible_s11_lr0.0001/transfer_sources.json \
python transformer_gru_v76_v2/scripts/train_full.py \
  --arm conc --seed 11 --output-dir outputs/new_full_s11 --execute
```

`train_comparison.py` additionally runs existing architectures with the shared
v76 training recipe. These are explicitly matched-recipe reruns; the primary
CoNC and native VDTransformer recipes are the entries documented above.

## Data contract

`configs/MIGRATION_CONTRACT_v2.json` identifies the selected TRAIN/DEV arrays,
initialization checkpoints, DEV reference metrics and initialization RNG files.
`configs/source_data.json` identifies the complete source tensors used for
TRAIN/DEV/TEST inference. Checksums and the archived split prevent accidental
changes in input identity. The input builder remains a finite token frontend;
computed numerical states are learned features, not verified arithmetic results.

The historical split and retained three-seed checkpoints support the reported
retrospective comparisons. Running this package does not convert them into an
independent test or establish effectiveness beyond the evaluated cohorts.
