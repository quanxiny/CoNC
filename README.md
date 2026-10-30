# Preserving Numerical Structure in Learned Buffer-Overflow Vulnerability Detection

Source code, frozen inputs, checkpoints and reference predictions for the paper. **CoNC** (Coupled Numerical and Contextual detector; historically `Full`) is the numerical/context model with liquid graph refinement and partial freezing (`liquid_transfer_frozen`, v70). The v76 model is the **no-refiner ablation**. CoNC+Trace+Eva (H) is the separate decision policy combining CoNC, conditional address evidence and Eva alerts.

[中文说明](README_zh-CN.md) · [Core models and training](transformer_gru_v76_v2/README.md) · [External experiments and baselines](external_eval/README.md) · [Data](DATASETS.md) · [Licenses](NOTICE.md) · [Validation](VALIDATION.json)

## Start with the saved results

The source archive includes per-sample predictions and cohort membership. Python 3.11 is sufficient for these commands; no machine-learning packages or GPU are required.

```bash
python scripts/verify_bundle.py --part source --syntax
python scripts/replay_results.py --output outputs/replay
python scripts/check_paper_tables.py
```

The table check compares 76 published rounded mean F1 values in the main tables and strict-study supplement. Strict-study values are checked against saved summaries. The replay recomputes confusion counts, precision, recall, F1, specificity and paired correctness, checks them against the retained reference tables, and writes per-seed and mean/sample-SD CSV files. It covers the 11,577 source test slices, the 11,342-slice common subset, the registered 1,164 MIT scenes, its 796- and 1,059-scene cohorts, and 448 controlled boundary programs. Missing predictions remain `UNKNOWN`. Replaying saved predictions is distinct from retraining or rerunning inference.

## Install and load the data

Use Python 3.11 and a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
```

Extract the release attachments beside the source archive; every archive has the same top-level `overflow-reproducibility/` directory. The data attachment contains source tensors, original dataset archives, fixed splits, first-stage initialization checkpoints, MIT inputs and boundary inputs. The model attachment contains the retained CoNC, VDTransformer, DeepWukong and DeepDFA checkpoints. The optional hybrid attachment contains the nine source-only continuation checkpoints.

```bash
unzip overflow-reproducibility-data.zip
unzip overflow-reproducibility-weights.zip
# Optional: unzip overflow-reproducibility-hybrid-weights.zip
cd overflow-reproducibility
python scripts/verify_bundle.py --part data
python scripts/verify_bundle.py --part weights
python transformer_gru_v76_v2/scripts/smoke_core.py --with-assets
```

Run extraction from the parent directory, not from inside the repository. If using the already expanded delivery directory, the attachments are already present. Raw C sources and the original slice/graph files can be materialized with `python scripts/prepare_data.py --mode full`; this is unnecessary for cached tensor inference.

## Train and evaluate

```bash
# Small CPU inference check using a supplied CoNC checkpoint.
python transformer_gru_v76_v2/scripts/evaluate_core.py \
  --model conc --seed 11 --split test --device cpu --limit-rows 2 \
  --output-dir outputs/full_smoke

# Inspect the CoNC training recipe; add --execute to train.
python transformer_gru_v76_v2/scripts/train_full.py \
  --arm conc --seed 11 --output-dir outputs/full_s11
```

Use seeds 11, 29 and 47. The [core guide](transformer_gru_v76_v2/README.md) specifies source-stage training, CoNC training, refiner controls, representation controls, full-split inference and resume behavior. The [experiment guide](external_eval/README.md) specifies independent baselines, the strict mechanism study, frozen external inference, address evidence and H branch ablations. Optional graph baselines have additional dependencies; DeepDFA uses its separate requirements file and environment.

| Paper experiment | Implementation | Released evidence |
|---|---|---|
| RQ1: CoNC and independent baselines | Core `train_full.py`, `train_source.py`; `external_eval/baselines`; static-analyzer adapter | Source tensors, split IDs, native graph inputs, checkpoints, reference predictions |
| RQ2: internal controls | Core model variants; `external_eval/tifs/models.py` and `train.py` | Historical split, strict mechanism split, initialization states, per-seed metrics and numerical-state summaries |
| RQ3: frozen MIT transfer | `external_eval/portable_inputs.py` and external evaluator | Original MIT archive, 1,164 registered scenes, 1,578 slices for 1,059 processable scenes |
| RQ4: H and continuation controls | `external_eval/boundary_optimization/script` | Decision evidence, boundary programs/tensors, branch controls, optional continuation checkpoints |

## Layout

- `transformer_gru_v76_v2/`: shared model, input builder, training and inference source. The historical directory name preserves compatible paths.
- `external_eval/`: necessary external-input, baseline, mechanism and H implementations.
- `data/`: normalized predictions/cohorts plus the optional frozen input attachments.
- `results/reference/`: published result values and supporting numerical/paired summaries.
- `scripts/`: checksum verification, dataset extraction and saved-result replay.
- `manifests/`: file sizes and SHA-256 checksums for each component.

Large dataset/checkpoint attachments are excluded from ordinary Git history by `.gitignore`. Upload those attachments as GitHub Release assets or to a data repository. The source archive contains no checkpoint binaries, raw dataset archives or training outputs.

## Reproduction scope

This release preserves the historical family split, model selection and three seeds. The strict mechanism study uses a different component-disjoint split and must not replace the historical main-table denominator. MIT and the controlled boundary bank support retrospective diagnostics. CoNC's numerical states are learned representations; literal restoration does not imply exact learned arithmetic. H's supported address decisions do not certify unrestricted C safety.

CPU architecture checks, real-checkpoint inference samples, input-value parity, checksums and saved-result metrics were verified during packaging. CoNC GPU retraining and complete native-tool reruns were not repeated. The native static tools (Cppcheck 2.22.0, Clang 14 and Frama-C Eva 33.0) and Joern are external dependencies; their adapters and settings are supplied. Optional LineVul transfer uses upstream pretrained resources, which are identified but not redistributed. Historical orchestration receipts are not required by the portable entry points; some retained research-stage modules need the intermediate artifacts described in the experiment guide.

See `NOTICE.md` for the separate permissions and attribution of the original code and third-party data.

## Names used in the paper and in saved artifacts

The current paper calls the historical `Full` detector **CoNC**. The independently trained strict-study configuration also has the paper name CoNC, but uses the strict component split and distinct checkpoints; it must not be substituted for the historical family-split checkpoint. `conc` is accepted by the model and strict-study command-line entries; the previous `full` and `liquid_transfer_frozen` identifiers remain supported.

| Paper configuration | Historical identifier in checkpoints or prediction CSVs |
|---|---|
| CoNC | `original_N` in prediction records; `liquid_transfer_frozen` for historical training |
| H: original checkpoint | `original_H` |
| H: ordinary continuation | `ordinary_H`; training arm `ordinary_continue` |
| H: fallback-weighted continuation | `residual_H`; training arm `residual_weighted`, also accepted as `fallback_weighted` |
| H: class-matched continuation | `classmatch_H`; training arm `class_prior_matched` |
| Residual refiner | `residual_refiner`, a separate architecture control |
| No refiner (stage two) | `no_refiner` / `no_refiner_frozen` |
| Frozen-encoder dropout off | `fixed_encoder_no_dropout` |
| Stage-one checkpoint | `numeric_all_visible` / `source_backbone` |
| Lexical-context, numerical-structure, decimal-GRU joint controls | `semantic`, `numeric`, `legacy_joint` |

The fallback-weighted loss refers to neural fallback in H; it is not the residual-refiner architecture. [experiment_names.json](experiment_names.json) maps the paper's labels to all retained prediction and strict-study identifiers. Saved prediction and reference CSVs retain their original machine identifiers; the replay output adds `paper_name` alongside `method`.

Figure 6 and the accompanying disagreement counts can be rebuilt from the supplied compact, retrospective evidence:

```bash
python scripts/analyze_disagreements.py
python scripts/plot_case_studies.py
```

The plotting command needs `matplotlib` (install with `python -m pip install matplotlib`); it uses the installed serif font, so typography may differ from the manuscript. These cases were selected retrospectively and do not estimate failure prevalence. The 621 initialization-alarm and 497 no-final-state counts in the supplement come from archived native reports, not from the released predictions; `data/case_studies/disagreement_summary.json` records those saved counts without claiming to recompute them.
