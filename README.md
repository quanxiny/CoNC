# Preserving Numerical Structure in Learned Buffer-Overflow Vulnerability Detection

Reproducibility materials for the Coupled Numerical and Contextual detector (**CoNC**), its baselines, ablations and static-evidence policy. The implementation corresponds to the manuscript dated 14 November 2026.

CoNC combines a numerical/context backbone and graph refiner with an exact arithmetic channel and learned residual fusion. The channel evaluates supported COPY, ADD and SUB operations in dependency order on supplied expression graphs, with known operands and signed 32-bit results. Unknown inputs, unsupported operations and out-of-range results stay unknown. It does not establish complete C-language semantics.

[中文说明](README_zh-CN.md) · [Data and attribution](DATASETS.md) · [Licenses](NOTICE.md) · [Historical backbone guide](transformer_gru_v76_v2/README.md) · [Historical experimental code](external_eval/README.md)

## Reproduce saved results

Python 3.10 or later is sufficient; this command uses only the standard library:

```bash
python scripts/verify_bundle.py --part source --syntax
python scripts/current_experiment.py replay
```

The command recomputes F1 from individual saved predictions. It reports strict-test CoNC 97.39%, family common-test CoNC 97.06%, strict CoNC+Trace+Eva (H) 98.01%, and strict Trace+Eva+0 93.61%; the three-seed standard deviations are also printed. `data/current/` includes the current method, baseline and intervention predictions, plus Trace and Eva decisions. `results/current/` contains per-seed and comparison summaries. Saved-prediction replay is not retraining or inference.

## Install and run the model

Create a Python 3.11 environment and install `requirements-core.txt`. Download the release attachments to the parent of the source directory and extract them there:

```bash
unzip overflow-reproducibility-data.zip
unzip overflow-reproducibility-current-assets.zip
# Family-split backbone checkpoints and historical comparisons:
unzip overflow-reproducibility-weights.zip
cd overflow-reproducibility
python scripts/verify_bundle.py --part data
python scripts/verify_bundle.py --part current_assets
python scripts/current_experiment.py evaluate --seed 11 --arm exact_topo --device cpu
python scripts/current_experiment.py live --seed 11 --limit 32 --device cpu
```

`evaluate` runs the selected fusion branch on the frozen strict graph tensors, backbone logits and FP16 cached context for all 11,769 test examples, and checks predicted classes against the reference. Run seeds 11, 29, 47 and arms `exact_topo`, `literal_only`, `four_round` for the full matched comparison. `live` loads the strict stage-two backbone and fusion branch, computes context directly from the encoded dataset, and checks a sample of complete model predictions. Set `--limit 11769` for the full strict test. The current-assets archive contains strict split inputs, cached contexts, strict backbones, strict fusion and matched-control heads, family fusion heads, and strict DeepWukong checkpoints.

To retrain one fusion branch from the supplied frozen strict backbone context:

```bash
python scripts/train_current_fusion.py --seed 11 --arm exact_topo --device cuda:0 --output-dir outputs/strict_exact_s11
```

The training recipe uses the strict TRAIN and DEV tensors, DEV cross-entropy selection including epoch zero, AdamW, and the fixed residual penalty. GPU retraining may exhibit platform-dependent floating-point differences. Backbones were trained separately; the code does not claim end-to-end retraining from cached context. The retained research-stage campaign code records the original split, baseline and native-tool protocols and may require intermediate artifacts beyond these compact attachments. See the historical guides for backbone training and baseline dependencies; those guides describe the earlier backbone-only CoNC and older experimental tables.

## Experiments and naming

| Manuscript name | Released identifier | Scope |
|---|---|---|
| CoNC | `exact_topo` | Frozen numerical/context backbone with exact channel and residual fusion |
| CoNC backbone | `original` | Backbone without the new fusion branch |
| Literal-only fusion | `literal_only` | Matched branch with literal states only |
| Four-round fusion | `four_round` | Matched finite-round propagation control |
| H: CoNC+Trace+Eva | `H` | Trace decision, then Eva positive alert, then CoNC fallback |
| Static evidence control | `Trace+Eva+0` | Same Trace/Eva evidence with zero fallback |

The strict component split has 94,157 TRAIN, 11,770 DEV and 11,769 TEST slices. The historical family split has 94,179 TRAIN, 11,940 DEV and 11,577 TEST slices; the family common-test result uses 11,342 slices. Strict and family checkpoints are different and must not be interchanged. Both are retrospective studies on an already-used corpus; component isolation does not establish semantic novelty. Strict TEST lacks several library-call forms. The 7,178/7,178 exact-channel result refers only to reference-computable COPY/ADD/SUB probe nodes, not all graph nodes or programs.

`results/current/strict/hybrid_reanalysis.json` is a subsequent post-hoc analysis; it is separate from the manuscript's main table. See [VALIDATION.json](VALIDATION.json) for the scope of the current checks.

Original datasets, third-party implementations and licenses remain subject to their respective terms; see [DATASETS.md](DATASETS.md) and [NOTICE.md](NOTICE.md).
