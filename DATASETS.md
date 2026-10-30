# Data and checkpoints

Extract the data and optional checkpoint attachments into the repository root. Their directory layouts match the source package. All dataset paths below are relative to that root. The data attachment supplies the frozen inputs used by the experiments; training on those inputs does not require rerunning a parser.

```bash
python scripts/prepare_data.py --check
```

This verifies the three original dataset archives, the selected TRAIN/DEV tensor checksums, and the portable MIT and controlled-boundary input manifests. The release inventories record each supplied file's SHA256. When metadata was made portable, its original SHA256 is recorded separately from the checksum of the released file. Numerical inputs and learned checkpoint parameters are unchanged. In the H continuation checkpoints, only the stored DEV-prediction path is made relative; both original and release checksums are recorded.

## OverflowGen source experiments

The original archive is `datasets/overflowgen_24455320/data.zip` (903,946,665 bytes; SHA256 `70d4785d8c3937055275482e97527a095b0eb28fb1b9deb9ec0b829d582d54fe`). It is attributed to Yang Cao, Yunwei Dong and Jiajie Peng under CC BY 4.0. See the supplied `datasets/overflowgen_24455320/NOTICE.md`, its source manifest, the [original dataset item](https://figshare.com/s/fbaf47e3ac2a9581dbd7), and [associated paper](https://doi.org/10.1002/cpe.8292).

| Asset | Location and purpose |
|---|---|
| Complete source cache | `transformer_gru_v76_v2/external/overflowgen/joint_graph_study_v1/`: 20 NumPy arrays for 117,696 slices, with row identities and a tensor schema. |
| Historical family assignment | `transformer_gru_v76_v2/external/overflowgen/protocol_family_v1/`: 94,179 TRAIN, 11,940 DEV and 11,577 TEST slices. This is the historical coarse-family allocation used by CoNC. |
| Row metadata | `transformer_gru_v76_v2/assets/comparisons/overflowgen/overflow_interfaces_v51/output/row_metadata.jsonl.gz`: stable IDs, source cases, families, labels and slice hashes. |
| Selected training inputs | `transformer_gru_v76_v2/src/research_project/ubuntu_local_research_store/hybrid_numeric_semantic_v59/baseline_dev60/`: 18 arrays covering the 106,119 TRAIN/DEV rows. `row.npy` maps to the complete source cache; `split.npy` uses 0 for TRAIN and 1 for DEV. `MANIFEST60.json` contains the checksums of exactly these 18 arrays. |
| Vocabulary and numerical resources | `transformer_gru_v76_v2/external/overflowgen/safe_numeric_artifacts/`: `word_index.json`, `word_vectors.npy` and `dice_Q.npy`. |

To obtain the author slice JSON files from the supplied archive:

```bash
python scripts/prepare_data.py --mode slices
```

To also unpack the original C programs and graph tables needed for frontend reconstruction and static-analysis reruns:

```bash
python scripts/prepare_data.py --mode full
```

The script creates `datasets/overflowgen_24455320/data/` and copies the author JSON inputs to `transformer_gru_v76_v2/external/overflowgen/author_data/data/overflow_gen/`. Existing identical files are verified and reused. Different existing files cause an error. Full extraction also creates the archive location expected by the cache-rebuild code under `transformer_gru_v76_v2/external/overflowgen/archives/`, using a local hard link where supported and an ordinary copy otherwise. No workspace-specific symbolic links are created. The full extraction needs substantially more disk space than the compressed archive.

## MIT external evaluation

`data/mit/` contains `scenes.jsonl`, `slices.jsonl`, `tensors.npz`, `manifest.json`, and sanitized source programs in `sources/`. All 1,164 target scenes remain in the scene manifest, including 105 scenes without complete neural inputs. Unknown scenes are not labeled safe.

- `strict_eligible` identifies the original 796 complete scenes: 531 positive and 265 negative.
- `extended_eligible` identifies the 1,059 complete scenes: 794 positive and 265 negative.
- The cache contains 1,578 slices: 1,186 original-range slices and 392 recovered slices. Each slice references a row of `tensors.npz` and its owning scene.
- `tensor_cache` and `source_path` are relative to the repository root. The 15 tensor fields retain their original dtype, shape, values and padding. Every exported row was compared with the recorded original cache, and the consolidated archive was reloaded and checked.

The original archive, `datasets/2014-06-08-a-taxonomy-of-buffer-overflows.zip`, and the official replacement of `basic-00180-ok`, `external_eval/vendor/mit/2073-v2.0.0.zip`, are supplied unchanged. The replacement is documented by the [NIST SARD case record](https://samate.nist.gov/SARD/test-cases/2073/versions/2.0.0).

The source files carry the 2004 M.I.T. permission notice allowing use, copying, modification and distribution with the notice retained. The full original notice is in `licenses/MIT-Taxonomy.txt` and remains embedded in the original archives. This dataset's name does not imply that its license is identical to the repository's standard MIT license.

## Controlled pairs and matched mechanism controls

`data/boundary/` supplies the 448 controlled programs used in the H comparison: 224 safe/overflow pairs in eight construction groups. It has the same scene/slice/tensor layout as MIT, with `pair_id`, `group`, `capacity` and `value` fields, and contains 896 required slices. This is the retrospective controlled bank reported in the paper, including its recorded development contacts.

`data/mechanism/source_manifest.csv` contains the separate component-based split for the matched mechanism study. Its 117,696 rows retain source IDs, labels, split membership, grouping and input hashes. Do not substitute the historical family split for this manifest.

`data/hybrid/` contains the fixed source-only continuation material: 94,179 TRAIN and 11,940 DEV gate records, three exact 1,280-by-32 row schedules, and the optimization hyperparameters. `D_trace`, `Eva` and `R` describe the address decision, static alert and neural-fallback selection. No MIT or controlled-boundary examples are used by these continuation schedules.

The controlled program generators, split definitions and gate-processing code are project contributions under the repository MIT license. OverflowGen-derived inputs retain the upstream CC BY 4.0 attribution.

## Independent baseline inputs

`data/baselines/deepwukong/` provides `source.jsonl.gz` and `mit.jsonl.gz`, containing the prepared graph/sequence inputs with original tokens, edges, IDs, labels and split membership. Only machine-specific provenance paths were changed.

`data/baselines/deepdfa/` provides `source.pt`, `mit.pt`, `vocabulary.json` and `coverage.jsonl.gz`. The packed graph tensors and vocabulary retain their original bytes; coverage records are restricted to the supplied source and MIT populations. Each baseline directory has a compact `manifest.json` with checksums. Upstream source-code licenses are retained separately in the source package.

FormAI, Juliet and PrimeVul data are not included in this main-experiment data attachment. Supplementary FormAI-v2 material originates from the [FormAI dataset repository](https://github.com/FormAI-Dataset/FormAI-dataset); the source path is recorded in the original experiment's dataset configuration. It is not required for the source, MIT or controlled-boundary commands in this release.

## Initialization and optional trained checkpoints

The data attachment includes `transformer_gru_v76_v2/assets/initialization/v52_all_visible_s{11,29,47}.pt`, the three source-DEV reference records under `assets/reference_metrics/`, and the three recorded initialization RNG states under `src/research_project/ubuntu_local_research_store/no_refiner_frozen_v76/initialization/`. These are necessary to reproduce the supplied two-stage training recipe.

The optional weights attachment includes the selected CoNC, VDTransformer, DeepWukong and DeepDFA checkpoints for seeds 11, 29 and 47, plus their required configuration and embedding data. CoNC is the `liquid_transfer_frozen` architecture under `assets/comparisons/overflowgen/adaptive_graph_liquid_v70/runs/`; the `no_refiner_frozen_v76` name identifies a separate ablation.

The separate hybrid-weights attachment contains `weights/hybrid/{ordinary_continue,residual_weighted,class_prior_matched}_s{11,29,47}.pt` and a portable manifest. Each is bound to its historical CoNC checkpoint and the recorded 1,280-update source-only continuation. Fixed-checkpoint H uses the historical CoNC weights.

All supplied checkpoint files have SHA256 sidecars. Learned project parameters and implementation code are distributed under the project MIT license; the training-data attribution remains CC BY 4.0. No external pretrained CodeBERT weights, package environments or parser binaries are supplied.
