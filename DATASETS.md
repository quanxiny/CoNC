# Data and checkpoints

The current experiments use OverflowGen. Yang Cao, Yunwei Dong and Jiajie Peng distribute the [original data and code](https://figshare.com/s/fbaf47e3ac2a9581dbd7) under CC BY 4.0; see their [paper](https://doi.org/10.1002/cpe.8292) and the notices in the data attachment. The supplied original `datasets/overflowgen_24455320/data.zip` has SHA-256 `70d4785d8c3937055275482e97527a095b0eb28fb1b9deb9ec0b829d582d54fe`. This checksum identifies the supplied bytes.

Extract `overflow-reproducibility-data.zip` and `overflow-reproducibility-current-assets.zip` into the parent of the source directory. Then verify the inventory:

```bash
python scripts/verify_bundle.py --part data
python scripts/verify_bundle.py --part current_assets
```

The data attachment provides the original OverflowGen archive, encoded graph cache, row metadata, historical family assignment, strict component manifest, prepared baseline inputs and backbone initialization files. The `overflow-reproducibility-strict-baselines.zip` attachment contains the strict-split VDTransformer, DeepDFA and backbone-intervention checkpoints, with original and released checksums. The current-assets attachment provides strict split graph tensors and cached backbone contexts for TRAIN, DEV and TEST; the strict backbone and fusion checkpoints; matched fusion-control checkpoints; family fusion checkpoints; and strict DeepWukong checkpoints. See the root README for supported commands and limitations. Model checkpoints loaded by the portable current entry points contain tensor-only state dictionaries.

To materialize original OverflowGen C programs, slices and graph tables for native tools:

```bash
python scripts/prepare_data.py --mode full
```

This needs additional disk space. It checks the archive and refuses to overwrite differing files. The strict component split and historical family split are distinct; use the corresponding checkpoints and sample IDs. Saved per-sample predictions and numeric summaries are included in the source package under `data/current/` and `results/current/`.

Third-party code and datasets retain their respective notices and licenses; see [NOTICE.md](NOTICE.md). The reproduction package does not include pretrained language-model weights.
