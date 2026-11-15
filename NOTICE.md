# Attribution and license scope

Original contributions are copyright (c) 2026 Quanxin Yang and are distributed under the MIT license in `LICENSE`. This license does not replace upstream code, model or dataset terms.

- **OverflowGen and VDTransformer**: Yang Cao, Yunwei Dong and Jiajie Peng, “Vulnerability Detection Based on Transformer and High-Quality Number Embedding,” [doi:10.1002/cpe.8292](https://doi.org/10.1002/cpe.8292). The [official code/data item](https://figshare.com/s/fbaf47e3ac2a9581dbd7) declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original archive is unchanged; derived tensors, numerical embeddings and source models retain this attribution. Recorded SHA-256 values are local checksums, not publisher-certified checksums.
- **DeepWukong, DeepDFA and LineVul**: upstream licenses and relevant source headers are retained beside the included source and under `licenses/`. Source locations and adaptations are described in `external_eval/README.md`. Third-party authorship strings are part of the attribution.
- Some upstream files contain Apache-2.0 and Microsoft MIT notices; corresponding license texts remain under `licenses/`.

Release adaptations provide relative paths, compact frozen-input readers, standalone training/inference entry points and normalized result files. The reference predictions and dataset split memberships preserve their original values. The accompanying manifests identify the exported bytes. Neural checkpoints included as optional attachments were trained on the attributed synthetic source data. No third-party pretrained language-model weights are included.
