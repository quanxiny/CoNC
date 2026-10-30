# 论文复现材料

对应论文 **Preserving Numerical Structure in Learned Buffer-Overflow Vulnerability Detection**。主方法 CoNC（Coupled Numerical and Contextual detector，历史代码名 Full）为带液态图细化器、第二阶段部分冻结的 `liquid_transfer_frozen`（v70）；v76 是删除细化器的消融。CoNC+Trace+Eva（H）是在 CoNC 之上组合条件地址证据和 Eva 告警的决策方法。

## 快速核对论文结果

在本目录运行，只有 Python 标准库依赖：

```bash
python scripts/verify_bundle.py --part source --syntax
python scripts/replay_results.py --output outputs/replay
python scripts/check_paper_tables.py
```

表格检查核对正文及补充材料中的 76 个平均 F1 数值；严格机制实验部分读取保存的汇总值。复算程序读取逐样本预测及固定比较集合，重算混淆矩阵、Precision、Recall、F1、Specificity 和配对正确率，并与随包参考结果核对。统计覆盖源域 11,577 个测试切片及 11,342 个共同切片、MIT 1,164 个注册场景及 796/1,059 个场景子集、448 个受控边界程序。无法处理的样本保留为 UNKNOWN。

## 代码、数据和模型

源码、数据、基线权重、H 继续训练权重分别打包；将附件解压到源码目录的上一级，所有文件会汇入同一个 `overflow-reproducibility/` 目录。本地已展开版本已经合并附件，无需再次解压。安装和运行命令见 [英文主页](README.md)。

- [主方法及内部消融](transformer_gru_v76_v2/README.md)：CoNC、VDTransformer、源骨干、表示控制、细化器控制、训练/推理入口及参数。
- [外部实验与基线](external_eval/README.md)：DeepWukong、DeepDFA、LineVul、静态分析器、机制消融、MIT 和 H。
- [数据说明](DATASETS.md)：原始数据、固定划分、缓存张量、来源和许可。
- `results/reference/`：参考指标、数值状态检查和配对结果；`manifests/`：各组件逐文件 SHA-256。

复现主方法的最小命令：

```bash
python -m pip install -r requirements-core.txt
python transformer_gru_v76_v2/scripts/smoke_core.py --with-assets
python transformer_gru_v76_v2/scripts/evaluate_core.py \
  --model conc --seed 11 --split test --device cpu --limit-rows 2 \
  --output-dir outputs/full_smoke
python transformer_gru_v76_v2/scripts/train_full.py \
  --arm conc --seed 11 --output-dir outputs/full_s11
```

最后一条默认仅显示训练参数，增加 `--execute` 才启动训练。正式实验分别使用 11、29、47 三个种子。训练需要 CUDA；小规模推理和结果复算可在 CPU 运行。原始数据解压：`python scripts/prepare_data.py --mode full`。

源代码可放入 GitHub 普通仓库；大数据和权重应作为 Release 附件或存入数据仓库，`.gitignore` 已排除这些二进制资产。保留第三方版权、数据集出处、许可和实验边界。包内不含论文写作文件、专利文件、私人目录、协作记录和训练日志。

本次完成了源码/校验检查、CPU 架构检查、真实权重小样本推理、输入张量一致性检查及保存结果复算；未重新进行完整 GPU 训练或原生工具全量分析。历史主表和严格机制实验采用不同划分，应按各自协议复现。额外第三方工具及预训练资源按实验说明单独安装或获取。

论文名称与历史代码标识的对应关系见 [实验命名表](experiment_names.json) 和[英文主页](README.md#names-used-in-the-paper-and-in-saved-artifacts)。其中，`fallback_weighted` 是论文的回退加权继续训练，兼容历史训练标识 `residual_weighted`；它与 `residual_refiner` 架构消融是两项不同实验。严格机制实验中的 CoNC 使用独立训练的严格划分权重。

最新版新增的案例分析可通过 `python scripts/analyze_disagreements.py` 复算保存的 CoNC/Eva 分歧计数，通过 `python scripts/plot_case_studies.py` 重绘图 6。绘图需要安装 `matplotlib`，字体外观可能与论文略有不同。案例是回顾性选取的说明材料，不代表错误的总体分布。
