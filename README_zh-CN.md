# CoNC 实验复现材料

本仓库对应 2026 年 11 月 14 日论文版本，包含 CoNC 方法、基线和消融代码、样本级预测、实验汇总，以及单独分发的数据和模型。CoNC 在数值与上下文骨干上增加精确算术通道和残差融合；该通道只对输入图中已知操作数、结果属于有符号 32 位范围的 COPY、ADD、SUB 求值，其他情况保持未知。

## 直接核验论文结果

```bash
python scripts/verify_bundle.py --part source --syntax
python scripts/current_experiment.py replay
```

该命令通过保存的逐样本预测重新计算 F1：严格划分 CoNC 为 97.39%，历史家族划分公共测试集为 97.06%，严格划分 H 为 98.01%，Trace+Eva+0 为 93.61%。这一步不需要 GPU，也不代表重新训练。

## 运行模型

在源码目录的上一级解压发行附件：

```bash
unzip overflow-reproducibility-data.zip
unzip overflow-reproducibility-current-assets.zip
# 严格划分基线和骨干消融权重：
unzip overflow-reproducibility-strict-baselines.zip
# 历史家族划分及比较方法权重：
unzip overflow-reproducibility-weights.zip
cd overflow-reproducibility
python -m pip install -r requirements-core.txt
python scripts/verify_bundle.py --part data
python scripts/verify_bundle.py --part current_assets
python scripts/verify_bundle.py --part strict_baselines
python scripts/current_experiment.py evaluate --seed 11 --arm exact_topo --device cpu
python scripts/current_experiment.py live --seed 11 --limit 32 --device cpu
python scripts/evaluate_strict_baseline.py --variant vd --seed 11 --limit 32
```

`evaluate` 使用冻结的图和上下文缓存对严格测试集完整重算融合分支；`live` 从已编码输入运行完整严格划分骨干和融合分支，默认验证 32 条样本。可将种子设为 11、29、47，分支设为 `exact_topo`、`literal_only` 或 `four_round`。若要从缓存的严格训练集和验证集重新训练融合分支：

```bash
python scripts/train_current_fusion.py --seed 11 --arm exact_topo --device cuda:0 --output-dir outputs/strict_exact_s11
```

严格划分基线附件还包含 VDTransformer、DeepDFA 和七种骨干消融的检查点。`evaluate_strict_baseline.py --help` 列出可直接运行的配置。

严格划分为 94,157 / 11,770 / 11,769，历史家族划分为 94,179 / 11,940 / 11,577，两者的检查点不能混用。部分旧版模型脚本对应此前没有精确算术分支的骨干模型；新版结果位于 `data/current/` 和 `results/current/`。重训原始骨干或完整重跑原生静态工具还需要相应环境及部分中间材料，详见英文说明。

数据来源、权限和引用见 [DATASETS.md](DATASETS.md)、[NOTICE.md](NOTICE.md)。
