"""Produce the RQ3 report and scientific figure from committed analysis outputs."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import campaign as c
from .rq3_access_types import OUT, GROUPS, NAMES, CHINESE, csv_read


def main():
    result = c.read(OUT / 'results.json')
    probe = c.read(OUT / 'strict_probe_v11.json')
    assert result['status'] == 'COMPLETE' and probe['status'] == 'PASS'
    rows = result['comparisons']; lookup = {r['access_group']: r for r in rows}
    differences = {(r['access_group'], r['comparator']): r for r in result['subgroup_intervals']}
    numeric = {(r['scope'], r['access_group']): r for r in result['numerical_coverage']}
    summaries = {(r['scope'], r['access_group'], r['arm']): r for r in csv_read(OUT / 'summary.csv')}
    display = [(g, n) for g, n in zip(GROUPS, NAMES) if lookup[g]['positive']]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7.7), gridspec_kw={'width_ratios': [1.1, 1]})
    colors = {'CoNC_f1': '#667085', 'literal_only_f1': '#277DA8', 'exact_topo_f1': '#D65A31', 'Eva_f1': '#48986C'}
    labels = {'CoNC_f1': 'Original CoNC', 'literal_only_f1': 'Literal-only fusion',
              'exact_topo_f1': 'ExactTopo v1.1', 'Eva_f1': 'Eva (saved baseline)'}
    positions = np.arange(len(display))
    for key, offset, marker in [('CoNC_f1', -.24, 'o'), ('literal_only_f1', -.08, 'o'),
                                 ('exact_topo_f1', .08, 'o'), ('Eva_f1', .24, 'D')]:
        ax.scatter([100 * lookup[g][key] for g, _ in display], positions + offset,
                   s=38, color=colors[key], marker=marker, label=labels[key], zorder=3)
    ax.set_yticks(positions, [f'{name}   (n={lookup[g]["n"]:,})' for g, name in display])
    ax.invert_yaxis(); ax.set_xlim(80, 101); ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_xlabel('Positive-class F1 (%)'); ax.set_title('Detection by target access form', loc='left', pad=14)
    ax.grid(axis='x', color='#E4E7EC'); ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(-.35, -.26), ncol=2, frameon=False)
    for comparator, offset, color, label in [('original', -.11, '#D65A31', 'ExactTopo minus original'),
                                             ('literal_only', .11, '#277DA8', 'ExactTopo minus literal-only')]:
        stats = [differences[(g, comparator)] for g, _ in display]
        point = np.array([r['difference'] for r in stats]) * 100
        lo = np.array([r['ci95_lower'] for r in stats]) * 100
        hi = np.array([r['ci95_upper'] for r in stats]) * 100
        # Draw intervals directly: percentile intervals need not contain the point.
        bx.hlines(positions + offset, lo, hi, color=color, linewidth=1.7)
        bx.scatter(point, positions + offset, color=color, s=34, label=label, zorder=3)
    bx.axvline(0, color='#667085', linestyle='--', linewidth=1)
    bx.set_yticks(positions, [name for _, name in display]); bx.invert_yaxis()
    bx.set_xlabel('F1 difference (percentage points)')
    bx.set_title('Paired family bootstrap: pointwise 95% intervals', loc='left', pad=14)
    bx.grid(axis='x', color='#E4E7EC'); bx.set_axisbelow(True)
    bx.legend(loc='lower left', bbox_to_anchor=(0, -.26), frameon=False)
    fig.suptitle('CoNC-ExactTopo v1.1 | First-paper RQ3', x=.07, ha='left', fontsize=18, fontweight='bold')
    fig.text(.07, .918, 'Same 11,342 common TEST slices; seeds 11 / 29 / 47; frozen models and threshold.', color='#475467')
    fig.text(.07, .025, 'Groups describe target syntax, not CWE labels. memset: 247 negative-only slices; positive-class F1 is NA.\n'
             'Intervals are exploratory (2,000 family draws), not adjusted for multiple subgroup comparisons.', fontsize=9, color='#475467')
    fig.subplots_adjust(left=.19, right=.98, top=.85, bottom=.24, wspace=.47)
    fig.savefig(OUT / 'rq3_comparison.png', dpi=200)
    fig.savefig(OUT / 'rq3_comparison.pdf')
    plt.close(fig)

    fmt = lambda value: 'NA' if value is None else f'{100 * value:.2f}'
    report = ['# CoNC-ExactTopo v1.1：第一篇 RQ3 分组测试', '', '## Material Passport', '',
              '- Origin Skill: academic-research-suite / experiment-agent，validate 模式。',
              '- Version: conc_exact_topo_v11_rq3；日期：2026-11-14。',
              '- Verification Status: VERIFIED（保存预测的重新统计、历史原模型逐组复核和 v1.1 数值通道重放通过；本轮未重训）。',
              '- 数据身份：第一篇历史 TEST，已曝光；按现有目标访问形式分组，属于回顾性描述分析。', '',
              '## 100% 的准确含义', '',
              '本轮直接调用 v1.1 的 `model.py::exact_states`，对原有 1,698 个探针图重新计算，先保存预测再读取参考值。'
              '7,178 个参考可计算 COPY/ADD/SUB 节点全部返回正确整数，7,507 个字面量也全部保持精确。'
              '三种子只用于对齐历史参考；确定性通道运行一次，不把它算作三次独立验证。', '',
              '| 对齐种子 | 原神经状态正确数 / 7,178 | v1.1 精确通道正确数 / 7,178 |',
              '|---|---:|---:|']
    for r in sorted((r for r in probe['scores'] if r['kind'] == 'computed'), key=lambda r: r['seed_alignment']):
        report.append(f'| {r["seed_alignment"]} | {r["original_neural_correct"]:,} | {r["v11_correct"]:,} |')
    total = numeric[('full11577', 'ALL')]
    report += ['', f'在完整 TEST 的全部 COPY/ADD/SUB 节点中，可确定值为 {total["computed_known"]:,} / {total["computed_total"]:,}，'
               f'覆盖率 {100 * total["computed_coverage"]:.3f}%。这些已知值在此前完整独立算术重放中全部一致。'
               '因此，100% 是“已支持且输入足够的计算节点”的整数精确率；不是所有表达式覆盖率、检测准确率或边界识别准确率。'
               '参考与实现共享给定图的数学契约，不证明原 C 程序的类型、路径、对象或跨过程绑定正确。', '',
               '## RQ3 使用的类型与样本', '',
               '当前 `paper/manuscript.tex` 的 RQ3 标题为 Performance across target access forms。'
               '沿用已保存且核验过的九类 target syntax 分组，原始数据没有逐样本 CWE 标签。'
               '主要表格复用七种原方法共同可评测的 11,342 个切片，含 4,961 个正例、6,381 个负例。'
               '新增版本与匹配对照的检查点、阈值 0.5 和预测全部固定，没有据分组结果再次选择模型。'
               '完整 11,577 条分组结果另存 CSV；例如 memmove 原有 386 条，仅 216 条在共同集内。', '',
               '## 同样本 F1 比较', '',
               '单位为 %；学习方法取种子 11/29/47 的均值，Eva 复用原确定性结果。差值单位为百分点。', '',
               '| 目标访问形式 | n / 正例 | 原 CoNC | 字面量融合 | ExactTopo v1.1 | 相对原模型 | Eva |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for group, chinese in list(zip(GROUPS, CHINESE)) + [('ALL', '总体')]:
        r = lookup[group]
        value = r['exact_minus_original_pp']
        delta = 'NA' if value is None else (f'{value:+.3f}' if 0 < abs(value) < .01 else f'{value:+.2f}')
        report.append(f'| {chinese} | {r["n"]:,} / {r["positive"]:,} | {fmt(r["CoNC_f1"])} | '
                      f'{fmt(r["literal_only_f1"])} | {fmt(r["exact_topo_f1"])} | {delta} | {fmt(r["Eva_f1"])} |')
    report += ['', '原论文另外六个基线的完整对比列保存在 `paper_rq3_comparison.csv`，没有重新运行静态工具。', '',
               '| 目标访问形式 | 精确版 − 原模型：差值 [95% CI] | 精确版 − 字面量融合：差值 [95% CI] |',
               '|---|---:|---:|']
    for group, chinese in list(zip(GROUPS, CHINESE)) + [('ALL', '总体')]:
        if not lookup[group]['positive']: continue
        vals = []
        for comp in ['original', 'literal_only']:
            r = differences[(group, comp)]
            vals.append(f'{100*r["difference"]:+.3f} [{100*r["ci95_lower"]:+.3f}, {100*r["ci95_upper"]:+.3f}]')
        report.append(f'| {chinese} | {vals[0]} | {vals[1]} |')
    report += ['', '区间按 family 配对重采样 2,000 次，每次内部对三个种子取均值；本轮有正例的各组均获得 2,000 个有效重采样结果。'
               '报告逐点 95% 区间，没有多重比较校正，分组结论是探索性的。'
               '区间跨零表示现有证据不足以确定该分组的差异方向；不把最好的分组当作总体或独立泛化结论。', '',
               'memset 共同集没有正例，因此正类 F1 标为 NA。其误报与特异度为：', '']
    for arm in ('original', 'literal_only', 'exact_topo'):
        r = summaries[('common11342', 'call_memset', arm)]
        report.append(f'- {arm}：平均 FP = {float(r["fp_mean"]):.3f} / 247，平均 specificity = {float(r["specificity_mean"])*100:.3f}%。')
    report += ['', '## 数值计算覆盖与精确率', '',
               '以下按同一共同集分组统计计算节点。全体已知值精确率均为 100%，但覆盖率因输入图的信息不同而变化。'
               '一条切片可以包含多个节点，节点数不能解释为独立程序数量。', '',
               '| 目标访问形式 | 已知计算 / 全部 COPY/ADD/SUB | 计算覆盖率 | 已知 ADD/SUB / 全部 ADD/SUB |',
               '|---|---:|---:|---:|']
    for group, chinese in list(zip(GROUPS, CHINESE)) + [('ALL', '总体')]:
        r = numeric[('common11342', group)]
        report.append(f'| {chinese} | {r["computed_known"]:,} / {r["computed_total"]:,} | '
                      f'{100*r["computed_coverage"]:.2f}% | {r["add_known"]+r["sub_known"]:,} / {r["add_total"]+r["sub_total"]:,} |')
    report += ['', '## 如何解释', '',
               '现有编码图中，memcpy、memmove、strcpy、strncpy 和 memset 分组没有 ADD/SUB 节点，其可计算状态全部来自 COPY。'
               '这限制了这些分组对加减算术能力的检验，不能把库函数分组的检测提升解释为精确加减运算带来的收益。', '',
               '分组检测表现并不等同于算术精确率。即使一个中间整数计算正确，检测仍依赖输入是否已知、'
               '对象与访问的绑定、边界信息以及融合分类器是否使用了正确证据。'
               '字面量融合对照用于区分额外参数、上下文融合训练与新增计算值的贡献；'
               '本轮不能从类型间差异推断精确算术对检测改善的因果机制。', '',
               '## 核验与统计解释检查', '',
               f'- {result["historical_original_confusion_checks"]} 组原模型分组混淆矩阵与历史 RQ3 逐项一致。',
               f'- {result["sklearn_confusion_and_f1_checks"]} 组新统计由 sklearn 混淆矩阵核算，有正例的组另核算 F1。',
               '- 所有预测哈希、检查点选择身份、共同集 ID、分组标签与 family 身份通过核对。',
               '- 完整数值覆盖来源于此前已独立验证的 TEST 状态；本轮另重放实际 v1.1 通道的历史 probe。',
               '- 未重训，未调阈值，未修改原始 RQ3 标签、模型和论文文件。', '',
               '11/11 项统计解释检查：', '',
               '| 检查 | 处理与限制 |', '|---|---|',
               '| Simpson 反转 | 同时展示总体及全部九组；不以总体方向替代每组方向 |',
               '| 生态谬误 | 切片、计算节点、family 单位分别报告 |',
               '| 选择偏差 | 保留历史共同集与完整集；不是新的随机外部样本 |',
               '| Collider 偏差 | 未按预测正确性筛选分组；共同可评测条件的限制已披露 |',
               '| 基础率忽略 | 每组报告正例数；memset 正类 F1 为 NA |',
               '| 均值回归 | 同样本固定预测及匹配对照，未仅选择原模型最差组 |',
               '| 幸存者偏差 | 全部九组及完整集结果保留，UNKNOWN 计入数值覆盖分母 |',
               '| 多重比较 | 报告所有组；逐点区间未经校正，不作确认性显著性宣称 |',
               '| 分析路径自由度 | 回顾性探索；分组沿用旧定义，新协议在本次统计前保存 |',
               '| 相关性当因果 | 不将算术覆盖与 F1 的组间关系解释为因果 |',
               '| 反向因果 | 预测和分组在此分析前已固定，不从检测结果重构标签 |', '',
               '## 文件', '',
               '- `rq3_comparison.png` / `.pdf`：分组 F1 与配对差异区间。',
               '- `paper_rq3_comparison.csv`：第一篇七个原方法与两个新版本的共同集比较。',
               '- `per_seed.csv` / `summary.csv`：完整集与共同集的逐种子混淆矩阵、F1、precision、recall、specificity、accuracy。',
               '- `numeric_coverage.csv`：各组 COPY/ADD/SUB 的总量与精确可知覆盖。',
               '- `strict_probe_v11.json` / `strict_probe_predictions.jsonl`：实际 v1.1 数值通道重放。',
               '- `results.json` / `protocol.json`：输入哈希、统计口径和结果。', '',
               '复算：在项目根目录运行 `python -m external_eval.conc_exact_topo_20261114.rq3_access_types`，'
               '再运行 `python -m external_eval.conc_exact_topo_20261114.present_rq3`。使用既有 research_agent 环境，仅需 CPU。']
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    outputs = list(OUT.glob('*.csv')) + list(OUT.glob('*.png')) + list(OUT.glob('*.pdf'))
    outputs += [OUT / n for n in ['REPORT.md', 'results.json', 'protocol.json', 'strict_probe_v11.json', 'strict_probe_predictions.jsonl']]
    c.save_json(OUT / 'completion.json', {'status': 'COMPLETE', 'script_sha256': c.sha(__file__),
                'files_sha256': {p.name: c.sha(p) for p in outputs}})
    print(json.dumps({'status': 'COMPLETE', 'report': str(OUT / 'REPORT.md'), 'figure': str(OUT / 'rq3_comparison.png')}))


if __name__ == '__main__':
    main()
