# C题《微网与外部电网电力调控策略》— 交付说明

本目录是"按提示词实现 + 建模 + 出表 + 制图 + 自审"的完整交付包。所有产物均可由
`bash run_all.sh` 在单机 CPU 上约 2 分钟重跑，且重跑结果逐位一致。

## 1. 直接可用的交付物

| 交付物       | 位置                                                                                             | 说明                                                                                                     |
| --------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 五个结果文件    | `output/result1.xlsx`、`result2.xlsx`、`result3.xlsx`、`result4-2.xlsx`、`result4-3.xlsx`          | 严格复刻附件 5 模板：工作表名、表头、时段标签逐字一致；日期为 Excel 日期；`⁝` 省略行已删除并扩展为 334 天                                         |
| 论文表 1–表 4 | `paper/表1-表4.md`                                                                               | 表1/表2（问题1）、表3（四个指定日期）、表4（紧急购电填写示例）                                                                     |
| 交付图集       | `figures/Fig01_*.pdf` … `figures/Fig25_*.pdf`                                                   | 按 `figures/CAPTIONS_3.md` 规格生成的 25 张 SCI 投稿级图（架构 2 + Q1 5 + Q2 6 + Q3 6 + Q4 5 + 总结 1），**只出矢量 PDF** |
| 图注与索引      | `figures/CAPTIONS.md`、`figures/captions3_index.json`                                            | 每张图的中文图注、英文标题、所属分组与数据口径说明                                                                     |
| 绘图数据证据包    | `figures/science_data.json`                                                                    | 每张图引用的数字来源（results/、clean/ 与求解器纯函数复用），供审查追溯                                                      |
| 图集生成程序     | `python3 -m code.report.science_figs --set all`                                                | 25 张图的唯一生成入口，可分组重跑（`arch,q1,q2,q3,q4,summary`）；不修改求解器                                    |
| 旧图集备份      | `figures_legacy_backup/`                                                                       | 替换前的 35 张出版级图（含 PNG 已删的 PDF 版），可随时回滚                                                            |
| 论文数字登记表   | `results/paper_numbers.json`                                                                   | 每个数字含数值、单位、来源文件、生成命令；论文禁止手抄                                                                            |
| 自审报告      | `REVIEW.md`                                                                                    | G0–G8 门禁逐条核验 + 自查修复记录 + 诚实的负面结果                                                                        |
| 不确定项处理    | `paper/不确定项与C题.pdf核对.md`、`results/uncertainty_ranges.json`、`figures/Fig09_aci_fan_lower_bound.pdf`、`figures/Fig11_coverage_calibration.pdf` | 能以 `C题.pdf` 确定的用原文；未明确项（U1–U5）用共形预测给区间                                                                 |

## 2. 目录结构

```text
code/cleaning_cpp/     C++17 数据清洗（zip+XML 读取、NR1–NR12、审计、隔离、.npy 导出）
code/common/           只读 clean 载入、xlsx 模板读写、门禁自检、确定性审计
code/forecast/         轻量预测、共形（split/加权/ACI）、覆盖率标定
code/optimize/         Q1 LP+DP、Q2 两阶段随机规划、Q3 滚动 MPC、Q4 价格风险
code/report/           五个结果文件、图表、论文数字与表 1–表 4
clean/                 C++ 清洗产物（含 manifest.json / SHA-256 / 审计 / 隔离 / 窗口 / σ）
results/               JSON 结果、场景数组、p2 覆盖报告、门禁报告、确定性报告
figures/               交付图集 Fig01–Fig25（矢量 PDF）+ 图注 + 数据证据包
figures_legacy_backup/ 替换前的 35 张图集（回滚用）
paper/                 论文表 1–表 4
templates/附件5/        附件 5 模板备份（原始空模板）
output/                五个最终结果文件
tasks/Q1.md … Q4.md    分问任务卡（模型、对照、清单）
TASK.md                总任务与门禁索引
```

## 3. 重跑方式

```bash
# 依赖：g++ 15 / cmake 3.31 / zlib / tinyxml2；python3 + numpy/scipy/matplotlib
cmake -S code/cleaning_cpp -B build/cleaning -DCMAKE_BUILD_TYPE=Release
cmake --build build/cleaning -j
./build/cleaning/cleaning_cpp selftest          # 规则黄金用例 + 200 组性质测试
bash run_all.sh                                  # 全流程（约 2 分钟）
```

分阶段命令：

```bash
./build/cleaning/cleaning_cpp clean  --in Data --out clean --templates templates/附件5
./build/cleaning/cleaning_cpp verify --in Data --out clean
python3 -m code.optimize.q1_lp          # Q1：35,126.95 元
python3 -m code.forecast.calibrate      # 共形覆盖率 0.80/0.25/0.70
python3 -m code.optimize.q2_stochastic  # Q2 全年回溯
python3 -m code.optimize.q3_mpc         # Q3 滚动 MPC + S0–S5
python3 -m code.optimize.q4_price       # Q4（附件4 电价 + CVaR/Copula/软约束）
python3 -m code.report.write_results    # 写出五个 result 文件
python3 -m code.report.science_data     # 生成 figures/science_data.json（只读数据包）
python3 -m code.report.science_figs --set all   # 生成 figures/Fig01–Fig25
python3 -m code.report.paper_numbers    # 论文表 1–4 + 数字登记
python3 -m code.common.checks           # G0–G8 门禁 + 敏感性
python3 -m code.common.determinism      # 重跑一致性（逐位）
```

## 4. 四个问题的核心结论

| 问题  | 模型                                                                   | 结果                                                                                   |
| --- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Q1  | 时域扩展网络流等价 LP（命题 1 保证充放互斥可松弛）+ SoC DP 互证                              | 35,126.95 元/日，较"光伏自用+弃光+无储能"基线 48,052.05 元降 26.90%；夜间贡献 51.4% 充电、76.0% 放电            |
| Q2  | 两阶段随机规划（残差块 bootstrap 场景 + 报童结构 p\*=0.80）                            | 全年 17,540,872.74 元；紧急购电 426,470.2 kWh（时段触发率 20.78%，属结构性缺额）                           |
| Q3  | 0:00/6:00/12:00/18:00 四时次滚动 MPC，计划层 0.25 / 调整层 0.70，含 0.5×/1.5× 违约结算 | 全年 14,705,624.88 元；6:00 时次价值 612.95 万元，12:00 为 43.32 万元，18:00 约为 0 → 需要更多时次但应加在上午/午后 |
| Q4  | 实时电价（附件4）重算 + 共形尖峰上界 + CVaR + 软约束 MPC                                | Q4-2 17,838,552.18 元、Q4-3 15,384,235.06 元；尖峰覆盖 94.11%；β = 0.9 时 CVaR₉₀ 下降 13.9%      |

## 5. 语言分工与红线遵守情况

> **数字勘误（2026-09-12 重写论文时核验）**：本文件第 4 节的 Q2 = 17,540,872.74 元、
> Q4-2 = 17,838,552.18 元为 Phase 12 之前的旧值。按“结果以 `output/` 为准”的口径，
> 现行结果文件与 `results/q*.json` 给出：**Q2 全年 17,469,237.66 元**
> （紧急购电 363,622.65 kWh、时段触发率 18.62%）、**Q4-2 全年 17,727,459.43 元**
> （紧急购电 360,039.2 kWh）、Q3 14,705,624.88 元、Q4-3 15,384,235.06 元、
> Q1 35,126.95 元/日。上述数字已由独立只读审计复算通过
> （`python3 -m code.common.audit_output` → `results/output_audit.json`，60/60 PASS）。
> 图集说明：`figures/` 现含汉化后的 Fig01–Fig25 与论文用灵敏度复合图
> `FigS_sensitivity_bundle.pdf`；`figS1`–`figS3` 为灵敏度/统计补充图（中文标注）。

- 值级修改（缺失判定、异常隔离、窗口推断、口径归一）全部在 `code/cleaning_cpp/` 完成；Python 仅在断言层复核不变量 I1–I5（`code/common/load_clean.py`），不重实现清洗逻辑。
- xlsx 读写用标准库 `zipfile` + XML 自实现（`code/common/xlsx_template_io.py`），未安装或依赖 openpyxl；openpyxl 只用于独立只读交叉核对。
- 信息集纪律：0:00 计划只用严格早于当日的窗口；6:00/12:00/18:00 只用对应时次及更早的信息；完美预报（S5、oracle）仅作界，不参与任何策略口径。
- 零随机性：场景削减为确定性分层抽样，全流程同输入重跑逐位一致（`results/determinism.json`）。
