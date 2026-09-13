---
### 全局风格与格式控制锚点（Global Style Anchor）

在将 Prompt 输入 Midjourney、DALL-E 3 或 SVG 生成工具时，请务必保持以下统一前缀：

> **Global Style Anchor:**
> `Academic publication vector illustration, IEEE Transactions on Smart Grid standard style, clean engineering design, vector plot, sharp line weights, crisp rendering, professional color scheme: deep navy (#1f77b4), forest green (#2ca02c), safety orange (#ff7f0e), crimson red (#d62728), dark slate (#333333). High contrast, white background, LaTeX style variables and mathematical annotations, highly reproducible, no photorealistic textures, no gradient background, no glossy 3D elements, flat technical schematic`

---

### 25 张 SCI 投稿级图表 Prompt 词库

#### 第一部分：全局架构与理论主线（Fig 1 – Fig 2）

**Fig 1: Hierarchical Multi-Time-Scale Energy Management Architecture**

* **用途**：全文整体技术架构与多时间尺度信息流闭环图（System Topology）。

* **Prompt**:

> `Global Style Anchor, A hierarchical multi-time-scale control architecture diagram for a microgrid. Top block: Day-Ahead Q1 Convex Graph-LP Dispatch with KKT dual price extraction. Middle block: Receding Horizon Q3 Information Value (NAV) re-optimization module driven by MaxEnt-ACI forecast bounds. Bottom block: Real-time 10-minute Q4 MT-RRC controller executing safe actions on a BESS battery with Student-t Copula, CBF safety filter, and DETC. Directed functional vector arrows with precise LaTeX labels (g_t^*, \lambda_t^*, NAV_k, u_\tau^{safe}, u_\tau^{real}). Technical system workflow.`

**Fig 2: Information–Decision–Control Cascade Conceptual Schema**

* **用途**：全文 Conceptual Anchor，展露 Physical $\rightarrow$ Risk $\rightarrow$ Information $\rightarrow$ Control 四大价值递进。

* **Prompt**:

> `Global Style Anchor, A conceptual cascade diagram showing 4 horizontal connected value stages: 1. Physical Value (Deterministic Convex Graph-LP Optimization & KKT Dual Shadow Prices), 2. Risk Value (MaxEnt Distribution, ACI Adaptive Conformal Defense, & Newsvendor Critical Quantile), 3. Information Value (Lead-Time Forecast Shrinkage, Endogenous No-Adaptation Band, & NAV Re-Optimization), 4. Control Value (High-Frequency MT-RRC, Student-t Copula, CBF Safety Barrier Invariant Set, & DETC Event-Triggered Execution). Elegant vector block style, labeled node icons, high-level theoretical conceptual illustration.`

---

#### 第二部分：Q1 物理建模与确定性凸基准（Fig 3 – Fig 7）

**Fig 3: Physical Topology of the Microgrid Power System**

* **用途**：微电网单线图（SLD）与高保真物理约束映射。

* **Prompt**:

> `Global Style Anchor, Single-line diagram (SLD) of a grid-connected industrial microgrid system. Central AC bus bar N_t connecting a PV solar farm v_t, a Battery Energy Storage System (BESS) internal node B_t with inverter/charge-discharge efficiencies (\eta_c, \eta_d), critical industrial load l_t, and Point of Common Coupling (PCC) transformer to the main utility grid g_t. Bi-directional vector arrows indicating real power balance variables (g_t, v_t, q_t, c_t, d_t, l_t). Inset schematic showing battery internal electrochemical capacity constraints and degradation cost \kappa_{deg}.`

**Fig 4: PV Inverter Physical Clipping & Power Envelope**

* **用途**：光伏逆变器物理截断与功率包络曲线。

* **Prompt**:

> `Global Style Anchor, A 2D scientific response curve plot. X-axis: Solar Irradiance (W/m^2). Y-axis: PV AC Power Output P_{AC}^{PV} (kW). Multiple line curves representing different operating ambient temperatures. Horizontal dashed crimson line clearly marking the maximum inverter physical power cap S_{inv}^{max}. Shaded region highlighting inverter power clipping saturation domain.`

**Fig 5: Time-Expanded Graph Flow & BESS State Transition**

* **用途**：广义三层网络流（Graph-LP）与 SOC 状态转移图。

* **Prompt**:

> `Global Style Anchor, Directed acyclic time-expanded generalized network flow diagram for energy storage over 24 discrete time steps. Nodes represent AC bus balance N_t and battery SoC states B_t at discrete time steps t=0 to 23. Directed edges represent continuous charging (c_t with efficiency \eta_c) and discharging (d_t with efficiency 1/\eta_d) decision transitions. Clear graph theory representation illustrating convex dynamic balance and self-discharge decay \gamma.`

**Fig 6: Optimal Day-Ahead Deterministic Energy Balance & SOC Trajectory**

* **用途**：1h 分辨率下的 24h 确定性能量平衡与储能 SOC 轨迹（无虚假环流，满足 $E_T = E_0$）。

* **Prompt**:

> `Global Style Anchor, Dual-panel time-series scientific line chart for 24 hours at 1-hour resolution (24 time intervals). Top panel: Stacked power generation bar chart showing PV generation (v_t - q_t), Utility Grid Purchase g_t, and BESS Discharge d_t balancing Total Load l_t and BESS Charge c_t. Bottom panel: Battery State of Charge (SoC) line trajectory bounded by strict dashed limit lines at SoC_{min} and SoC_{max}, achieving exact start-end equilibrium E_T = E_0.`

**Fig 7: KKT Dual Shadow Price and Marginal Energy Value Profile**

* **用途**：KKT 对偶乘子 $\mu_t$（母线影子供应价格）与 $\lambda_t$（储能内部状态边际价值）时序轨迹与充放电死区解释。

* **Prompt**:

> `Global Style Anchor, Dual-axis step-line plot over 24 hours. Primary Y-axis: Grid electricity purchasing tariff p_t alongside optimal nodal power balance dual multiplier \mu_t. Secondary Y-axis: BESS state marginal shadow price \lambda_t. Vertical shaded background regions emphasize arbitrage intervals where shadow price \mu_t triggers charging (\mu_t \le \eta_c \lambda_{t-1} - \kappa_{deg}) or discharging (\mu_t \ge \lambda_{t-1}/\eta_d + \kappa_{deg}), highlighting the central KKT no-action deadband.`

---

#### 第三部分：Q2 风险防守与自适应共形预测（Fig 8 – Fig 13）

**Fig 8: Forecast Error Empirical Distribution & MaxEnt Prior Fit**

* **用途**：最大熵（MaxEnt）变分凸对偶残差拟合与标准假设分布对比。

* **Prompt**:

> `Global Style Anchor, Statistical probability distribution plot. Histogram of historical PV forecasting residual errors overlayed with theoretical probability density function (PDF) curves: Standard Gaussian (dashed blue), Weibull (dotted orange), and proposed Maximum Entropy MaxEnt exponential-family distribution p*(x) (solid green). Inset table displaying Kolmogorov-Smirnov (KS) test p-values confirming heavy-tail fit superiority of MaxEnt.`

**Fig 9: ACI Adaptive Conformal Forecast Fan Chart & Defensive Lower Bound**

* **用途**：ACI 动态共形预测包络与 80% 临界分位数（Newsvendor $F^*=0.80$）物理截断防守下界 $P_{t,L}^{PV}$。

* **Prompt**:

> `Global Style Anchor, Time-series uncertainty fan chart for PV generation over 24 hours. Gradient shaded confidence bands representing dynamic ACI conformal prediction intervals. Solid black line represents actual realized PV output \hat{P}_t^{PV}. A prominent bold crimson line highlights the lower defensive power threshold P_{t,L}^{PV} = \max(0, \hat{P}_t^{PV} - r_t) truncated strictly at zero for 80% Newsvendor coverage defense against 5x penalty risk.`

**Fig 10: Lead-Time-Dependent Prediction Uncertainty Expansion**

* **用途**：预报 Lead-time 依赖的“喇叭口”不确定性膨胀曲线（Q2-Q3 桥梁）。

* **Prompt**:

> `Global Style Anchor, A 2D scientific line chart showing forecast uncertainty interval width vs forecast lead time h (hours from 1h to 24h). Multiple curves correspond to different base forecasting models. Shows clear monotonic divergence (funnel expansion) as lead time increases. Annotated vector arrow showing epistemic uncertainty growth.`

**Fig 11: Empirical Coverage Calibration Plot & ACI Path Identity**

* **用途**：ACI 自适应共形预测覆盖率校准对齐曲线与路径收敛轨迹。

* **Prompt**:

> `Global Style Anchor, Model calibration reliability diagram and convergence plot. Main panel X-axis: Target nominal coverage rate (1-\alpha) from 0.5 to 0.99. Y-axis: Empirical observed coverage rate. Diagonal black dashed line represents perfect ideal calibration (y=x). Plotted curves compare Static Quantile method, Gaussian Assumption, and proposed ACI perfectly aligning along y=x. Inset panel: Empirical coverage error converging to zero over time via ACI path identity.`

**Fig 12: Risk–Cost Pareto Frontier under Asymmetric Penalty Ratio (1x vs 5x)**

* **用途**：报童模型下 5 倍非对称惩罚（$C_u=4p_t, C_o=p_t$）对应的期望成本与尾部风险 Pareto 响应曲线。

* **Prompt**:

> `Global Style Anchor, Pareto trade-off frontier scatter plot under 5x asymmetric imbalance penalty. X-axis: Expected Daily Operating Cost ($). Y-axis: CVaR 95% Tail Realtime Penalty ($). Individual marker points represent different decision models (Deterministic Mean Forecast, Gaussian-Quantile, Proposed MaxEnt-ACI Defensive LP). Golden star annotations highlight the optimal critical quantile point at F* = 0.80.`

**Fig 13: Scenario-Wise Tail Loss Distribution & Settlement Penalty**

* **用途**：1000 次 Monte Carlo 极端场景下的实时不平衡结算惩罚尾部风险（Violin Plot）。

* **Prompt**:

> `Global Style Anchor, Comparative Violin and Boxplot chart showing real-time grid 5x imbalance settlement penalty distributions across 1000 Monte Carlo weather scenarios. Comparing Deterministic Dispatch, Gaussian Distribution, and Proposed MaxEnt-ACI Defensive Dispatch. Highlighted long upper tails for baseline models contrasting with tightly bounded, truncated tail losses for the proposed strategy.`

---

#### 第四部分：Q3 信息价值（NAV）与事件驱动重规划（Fig 14 – Fig 19）

**Fig 14: Sequential Information Revelation & Funnel Shrinkage**

* **用途**：多节点（0:00, 6:00, 12:00, 18:00）信息到达后的预测带“喇叭口”收缩与不确定性衰减。

* **Prompt**:

> `Global Style Anchor, Multi-panel time-series forecast update plot at discrete information arrival nodes T_k \in {00:00, 06:00, 12:00, 18:00}. Shows the dynamic shrinking of prediction uncertainty bands (funnel shrinkage effect) as the target operating interval approaches, illustrating sequential reduction of variance upon new forecast arrival.`

**Fig 15: Endogenous No-Adaptation Zone under Asymmetric Adjustment Cost**

* **用途**：非对称契约调整成本（$1.5p_t / 0.5p_t$）导出的 KKT 次梯度内生无调整带 $\partial C^{adj} \in [-0.5p_t, 1.5p_t]$。

* **Prompt**:

> `Global Style Anchor, Analytical piecewise linear diagram showing marginal contract adjustment cost vs forecast error deviation. Clear visual highlighting of the central "Endogenous No-Adaptation Zone" where subgradients \partial C^{adj} \in [-0.5p_t, 1.5p_t] contain the marginal benefit m_t, creating an intrinsic mathematical damping band with zero contract alteration.`

**Fig 16: Dynamic Contract Adjustment Policy Region Map**

* **用途**：(边际收益 $m_t$ vs 实时基准电价 $p_t$) 的二维契约调整决策相图。

* **Prompt**:

> `Global Style Anchor, A 2D decision boundary phase map. X-axis: Forecast Deviation Marginal Benefit m_t. Y-axis: Electricity Price p_t. The space is partitioned into 3 distinct color-coded topological regions: Downward Contract Adjustment (m_t < -0.5p_t), No-Adaptation Dead-band (Keep, -0.5p_t \le m_t \le 1.5p_t), and Upward Contract Adjustment (m_t > 1.5p_t).`

**Fig 17: Forecast Accuracy (CURI) vs Net Adjustment Value (NAV)**

* **用途**：揭示 **预测精度提升 $\neq$ 决策调整价值** 的关键散点与非线性响应图。

* **Prompt**:

> `Global Style Anchor, 2D scientific scatter plot with non-linear trendline. X-axis: Continuous Uncertainty Reduction Index (CURI_k, forecasting precision gain). Y-axis: Net Adjustment Value (NAV_k = J_k^{keep} - J_k^{reopt}, financial improvement in $). Demonstrates that high forecasting accuracy gains do not automatically produce positive NAV, highlighting decision-aware threshold filtering.`

**Fig 18: Event-Driven NAV Trigger Trajectory & Dynamic Re-Optimization**

* **用途**：全天事件驱动触发轨迹，显示 NAV 点位与 Re-opt/Keep 标定及夜间 Mask 屏蔽。

* **Prompt**:

> `Global Style Anchor, Time-series impulse bar chart over 24 hours across forecasting nodes T_k. Bar heights represent computed Net Adjustment Value (NAV_k). Horizontal dashed crimson line marks the execution threshold barrier (NAV > \delta). Vertical callout markers distinctly label triggered REOPT points versus suppressed KEEP points, with a gray shaded block over nighttime hours (19:00-05:00) indicating valid PV mask truncation.`

**Fig 19: Comparative Trajectory Array across Subsets S0–S5**

* **用途**：S0~S5 对比子集下的多时间尺度契约修正轨迹与储能基准 SOC Trajectory。

* **Prompt**:

> `Global Style Anchor, Two-panel comparative trajectory plot across multi-forecast subsets S0 (baseline 00:00 only), S3 (full 4-node MPC), S4 (NAV adaptive trigger), and S5 (Oracle perfect forecast). Panel (a): Grid contract schedule g_t^{(k)} evolutions. Panel (b): Corresponding BESS State of Charge (SoC) reference evolution S_t^{ref}, demonstrating dynamic contract re-optimization cost-saving performance.`

---

#### 第五部分：Q4 实时控制、CBF 安全屏障与 DETC（Fig 20 – Fig 24）

**Fig 20: MT-RRC Hierarchical Control & Multi-Time-Scale Variable Mapping**

* **用途**：1h $\to$ 10min 多时间尺度变量传递（契约映射 $P_\tau^{contract}$、SOC 参考线性插值 $S_\tau^{ref}$）与闭环控制结构。

* **Prompt**:

> `Global Style Anchor, Layered block schematic diagram for the Multi-Time-Scale Risk-Aware Closed-Loop Control (MT-RRC). Top level: 1-hour Day-Ahead Contract g_t^{final} and Reference S_t^{ref}. Interface level: Interpolated 10-minute baseline P_\tau^{contract} and S_\tau^{ref}. Decision level: Student-t Copula & Soft-MPC. Safety level: CBF Filter. Execution level: DETC controller executing actions on BESS hardware.`

**Fig 21: Student-$t$ Copula High-Frequency Joint Risk Scenario Set**

* **用途**：利用 Student-$t$ Copula 捕捉（低 PV + 高负荷 + 电价 Spike）极端尾部相关的高频场景生成图。

* **Prompt**:

> `Global Style Anchor, 3D scatter plot and joint marginal distribution curves showing 10-minute real-time residual scenarios generated by Student-t Copula. Axes represent PV Residual \varepsilon^\text{PV}, Load Residual \varepsilon^L, and Real-time Price Residual \varepsilon^p. Highlighted red tail-dependence clusters showing concurrent extreme conditions ("Low PV + High Load + High Price Spike") for robust CVaR calculation.`

**Fig 22: Control Barrier Function (CBF) Phase-Space Safety Filtering**

* **用途**：(SOC vs 充放电功率) 相平面下的控制不变集与 CBF 强行二次规划（QP）拉回效应。

* **Prompt**:

> `Global Style Anchor, Phase-space trajectory diagram. X-axis: Battery Energy State E_\tau. Y-axis: Control Power Action u_\tau. Shaded safe invariant set bounded by CBF dynamic condition \dot{h}(E,u) + \gamma h(E) \ge 0. Vector arrows show nominal unconstrained MPC trajectory u_\tau^* attempting to exit the safe SOC boundaries, which is orthogonally projected along the smooth boundary to u_\tau^{safe} by the CBF quadratic program safety filter.`

**Fig 23: Dynamic Event-Triggered Control (DETC) Execution & Threshold Evolution**

* **用途**：高频控制下的触发误差 $\Vert e_\tau \Vert_2^2$ 与动态阈值演变，展现开关动作休眠与更新。

* **Prompt**:

> `Global Style Anchor, Two-panel 10-minute resolution execution plot over 144 steps. Upper panel: Continuous control tracking error magnitude ||e_\tau||_2^2 compared against dynamic time-varying threshold curve. Vertical impulse markers indicate sparse DETC control execution instances u_\tau^{real} = u_\tau^{safe}. Lower panel: BESS discrete real power step function showing extended hold periods, illustrating reduced PCS switching chatter.`

**Fig 24: Cost–Switching Frequency–Safety Violation Multi-Objective Ablation Map (A0–A4)**

* **用途**：消融实验 A0–A4 散点图，对比全功能 MT-RRC、无 Copula (A1)、无无调整带 (A2)、无 CBF (A3)、无 DETC (A4) 在总成本、SOC 越界频次和开关次数上的表现。

* **Prompt**:

> `Global Style Anchor, 2D scatter trade-off plot with bubble sizing representing ablation studies A0 to A4. X-axis: Number of Actuator Switches N_{switch}. Y-axis: Total Real-time Operating Cost ($). Bubble size represents Maximum SOC Hard Limit Violation Depth. Clearly illustrates full model A0 achieving minimal cost, zero SOC violation, and low switching frequency compared to non-CBF (A3) and non-DETC (A4) baselines.`

---

#### 第六部分：全局消融与系统级总结（Fig 25）

**Fig 25: Global Performance Parallel Coordinates Map across Architectures M0–M4**

* **用途**：全篇论文终极总结图，平行坐标系展示 M0–M4 在 6 大指标上的归一化性能包络。

* **Prompt**:

> `Global Style Anchor, Parallel coordinates plot comparing 5 model architectures (M0: Deterministic Graph-LP Baseline, M1: +MaxEnt-ACI Conformal Defense, M2: +NAV Information Value Re-Opt, M3: +MT-RRC High-Frequency Soft-MPC, M4: Full Proposed CBF-DETC Closed-Loop) across 6 normalized vertical metric axes: Total Operating Cost, Imbalance Penalty, CVaR Risk, Renewable Curtailment, Battery Degradation, and PCS Switching Chatter. Polyline paths clearly show continuous performance improvement envelope from M0 to M4.`