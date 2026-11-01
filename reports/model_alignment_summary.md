# 第7轮：跨引擎模型对齐与力矩/接触抽查

## 结论摘要

- goal 的差距主要出现在 2~4 秒腿部运动，最大单关节差为 `l_thigh`，不是只发生在收尾。
- 静态 URDF/MJCF 对齐显示：除 base_link 的 body 聚合方式外，绝大多数 link 质量与惯量特征值接近，关节轴和关节范围一致。
- `pi_auto.xml` 是 urdf2mjcf 生成文件，原始 XML 没有 damping/armature/frictionloss；这些由评估脚本运行时注入。
- 自动 MJCF 的 base_link 质量显著大于 URDF base_link，这是**body 合并差异**，不能直接当作机器人总质量差异；必须做总质量/质心聚合后再下结论。
- 当前力矩抽查不是严格同状态：Isaac 数据是 `computed_torque_estimate`，不是 PhysX 实测 applied torque；因此只作为量级线索。
- 本轮没有找到单个简单参数解释全部 gap；下一步应做严格同状态执行器力矩实验与腿部几何/质量对齐。

## 静态模型对齐

产物：`static_alignment.json`、`static_alignment.md`。

### 质量/惯量

- URDF 与 MJCF 的大多数 link 质量差异接近 0，l/r ankle roll 的惯量特征值相对差异约 15%，需要重点核对坐标变换和惯性张量。
- **已确认一个真实模型 bug（本轮最重要的发现）**：`pi_auto.xml` 的 `base_link` **没有 `<inertial>`**，质量由碰撞几何密度算出，导致 `base_link` 10.52 kg（URDF 2.76 kg），**整机总质量 18.76 kg，而 URDF/Isaac 只有 11.00 kg（+70%）**。
  - 佐证：第 1 轮 `recheck_20261007/results/model_info.json` 记录的 `robot_mass = 18.755` → **从第 1 轮起所有 MuJoCo 评估都在用这个 18.76 kg 的错误模型**。
  - 手写 MJCF `pi_20dof.xml` 总质量 11.18 kg（base 3.45 kg），与 URDF 接近。
- **已做修复验证**：给 `base_link` 补上 URDF 惯性（mass 2.7616、fullinertia）后总质量回到 11.0002 kg，重跑 goal：
  - 原段 MAE 0.096409 → **0.095198**
  - 基座 up error max 11.078° → **12.019°**
  → **修正质量后 goal 的引擎差距几乎没有变化，质量 bug 不是 gap 的原因，但必须修正以保证模型正确性。**
- 关节轴与关节范围在静态解析中一致。

### 碰撞几何

URDF/MJCF 的类型大体对应：box→MuJoCo type 6，mesh→MuJoCo type 7。但仍需核对具体尺寸和接触属性，重点为：

- l/r ankle pitch：URDF 多个 box collision，MJCF 多个 box geom；
- l/r ankle roll：URDF mesh，MJCF mesh；
- l/r calf：mesh + box 混合；
- l/r thigh：mesh；
- Isaac 使用 `replace_cylinders_with_capsules=True`，本模型是否实际存在受此开关影响的 collision primitive 还需按 geom 清单确认。

## 用修正质量模型复验（重要）

采用 `pi_auto_fixedmass.xml`（总质量 11.0002 kg）重跑之前“通过”的动作：

| 动作 | 错误质量(18.76kg) MuJoCo MAE | 修正质量(11.0kg) MuJoCo MAE | Isaac MAE | 修正后 up error | 判定 |
|---|---:|---:|---:|---:|---|
| bigkick | 0.060695 | **0.052340** | 0.052793 | 2.88° | 通过 |
| kick1015 | 0.062478 | **0.053173** | 0.052754 | 2.71° | 通过 |
| goal | 0.096409 | 0.095198 | 0.079117 | 12.02° | 通过 |

关键结论：

- **bigkick / kick1015 的质量 bug 修正后，MuJoCo MAE 从比 Isaac 高约 15~18% 变为基本相等**（0.0523/0.0532 vs 0.0528）。
  → 之前"系统性 15~18% 引擎差距"对这两个动作**主要由 MJCF 质量 bug 解释，而不是求解器差异**。
- **goal 修正后仍有明显 gap**（0.0952 vs 0.0791，up error 12.0° vs 4.56°）→ goal 存在一个**真实的、动作特有**的剩余差异，需单独追。

## 分叉定位

goal T4 双引擎原动作段：

| 时间段 | MuJoCo MAE | Isaac MAE | MuJoCo up error | Isaac up error |
|---|---:|---:|---:|---:|
| 0–2s | 0.0803 | 0.0591 | 9.01° | 3.13° |
| 2–4s | 0.1106 | 0.0993 | 11.08° | 4.56° |
| 4–4.88s | 0.1000 | 0.0780 | 3.25° | 3.51° |

最大 per-joint gap：

| 关节 | MuJoCo MAE | Isaac MAE | 差值 |
|---|---:|---:|---:|
| l_thigh | 0.1249 | 0.0534 | +0.0716 |
| l_ankle_roll | 0.1465 | 0.1106 | +0.0359 |
| r_calf | 0.1790 | 0.1445 | +0.0345 |
| r_thigh | 0.0787 | 0.0486 | +0.0301 |
| l_hip_roll | 0.0943 | 0.0654 | +0.0289 |

## 力矩与接触量级抽查

这不是严格同状态辨识，Isaac 项为 `computed_torque_estimate`，不能当作 PhysX 实测 applied torque。

| 引擎 | torque mean | p95 | max |
|---|---:|---:|---:|
| MuJoCo effective_tau | 2.027 | 9.286 | 29.314 |
| Isaac computed_torque_estimate | 1.263 | 5.203 | 17.875 |

接触统计：

- MuJoCo 平均接触数：3.54，零接触采样比例 8.20%，接触力范数峰值 2168.5N；
- Isaac body net force 范数峰值 367.0N，body contact-force >1N 的比例约 6.81%；
- 由于两边统计对象和接触定义不同，这些只能说明接触/力矩量级不同，不能直接比较为同一物理量。

## 本轮 A1-A8 隔离结果

| 变体 | 原动作 MAE | up error max | 结果 |
|---|---:|---:|---|
| A1 baseline | 0.096409 | 11.078° | 基线 |
| A2 position actuator dt=.002 | 0.096409 | 11.078° | 与 A1 同构，轨迹差≤1e-6，不构成独立对照 |
| A3 position actuator dt=.005 | 0.097386 | 20.057° | 变差 |
| A4 qvel clamp | 0.095630 | 11.963° | 影响小 |
| A5 PD 每物理步更新 | 0.305660 | 82.967° | 改变 reference 时序，协议错误 |
| A6 damping=0 | 0.098235 | 10.881° | 小影响 |
| A6 frictionloss=0 | 0.096409 | 11.078° | 原值就是0，无效 |
| A7 implicit integrator | 0.097656 | 10.783° | 小影响 |
| A8 implicitfast | 0.096409 | 11.078° | 与基线近似一致 |

## 质量自评

本轮最有价值的是生效参数快照和无效实验识别；A2 最初被误写成“position actuator 没影响”，复核代码后确认它和基线都是每个物理步用相同 kp/kd 计算 PD，属于同构控制器。已修正报告，不把 A2 当作隐式 PhysX 求解的排除证据。

仍未完成严格同状态力矩 A/B；这是下一轮首要实验。静态质量/几何结果也需要做系统 COM/总惯量聚合，不能把 base_link 合并差异直接归因于动力学。

## 下一步

1. 同状态同 target 的 Isaac applied/computed torque 与 MuJoCo effective_tau A/B；
2. URDF/MJCF/Isaac 的系统总质量、COM、惯量聚合对齐；
3. 对 l_thigh、l_ankle_roll、r_calf 做 collision geom 作用点和接触时刻对照；
4. 最后再决定是否需要构造更接近 PhysX 的 MuJoCo 执行器/接触模型。
