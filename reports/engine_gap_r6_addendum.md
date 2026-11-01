# goal 执行器模型定位（第 6 轮）— 修正版

## 实验对象

goal T4 policy、同一 `pi_plus_goal_tail_v3.npz`、同一 `pi_auto.xml`，MuJoCo CPU 单初始状态零复位评估。
Isaac 对照固定为：原段 MAE 0.079117、基座 up-direction max 4.561°。

## 重要更正：A2 不是有效的执行器对照

第 6 轮初版把 A2（MuJoCo position actuator）写成“没有产生可见动力学差异”。核对代码后发现：

- 基线 `auto_sim2sim.py` 的控制循环是**物理步循环**，每个物理步都用当前 `d.qpos/d.qvel` 重算 PD
  （`auto_sim2sim.py:417-423`），即基线本来就是**物理步级（500Hz）闭环 PD**，不是“50Hz 显式 + 20ms 保持”。
- 因此把 motor 换成 MuJoCo position actuator、且 kp/kv 相同，**在数学上就是同一个控制器**。

用保存轨迹验证（A1 vs A2）：

| 数组 | 最大绝对差 |
|---|---:|
| joint_pos | 1.32e-7 |
| joint_vel | 3.83e-6 |
| effective_tau | 9.61e-6 |
| base_quat | 3.63e-8 |

**结论：A2 与 A1 是同一控制器（浮点级差异），A2 不构成“position vs motor”的对照实验，不能据此说“测试过并排除”。**

## 生效检查

每个变体保存了实际运行时的 timestep、control decimation、actuator gain/bias、damping、frictionloss、armature 和限幅。

- A1：dt 0.002、decimation 10、motor gain=1、bias=0、damping=0.05、frictionloss=0、armature 已注入。
- A2/A3：加载了 22 个 position actuator，kp/kv 生效。
- A4：qvel clamp 生效。
- A6 damping=0：读回 0.0，真实生效。
- A6 frictionloss=0：模型本来就是 0，**无效因素**。

## 结果

| 变体 | dt/控制 | 首次倒地 | 原段 MAE | up error max | 结论 |
|---|---|---:|---:|---:|---|
| A1 baseline | .002 / 50Hz | 未倒 | 0.096409 | 11.078° | 基线 |
| A2 position actuator | .002 / 50Hz | 未倒 | 0.096409 | 11.078° | **与 A1 同构（差 ≤1e-6），无效对照** |
| A3 position actuator | .005 / 50Hz | 5.96s | 0.097386 | 20.057° | 步长对齐不行（≈第5轮 dt=.005 结果） |
| A4 qvel clamp 8/17 | .002 / 50Hz | 未倒 | 0.095630 | 11.963° | 影响很小，且姿态略差 |
| A5 PD 每物理步更新 | .002 / 500Hz | 1.00s | 0.305660 | 82.967° | 破坏 reference 时序，协议错误 |
| A6 damping=0 | .002 / 50Hz | 未倒 | 0.098235 | 10.881° | 小幅影响，非主因 |
| A6 frictionloss=0 | .002 / 50Hz | 未倒 | 0.096409 | 11.078° | 无效因素 |
| A7 integrator=implicit | .002 / 50Hz | 未倒 | 0.097656 | 10.783° | 影响很小 |
| A8 integrator=implicitfast | .002 / 50Hz | 未倒 | 0.096409 | 11.078° | 与基线几乎一致 |

## 结论

1. 第 5 轮的 damping 无效实验已修正；damping=0 真实生效但只带来极小变化，不是主因。
2. frictionloss 在自动 MJCF 里结构上不存在，属于无效因素，排除。
3. **A2 不是有效的执行器对照**（与基线同构）；执行器“隐式 vs 显式”这条主线并未被真正测试。
4. 把 dt 对齐到 Isaac（A3）反而变差；MuJoCo 自身积分器改成 implicit/implicitfast（A7/A8）也几乎没有变化。
5. 简单 qvel clamp（A4）不是 PhysX 速度约束的等价实现，影响很小。
6. 因此，**goal 的 gap 更可能来自 PhysX 求解器内部的约束/接触解算、以及质量/惯量/质心/腿部碰撞几何**，而不是 MuJoCo 侧的 dt、积分器、actuator 类型或被动阻尼这些简单旋钮。

## 质量自评（第 6 轮）

| 维度 | 评价 |
|---|---|
| 生效证明机制 | 好（修复了第 5 轮“无效变体被当结论”的问题） |
| 负结果有效性 | 较高（dt/阻尼/积分器/速度限幅/qvel clamp 均非主因） |
| 执行器对照有效性 | **低**（A2 同构，未测到核心；已在本次更正） |
| 单因子隔离 | 中（A3/A5 同时改了协议） |
| 覆盖度 | 低（未做同状态力矩 A/B、未查质量/几何、未在 Isaac 侧对照） |
| 过程卫生 | 中（多次重跑，日志 version 混杂） |

净评价：**一轮“可信的负结果 + 一个无效的执行器对照（已更正）”。**

## 下一步建议（按优先级）

1. **同状态同 target 的力矩 A/B**：固定相同关节角/速度/target，比较 Isaac `applied_torque` 与 MuJoCo `effective_tau`。这是唯一能真正区分执行器/求解差异的实验。
2. **对齐腿部动力学**：l_thigh / l_ankle_roll / r_calf 的质量、惯量、质心、关节轴、足底碰撞几何逐项对齐。
3. **Isaac 侧反向对照**：在 Isaac 里关掉速度限制/换近似显式 PD，看差异是否来自 Isaac 侧。
4. 至少在一个通过动作（kick1015）上复核这些负结果，避免只依赖 goal。

## 产物

服务器：`/data/gaoli/logs/engine_gap_round6/`（`results.json`、`addendum.json`、`variants*.log`、本报告）
