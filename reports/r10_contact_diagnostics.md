# 第 10 轮：goal 剩余 gap 的动力学定位

## 1. 结论摘要

本轮完成了真正的双引擎同状态动力学 step 对照。与第 9 轮只比较 PD 指令力矩不同，本轮在两个引擎中都写入相同状态，并让两个引擎实际物理 step；两边使用统一 `dt=0.005s`、同一 goal action/target 和同一初始状态。

结果：

- 单步后，关节位置差为 `0.00178~0.00275 rad`（5 个时刻最大值）；速度差最大 `0.663 rad/s`。
- 同一 action 的第一步实际 torque 最大差为 `7.77e-6 N·m`，说明输入 torque 对齐；但 step 后状态已经出现小幅动力学分叉。
- 固定 action、闭环 PD 每步重算，分叉随步数增长：50 步（0.25s）全关节最大位置差 `0.0247~0.0509 rad`，最大值出现在 `t=3.5s` 的 `l_thigh_joint`。
- 分叉最大的关节随时刻不同，集中在脚/腿和部分上肢小惯量关节；不能仅凭本轮数据把差异归因到某一个 contact 参数。
- 接触对象级严格对照仍无法完成：MuJoCo 有逐 geom/contact point，Isaac 当前 ContactSensor 只有 body-level net force，`force_matrix_w/pos_w` 为空。

因此，当前最可靠的结论是：

> **goal 的剩余 gap 确实包含真实的跨引擎动力学分叉；第 10 轮证明了“相同输入 torque 经过一步物理演化后状态就不完全相同”。但本轮还不能把分叉唯一归因于接触求解器、碰撞几何、积分器或质量/惯量中的某一项。**

## 2. 实验协议与 proof-of-effect

### 2.1 输入状态

取 `/data/gaoli/logs/model_align_r2/isaac_goal_contact/T4_goal_hold.npz` 中的：

```text
t = 2.0 / 2.5 / 3.0 / 3.5 / 4.0 s
root_pos, root_quat, root_lin_vel, base_omega_body
joint_pos[22], joint_vel[22], action[22]
```

两边每个 horizon 独立写入同一个状态；状态写入后 q/qvel 最大误差为 `0`（Isaac 侧逐样本检查）。状态写入仅是 snapshot 实验设置，不计为连续轨迹 reset。

### 2.2 实际参数

| 项目 | Isaac | MuJoCo |
|---|---:|---:|
| physics dt | 0.005s | 0.005s |
| horizon | 1/5/20/50 steps | 1/5/20/50 steps |
| 控制方式 | 固定 action，每步 implicit PD 重算 | 固定 target，每步显式 PD 重算 |
| armature | Isaac config 实际读回 | 复用评估器注入：腿 0.044277，臂 0.008419 |
| damping | Isaac config 实际读回 | 复用评估器 fallback：低于 0.01 的 DOF 设为 0.05 |
| effort limit | 腿 30N·m，臂 20N·m | 同样限幅 |
| 实际 step 数 | horizon | horizon |

MuJoCo 多步没有固定第一个 torque；每一步根据各自当前 q/dq 重新计算闭环 PD。这一点与 Isaac 的固定 action 协议一致，避免把 open-loop 固定 torque 与 closed-loop implicit actuator 混在一起。

### 2.3 torque proof-of-effect

同一时刻第一步的两边实际 torque 最大差：

```text
t=2.0s: 1.84e-6 N·m

这仍然只是输入 torque 对齐证据；动力学结论使用后续 q/dq 差异，不使用 PD 公式一致性本身。

## 3. 阶段 A：修正模型动作表

阶段 A 沿用已确认的第 8/9 轮数字，不重复跑训练和长轨迹：

| 动作 | 引擎 | 基线 A | 阈值 A×1.10 | 收尾 MAE | 收尾稳定性 | 判定 |
|---|---|---:|---:|---:|---|---|
| kungfu | MuJoCo 修正 | 0.117300 | 0.129030 | N/A | 基线未倒 | 无收尾 checkpoint |
| kungfu | Isaac | 0.113527 | 0.124880 | N/A | 基线未倒 | 无收尾 checkpoint |
| dance | MuJoCo 修正 | 0.121471 | 0.133618 | N/A | 基线未倒 | 无收尾 checkpoint |
| dance | Isaac | N/A | N/A | N/A | N/A | 既有产物未覆盖 |
| bigkick | MuJoCo 修正 | 0.055356 | 0.060892 | 0.052340 | 收尾保持 3.54s | 通过 |
| bigkick | Isaac | 0.056275 | 0.061902 | 0.052793 | 收尾稳定 | 通过 |
| goal | MuJoCo 修正 | 0.089184 | 0.098103 | 0.095198 | 收尾保持 4.42s | 通过 |
| goal | Isaac | 0.077650 | 0.085415 | 0.079117 | 收尾稳定 | 通过 |
| kick1015 | MuJoCo 修正 | 0.051073 | 0.056180 | 0.053173 | 收尾保持 3.54s | 通过 |
| kick1015 | Isaac | 0.048528 | 0.053381 | 0.052754 | 收尾稳定 | 通过 |

## 4. 阶段 B：接触能力边界

### MuJoCo

第 9 轮已经实际记录：

- 2~4s 主要接触 pair：`floor ↔ l_ankle_roll_link_collision`、`floor ↔ r_ankle_roll_link_collision`；
- 有逐 contact geom、contact point、contact frame 和 `mj_contactForce` 法向分量。

### Isaac

第 9 轮实际读回：

```text
net_forces_w: [1, 23, 3]
force_matrix_w: None
pos_w: None
```

因此当前 Isaac API 只能可靠给出 body-level 接触力和左右脚接触时序，不能给出逐 contact pair / contact point。按照本轮规则，Isaac 与 MuJoCo 的接触法向力不能做严格定量一一对应；不能把 Isaac net force 范数替代 MuJoCo contact normal force。

已知的脚级线索仍可保留：左右 ankle-roll 是共同的主要接触 body，但接触时刻存在差异。它是后续归因候选，不是本轮已证因果。

## 5. 阶段 C：真实 step 分叉结果

以下是全 22 个关节的 `q_after_isaac - q_after_mujoco`。`q_before` 两边均等于相同输入状态。

| 时刻 | horizon | max |Δq| (rad) | RMS |Δq| | max |Δdq| (rad/s) | 最大差异关节 |
|---:|---:|---:|---:|---:|---|
| 2.0s | 1 | 0.002497 | 0.000767 | 0.663 | l_ankle_pitch |
| 2.0s | 5 | 0.005616 | 0.002028 | 0.212 | l_ankle_pitch |
| 2.0s | 20 | 0.015658 | 0.005458 | 0.579 | r_thigh |
| 2.0s | 50 | 0.027030 | 0.009224 | 0.291 | l_ankle_pitch |
| 2.5s | 1 | 0.002114 | 0.000600 | 0.303 | l_ankle_roll |
| 2.5s | 5 | 0.009211 | 0.002343 | 0.190 | l_ankle_roll |
| 2.5s | 20 | 0.010612 | 0.003070 | 0.331 | l_ankle_roll |
| 2.5s | 50 | 0.024674 | 0.008157 | 0.916 | r_hip_roll |
| 3.0s | 1 | 0.002248 | 0.000762 | 0.509 | r_ankle_roll |
| 3.0s | 5 | 0.015264 | 0.004602 | 0.472 | r_ankle_roll |
| 3.0s | 20 | 0.015648 | 0.006283 | 0.515 | r_thigh |
| 3.0s | 50 | 0.039234 | 0.011166 | 0.713 | l_thigh |
| 3.5s | 1 | 0.002746 | 0.000902 | 0.330 | l_thigh |
| 3.5s | 5 | 0.014850 | 0.004699 | 0.317 | l_thigh |
| 3.5s | 20 | 0.008345 | 0.003122 | 0.245 | l_thigh |
| 3.5s | 50 | **0.050871** | **0.015184** | 1.438 | l_thigh |
| 4.0s | 1 | 0.001777 | 0.000805 | 0.216 | r_elbow |
| 4.0s | 5 | 0.011539 | 0.004850 | 0.244 | l_shoulder_pitch |
| 4.0s | 20 | 0.047227 | 0.014124 | 0.537 | l_thigh |
| 4.0s | 50 | 0.037119 | 0.014819 | 0.614 | r_hip_pitch |

关键观察：

1. **一步已经分叉**：最大单步位置差约 `0.00275 rad`，不是等到多步后才出现。
2. **多步继续放大**：50 步最大位置差达到 `0.05087 rad`，但不同初始时刻不单调，说明接触/约束和闭环控制共同影响分叉。
3. 分叉不是只集中在一个关节：短时主要出现在 ankle/thigh/hip；3.5s 的 50 步最大项是 `l_thigh`。
4. 本轮没有完成严格接触 pair 对齐，因此不能说某一次分叉一定由某个脚接触造成。

## 6. 证据等级与结论

| 结论 | 状态 | 证据等级 |
|---|---|---|
| 第 9 轮 torque 数值一致 | 已确认 | 非动力学：输入/控制参数对齐 |
| 同状态、同 dt、同 action 下下一步状态不同 | **已确认** | **动力学：两边真实 step 后 q/dq 差异** |
| gap 不是 actuator target/scale 映射错误 | 已确认 | 非动力学 + 第 10 轮第一步 torque proof |
| gap 由接触求解器单独造成 | 未确认 | 缺 Isaac contact pair/point |
| gap 由碰撞几何单独造成 | 未确认 | 尚未做单项 geometry A/B |
| gap 由质量/惯量/COM 单独造成 | 未确认 | 尚未完成单项动力学隔离 |
| gap 具有真实跨引擎动力学成分 | **已确认** | **动力学 step 分叉证据** |

本轮最谨慎的最终结论：

> **goal 的剩余 gap 不是只有评估口径或 PD 公式问题；在同状态、同 dt、同第一步 torque 下，两个引擎真实演化已经出现可测分叉。分叉随后在闭环多步中增长，但目前只能确认“引擎动力学/接触/模型组合存在差异”，不能把责任唯一归给接触求解器或某个碰撞 geom。**

## 7. 下一步

1. 先完成 Isaac 侧 PhysX contact report / simulation view 的逐 pair 和 contact point 读取；否则接触归因无法闭环。
2. 在同一 step 分叉 harness 中增加单项 A/B：
   - 关闭/简化脚底 contact；
   - 替换 ankle-roll collision mesh 为等包围盒；
   - 对齐 COM/inertia；
   - 统一积分器/dt。
3. 每个变体都以 `max|Δq|`（1/5/20/50 steps）作为相同指标，禁止只看最终轨迹 MAE。

## 产物

- `/data/gaoli/logs/model_align_r3/mujoco_goal_dynamics_step.py`
- `/data/gaoli/logs/model_align_r3/isaac_goal_dynamics_step.py`
- `/data/gaoli/logs/model_align_r3/mujoco_dynamics_step.json`
- `/data/gaoli/logs/model_align_r3/isaac_dynamics_step.json`
- `/data/gaoli/logs/model_align_r3/mujoco_step.log`
- `/data/gaoli/logs/model_align_r3/isaac_step2.log`
