# 第 11 轮：goal 分叉归因与接触对齐

## 结论摘要

本轮没有通过阶段 A 的 root 状态协议，因此**不把本轮闭环分叉数字当作有效动力学结论**。第 10 轮的固定 action 动力学 step 结果仍可作为历史数据，但不能用本轮未对齐 root 的闭环结果解释全轨迹 gap。

本轮确认了一个新的关键协议问题：保存的 `base_omega` 与当前 Isaac `write_root_state_to_sim` 的 root angular velocity frame/API 语义没有直接对齐。joint q/dq 可以写回，但 root angular velocity 回读误差为 `0.043~0.252 rad/s`，所以闭环 policy action 也不一致，最大初始 action 差约 `0.16`。这会污染闭环分叉实验，必须先解决。

本轮阶段 C 的 Isaac contact API 探针已真实执行，获得了以下有效边界：

- `ContactSensor.data.force_matrix_w` 字段存在但值为 `None`；`net_forces_w` 是 body-level `[1,23,3]`；
- `root_physx_view` 能读到 contact offsets、DOF force、incoming joint force 等，但当前 articulation view 没有逐 contact pair/point 查询 API；
- `omni.physx` 可以导入并暴露 PhysX scene-query/simulation interface 名称，但本轮没有成功取得 contact report pair/point 数据。

因此当前逐 contact pair/point 对照仍为【不可比】，但这次结论有实际 API 探针证据，不再是沿用旧判断。

## 1. 阶段 A：闭环分叉（协议阻塞）

### 1.1 实验设置

- 两引擎拟使用 `dt=0.005s`，每 4 个 physics step 更新一次 policy action；
- t = `2.0/2.5/3.0/3.5/4.0s`；
- horizon = `1/5/20/50` physics steps；
- MuJoCo 使用 ONNX policy + 显式 PD；Isaac 使用 ONNX policy + native implicit actuator；
- root position/quaternion/linear velocity 与 joint q/dq 写入路径均执行；
- 两边都尝试回读 root position/quaternion/linear velocity/angular velocity。

### 1.2 root 回读结果

本轮最终采用直接写入 `root_ang_vel` 的试验仍得到：

| 时刻 | root angular velocity 回读最大误差 |
|---:|---:|
| 2.0s | 0.2523 rad/s |
| 2.5s | 0.1466 rad/s |
| 3.0s | 0.2381 rad/s |
| 3.5s | 0.0427 rad/s |
| 4.0s | 0.0475 rad/s |

关节 q/dq 写入没有报告同等级误差，但 root angular velocity 未对齐，因此状态 S 不相同。

### 1.3 policy action 证据

因为 Isaac 与 MuJoCo 的 root angular velocity 输入/回读不一致，两个引擎的初始 observation 不一致，初始 ONNX action 也不一致。此前闭环尝试的初始 action 最大差约 `0.01~0.16`，不能作为动力学分叉证据。

### 1.4 阶段 A 判定

```text
状态写入：joint q/dq 可用；root angular velocity 未通过
闭环 policy：无效，不能回答固定 action 是否放大分叉
闭环 q/dq 分叉：全部作废，不用于因果结论
```

这一步是【实验协议质量检查】，不是动力学结论。

## 2. 阶段 B：分叉-接触事件对齐

本轮没有在 root 未对齐的闭环实验上做因果接触分析，避免把协议错误映射成接触结论。

可复用的第 10 轮历史数据仍表明：

- MuJoCo 有逐 geom/contact point/contact normal force；
- Isaac 旧 ContactSensor 只有 body-level net force；
- 第 10 轮固定 action 的 `k=1/5/20/50` 只保存 horizon 末端 q/dq，缺少可直接与每个物理 step 对齐的 Δq 序列。

所以本轮阶段 B 判定：**未完成**。下一次必须先修 root，再在同一有效闭环 harness 中逐 physics step 同时记录 Δq、接触集合和接触力。

## 3. 阶段 C：Isaac contact API 真实探针

产物：`/data/gaoli/logs/model_align_r4/contact_api_probe.json`。

### 3.1 ContactSensor

实际对象字段：

```text
sensor.data.net_forces_w       -> [1, 23, 3]
sensor.data.force_matrix_w     -> None
sensor.data.force_matrix_w_history -> None
sensor.data.current_contact_time
sensor.data.last_contact_time
```

实际 body-level `net_forces_w` 可读；逐 contact pair/point 的 force matrix 与 position 当前为 None。

### 3.2 root_physx_view

实际枚举并调用了以下 API：

```text
get_contact_offsets()                       -> Tensor [1,44]
get_coriolis_and_centrifugal_forces()      -> Tensor [1,22]
get_dof_actuation_forces()                 -> Tensor [1,22]
get_dof_max_forces()                       -> Tensor [1,22]
get_dof_projected_joint_forces()           -> Tensor [1,22]
get_generalized_gravity_forces()           -> Tensor [1,22]
get_gravity_compensation_forces()          -> Tensor [1,28]
get_link_incoming_joint_force()            -> Tensor [1,23,6]
```

`get_link_incoming_joint_force()` 是 body/joint incoming wrench，**不是逐地面 contact pair/point**，不能直接替代 MuJoCo `mj_contactForce`。

### 3.3 omni.physx

实际导入成功：

```text
omni.physx import: OK
```

发现的接口名称包括：

```text
acquire_physx_interface
acquire_physx_scene_query_interface
acquire_physx_simulation_interface
get_physx_scene_query_interface
get_physx_simulation_interface
```

本轮尚未成功通过这些 interface 获得 contact report 的逐 pair/point 数据。

### 3.4 阶段 C 判定

真实尝试清单已经完成，但当前 Isaac API 只拿到 body-level force 和 incoming joint wrench，未拿到逐 pair/point。

结论：

```text
Isaac 逐 contact pair/point：当前仍不可用
Isaac body-level contact force：可用
MuJoCo 逐 pair/point：可用
两边逐点法向力定量对照：不可比
```

## 4. 阶段 D：单项隔离

本轮没有执行阻尼/积分器/接触参数/几何/惯量的正式单项隔离，因为阶段 A 的 root 状态协议尚未通过。若继续做这些变体，会把 root frame 错误带入所有结果，违反本轮验收要求。

判定：未完成，原因明确且可复现。

## 5. 本轮结论表

| 项目 | 结论 | 证据等级 |
|---|---|---|
| Isaac ContactSensor 有 body-level net force | 已确认 | API 实读字段/shape |
| Isaac 当前 ContactSensor 有逐 pair force matrix | 未确认，实际为 None | API 探针 |
| root_physx_view 有逐 contact pair 查询 | 未发现 | API 枚举/调用 |
| `omni.physx` 可导入 | 已确认 | API import |
| `omni.physx` 已拿到逐 pair/point | 未完成 | 无有效数据 |
| 第 10 轮固定 action 分叉是否被放大 | 本轮无法回答 | 阶段 A root 阻塞 |
| 本轮闭环分叉是否真实 | 不可用 | root angular velocity 未对齐 |
| goal gap 是否由接触事件驱动 | 仍无法判定 | 阶段 B 未完成 |

## 6. 下一步必须先做的事

1. 先建立最小 Isaac root frame 单元测试：只写一个 quaternion + body angular velocity，分别输入 `b`、`R@b`、`R.T@b`，逐项读回 `root_ang_vel_w/root_ang_vel_b`，确定 `write_root_state_to_sim` 的确切 frame 语义。
2. 用该单元测试修正阶段 A 的 root 写入，要求 position/quaternion/linear velocity/body angular velocity 全部误差小于 `1e-6`。
3. 重新跑闭环 ONNX 分叉，并比较固定 action vs policy closed-loop。
4. 在有效闭环 harness 中逐物理步记录 Δq 与 contact body/pair，才能判断分叉是否接触驱动。
5. 如需逐 pair，继续从 `omni.physx` scene query/simulation interface 入手；若最终失败，保留完整探针日志并明确标记不可比。

## 产物

- `/data/gaoli/logs/model_align_r4/mujoco_goal_closedloop.py`
- `/data/gaoli/logs/model_align_r4/isaac_goal_closedloop.py`
- `/data/gaoli/logs/model_align_r4/mujoco_closedloop.json`（阶段 A 协议未通过，不作结论）
- `/data/gaoli/logs/model_align_r4/isaac_closedloop.json`（阶段 A 协议未通过，不作结论）
- `/data/gaoli/logs/model_align_r4/isaac_contact_api_probe.py`
- `/data/gaoli/logs/model_align_r4/contact_api_probe.json`
- `/data/gaoli/logs/model_align_r4/report.md`
