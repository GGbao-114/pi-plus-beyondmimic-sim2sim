# 第 15 轮：旧闭环复现与固定 action 复验

## 结论摘要

本轮完成了两项有效复验：

1. 第 13 轮旧闭环脚本的 root error `0.1222/0.1667/0.2637/0.1136/0.0107` 可以原样复现；
2. 第 10 轮固定 action 的动力学分叉已经在第 14 轮 A0 逐点复现。

但是，本轮仍没有完成旧闭环 0.12 的根因定位。一个与旧闭环同配置的最小化脚本在 step 前读回 root 误差为 0，而旧闭环在同样声称的 step 前位置报告 0.12~0.26。这个矛盾已经被稳定复现，但还没有通过 readback/状态写入段换件完全解释。

因此，本轮不能把闭环 policy 分叉或接触归因写成科学结论。

## 1. 阶段 0：旧闭环 0.12 复现

原样运行：

```text
/data/gaoli/logs/model_align_r6/isaac_goal_closedloop.py
```

输出 root error：

```text
2.0s: 0.1221957505
2.5s: 0.1666548848
3.0s: 0.2637487054
3.5s: 0.1135706902
4.0s: 0.0106801987
```

因此第 13/14 轮的 `0.12~0.26` 不是偶然噪声，本轮可以复现。

2.0s 的旧闭环回读字段：

```text
root_pos 与目标一致
root_quat 与目标一致
root_lin_vel 与目标一致
root_ang_vel_b = [0.11815573, 0.82201302, 0.07025026]
目标 body omega = [0.17061673, 0.82717586, -0.05194549]
```

差异主要来自 root angular velocity。

## 2. 最小化脚本对照

使用最小化差分脚本：

```text
/data/gaoli/logs/model_align_r7/isaac_closedloop_minimal_diff.py
```

逐步加入：

```text
write_root
write_joint
scene.write_data_to_sim
sim.forward
scene.update
action_manager.reset
ONNX session
motion load
process/apply action
再次 scene.write/forward/update
sim.step
```

结果：

- 所有 `sim.step()` 之前的阶段：
  ```text
  err_b = 0
  err_w ≈ 8.08e-9
  ```
- 第一个超过阈值的阶段是：
  ```text
  add_i_one_step
  ```
- step 后的变化：
  ```text
  err_b ≈ 0.0220 rad/s
  err_w ≈ 0.0186 rad/s
  ```

这说明基础流程和最小化闭环流程没有在 step 前破坏 root，但它没有解释旧闭环为何在 step 前报告 0.12~0.26。

### 当前未解释的矛盾

```text
旧闭环：step 前 root error = 0.12~0.26
最小化脚本：step 前 root error = 0
```

已知 frame、输入状态、基础写入调用均已单独验证，但旧脚本与最小脚本仍存在未定位的运行差异。旧闭环的 readback 函数/状态写入段还没有完成 a/b/c 三种换件实验，因此不能宣布根因。

阶段 0 判定：

```text
旧值可复现：是
根因已定位：否
旧闭环可用于动力学归因：否
```

## 3. 阶段 0b

本轮没有完成正式的 5 时刻 state helper 误差表。当前只完成了：

- t=2.0s 的基础二分；
- 旧闭环 5 个时刻的异常复现。

所以阶段 0b 仍未通过，不进入有效闭环阶段。

## 4. A0：第 10 轮固定 action 复验

第 14 轮已经完成，本轮沿用结果：

| 时刻 | 1 step | 5 steps | 20 steps | 50 steps |
|---:|---:|---:|---:|---:|
| 2.0s | 0.002497 | 0.005616 | 0.015658 | 0.027030 |
| 2.5s | 0.002114 | 0.009211 | 0.010612 | 0.024674 |
| 3.0s | 0.002248 | 0.015264 | 0.015648 | 0.039234 |
| 3.5s | 0.002746 | 0.014850 | 0.008345 | **0.050871** |
| 4.0s | 0.001777 | 0.011539 | 0.047227 | 0.037119 |

结论：第 10 轮固定 action 动力学分叉可复现。这是【动力学对照】证据。

## 5. 阶段 A/B/D

- 闭环 policy vs fixed action：未完成；
- 逐物理步分叉-接触对齐：未完成；
- 阻尼/积分器/接触/几何隔离：未完成。

原因：阶段 0b 尚未通过，且旧闭环 0.12 的换件根因尚未定位。

## 6. 当前结论表

| 项目 | 状态 | 证据等级 |
|---|---|---|
| root frame 是 R@b | 已确认 | 状态写入实验 |
| 基础 write/forward/update 链破坏 root | 未发现 | 最小二分 |
| 旧闭环 0.12 可复现 | 已确认 | 旧脚本原样复跑 |
| 旧闭环 0.12 的根因 | 未确认 | 换件定位未完成 |
| 第 10 轮 fixed-action 分叉可信 | 已确认 | 动力学对照 |
| 0.05 是否被固定 action 放大 | 仍无法判定 | 闭环尚未有效 |
| 分叉是否接触驱动 | 仍无法判定 | 阶段 B 未完成 |

## 7. 下一步

必须先完成旧闭环换件定位，不得继续跑长实验：

```text
a) 仅替换旧 readback 函数为 minimal_repro 的 readback
b) 仅替换旧状态写入段为 minimal_repro 的写入段
c) 同时替换 a+b
```

每个变体只跑 t=2.0s、horizon=1，并打印：

```text
实际写入 root angular velocity
read_root_ang_vel_b/w
root error 分解
```

定位完成后再补 0b 五时刻表，然后才允许闭环 vs fixed-action。

## 产物

- `/data/gaoli/logs/model_align_r8/old_run/isaac_closedloop.json`
- `/data/gaoli/logs/model_align_r8/old_run.log`
- `/data/gaoli/logs/model_align_r8/variant/isaac_closedloop.json`
- `/data/gaoli/logs/model_align_r8/min2/minimal_repro.json`
- `/data/gaoli/logs/model_align_r8/report.md`
