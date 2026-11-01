# 第 17 轮：v2 尝试结果

## 结论摘要

本轮生成并运行了 `isaac_closedloop_v2.py` / `mujoco_closedloop_v2.py`，但 Isaac v2 实际上仍然继承了旧闭环脚本的状态读取问题；它不是从最小 harness 完整重建的独立实现。因此 0b gate 未通过，闭环分叉数字不能作为答案 1 的科学结论。

本轮没有完成两个要求的科学答案：

```text
答案 1：0.05087 是否被固定 action 放大？仍无法判定
答案 2：分叉是否接触事件驱动？未执行
```

这轮产物保留为【未验证】数据，不得与第 14 轮 A0 的固定 action 结果混用。

## 1. v2 运行状态

生成文件：

```text
/data/gaoli/logs/model_align_r10/isaac_closedloop_v2.py
/data/gaoli/logs/model_align_r10/mujoco_closedloop_v2.py
```

两份脚本均通过 `py_compile` 并实际运行。

但 Isaac v2 是从旧闭环脚本复制而来，没有真正从最小 harness 逐步重建，因此不能视为合格的新 harness。

Isaac root gate 仍为：

```text
2.0s: 0.1221957505
2.5s: 0.1666548848
3.0s: 0.2637487054
3.5s: 0.1135706902
4.0s: 0.0106801987
```

因为 step 前状态 gate 未通过，本轮所有闭环分叉数字均标记为【未验证】。

## 2. 初步闭环轨迹差异（仅供参考）

MuJoCo/Isaac 轨迹 q 差的初步结果如下，但不作为答案 1：

| 时刻 | 1 step | 5 steps | 20 steps | 50 steps |
|---:|---:|---:|---:|---:|
| 2.0s | 0.002511 | 0.010512 | 0.033291 | 0.043738 |
| 2.5s | 0.001929 | 0.007588 | 0.024960 | 0.075184 |
| 3.0s | 0.002510 | 0.019616 | 0.108078 | 0.128542 |
| 3.5s | 0.005479 | 0.031543 | 0.030060 | 0.081653 |
| 4.0s | 0.001380 | 0.022746 | 0.028099 | 0.034254 |

这些数字受旧 root gate 影响，不可用来判断闭环是否缩小了固定 action 的分叉。

## 3. 固定 action 基线

第 14 轮 A0 的固定 action 结果仍是当前有效基线：

```text
3.5s / 50 steps / max|Δq| = 0.050871 rad
```

该数字是【动力学对照】且已逐点复现。

## 4. 接触对齐

本轮没有执行逐物理步接触-分叉对齐，因为 Isaac v2 的状态 gate 未通过。

第 11 轮 contact API 结论继续沿用：

```text
Isaac body-level net force 可用；逐 contact pair/point 当前不可比。
```

## 5. 本轮结论表

| 项目 | 状态 | 证据等级 |
|---|---|---|
| v2 脚本可启动 | 已确认 | 工程运行 |
| v2 已从最小 harness 正确重建 | 未确认 | 脚本仍继承旧闭环结构 |
| 5 时刻 0b gate | 未通过 | root error 0.01~0.26 |
| 闭环是否缩小 fixed-action 分叉 | 仍无法判定 | 闭环结果未验证 |
| 分叉是否接触驱动 | 未执行 | 阶段 B 未完成 |
| 第 10/14 轮 fixed-action 分叉 | 已确认 | 动力学对照 |

## 6. 下一步

本轮暴露出 v2 不能只复制旧脚本。下一轮必须真正从 `/data/gaoli/logs/model_align_r7/isaac_closedloop_minimal_diff.py` 重写：

1. 先只保留最小写入和 5 时刻 0b；
2. 确认全部 root/joint 回读 <1e-6；
3. 再逐步加入 ONNX、motion、action manager、horizon；
4. 只有 gate 通过后才计算闭环 vs fixed action；
5. 最后做接触对齐。

## 产物

- `/data/gaoli/logs/model_align_r10/isaac_closedloop_v2.py`
- `/data/gaoli/logs/model_align_r10/mujoco_closedloop_v2.py`
- `/data/gaoli/logs/model_align_r10/isaac_run/isaac_closedloop.json`
- `/data/gaoli/logs/model_align_r10/mujoco_closedloop.json`
- `/data/gaoli/logs/model_align_r10/report.md`
