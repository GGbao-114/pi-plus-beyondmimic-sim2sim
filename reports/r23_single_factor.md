# 第23轮结果

## 阶段0：记录与公式

已在 `model_align_r12/report.md` 追加更正：phantom 根因是旧脚本 quat->matrix 第三行第一列符号错误，错误 `2*(x*z+y*w)`，正确 `2*(x*z-y*w)`。错误矩阵预测的五个 phantom 值与历史观测吻合约 1e-7。

公式审计：旧 Isaac fixed-action、旧 r6、旧 1c 为 DIRTY；1a、1b、phantom_stage 为 CLEAN；MuJoCo fixed/closed scripts 无该旋转函数。

## 阶段1：修正 fixed-action 污染量化

修正 fixed-action 相对历史 DIRTY fixed-action 的 max|q_after 新-旧|：

| time | 1 | 5 | 20 | 50 |
|---:|---:|---:|---:|---:|
| 2.0 | 0.000366 | 0.001253 | 0.000648 | 0.000378 |
| 2.5 | 0.000640 | 0.002912 | 0.003909 | 0.003423 |
| 3.0 | 0.000948 | 0.004554 | 0.003261 | 0.002047 |
| 3.5 | 0.000102 | 0.000312 | 0.000290 | 0.002274 |
| 4.0 | 0.000029 | 0.000126 | 0.000143 | 0.000116 |

修正 fixed-action 的 max|Delta q|（Isaac vs MuJoCo）：

| time | 1 | 5 | 20 | 50 |
|---:|---:|---:|---:|---:|
| 2.0 | 0.002457 | 0.005276 | 0.016306 | 0.027062 |
| 2.5 | 0.002298 | 0.010009 | 0.011829 | 0.024129 |
| 3.0 | 0.002487 | 0.014843 | 0.013859 | 0.038751 |
| 3.5 | 0.002674 | 0.014588 | 0.008228 | **0.048892** |
| 4.0 | 0.001780 | 0.011573 | 0.047369 | 0.037128 |

历史 dirty fixed-action 的 `0.050871` 在修正后变为 `0.048892`，下降 `0.001979 rad`，相对约 `3.9%`。污染存在，但没有改变数量级结论。

## 答案1：fixed action 是否放大

闭环 v3 max|Delta q| 与修正 fixed-action 对比：

| time | horizon | fixed corrected | closed-loop | ratio |
|---:|---:|---:|---:|---:|
| 2.0 | 50 | 0.027062 | 0.104156 | 3.85 |
| 2.5 | 50 | 0.024129 | 0.069824 | 2.89 |
| 3.0 | 50 | 0.038751 | 0.223926 | 5.78 |
| 3.5 | 50 | 0.048892 | 0.101123 | 2.07 |
| 4.0 | 50 | 0.037128 | 0.087321 | 2.35 |

闭环五个时刻的 50-step 分叉均大于 fixed-action；没有出现闭环 ≤0.5× fixed-action。

**答案1：0.05087 没有被 fixed action 放大。** 在当前协议下，fixed-action 反而低估了 policy 闭环下的跨引擎分叉；闭环约为 fixed-action 的 2.1~5.8 倍。这是【动力学对照】结论，使用了修正公式与通过 step 前 gate 的闭环 v3 数据。

## 答案2：是否接触事件驱动

本轮已有 MuJoCo 接触记录和 Isaac 闭环逐步记录，但 MuJoCo 闭环产物没有保存逐物理步 joint q 序列，只有 horizon 末端 q 与逐步 contact 记录，无法严格计算同一 physics step 的 Delta q(k) 与接触状态关联。

**答案2：仍无法判定。** 不能用 horizon 末端数据替代逐步因果统计。

## 产物与 sha256

- `/data/gaoli/logs/model_align_r13/formula_audit.json` — `32a2137fbaac`
- `/data/gaoli/logs/model_align_r13/isaac_fixed_action_v2.py` — `66a1b199ebbf`
- `/data/gaoli/logs/model_align_r13/isaac_closedloop_v3.py` — `63a8c46a7e64`
- `/data/gaoli/logs/model_align_r13/isaac_dynamics_step.json`
- `/data/gaoli/logs/model_align_r13/v3b/isaac_v2_1c.json`
- `/data/gaoli/logs/model_align_r13/mujoco_closedloop.json`

## 下一步

只补答案2：重新运行两边 fixed-action lockstep，并每个 physics step 同时保存 q、dq、接触状态，再计算接触集一致/不一致时的 Delta q 增长。
