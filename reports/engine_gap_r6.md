# goal 引擎差距定位（第 5 轮）

## 结论先行

本轮没有发现一个简单物理开关可以解释全部 gap。已确认：

1. goal 的 MuJoCo/Isaac 差距主要在动作前半段和腿部/基座姿态，不是只出现在收尾保持。
2. 当前 MuJoCo 的 `dt=0.002` 是稳定性更好的配置；强行改成 Isaac 的 `dt=0.005` 反而在 6.0s 倒地，并没有缩小差距。
3. 用目标速度限制器会明显恶化 tracking，不能把它当作 PhysX 速度限制的等价实现。
4. 将被动阻尼或 `frictionloss` 置零，本次 goal 结果不变，贡献可排除在本模型/协议下。
5. 后续真正值得做的是执行器模型（显式 PD vs MuJoCo position/implicit actuator）和质量/质心/接触几何的精确同状态实验；本轮没有把它们的贡献冒充成已测结论。

## 阶段一：原始 goal 轨迹分叉

使用 T4 goal 策略、同一 tail reference、零复位 12s 轨迹：

| 指标 | MuJoCo | Isaac | 差值 |
|---|---:|---:|---:|
| 原动作段 MAE | 0.096409 | 0.079117 | +0.017292 |
| 原动作段最大 up-direction error | 11.078° | 4.561° | +6.517° |
| 原动作段 arms MAE | 0.062493 | 0.051952 | +0.010541 |
| 原动作段 legs MAE | 0.124672 | 0.101754 | +0.022918 |
| 原动作段最大关节速度 | 11.79 | 10.35 | — |

按 2 秒窗口观察：

| 时间段 | MuJoCo MAE | Isaac MAE | MuJoCo up error max | Isaac up error max |
|---|---:|---:|---:|---:|
| 0–2s | 0.080341 | 0.059057 | 9.01° | 3.13° |
| 2–4s | 0.110615 | 0.099336 | 11.08° | 4.56° |
| 4–4.88s | 0.099959 | 0.077974 | 3.25° | 3.51° |

最大 per-joint gap：

| 关节 | MuJoCo MAE | Isaac MAE | 差值 |
|---|---:|---:|---:|
| l_thigh | 0.124918 | 0.053362 | +0.071555 |
| l_ankle_roll | 0.146492 | 0.110609 | +0.035883 |
| r_calf | 0.178999 | 0.144472 | +0.034526 |
| r_thigh | 0.078675 | 0.048557 | +0.030118 |
| l_hip_roll | 0.094308 | 0.065441 | +0.028867 |

**分叉集中在 0–4s，尤其是 2–4s 的腿部和基座方向。**不是仅仅由动作结束后的保持姿势造成。

## 阶段二：MuJoCo 单项隔离

目标：同一个 goal T4 policy、同一个 tail reference、同一个初始状态；只改变一项。

| 变体 | 首次倒地 | 原段 MAE | 原段 up error max | 结论 |
|---|---:|---:|---:|---|
| baseline（dt=.002，当前 PD） | 未倒（8s） | 0.096409 | 11.078° | 基线 |
| dt=.005 / decimation=4 | 6.00s | 0.097386 | 20.057° | **变差**，不能简单把步长对齐到 Isaac |
| 目标速度限制 8/17 rad/s | 3.60s | 0.161249 | 50.916° | **明显变差**；目标速率限幅不是 PhysX 速度限制的等价实现 |
| 被动阻尼 ×0 | 未倒（8s） | 0.096409 | 11.078° | 本实验无可见影响 |
| frictionloss ×0 | 未倒（8s） | 0.096409 | 11.078° | 本实验无可见影响 |

速度限制变体应谨慎解释：本实验使用的是目标位置 rate limiter，并不等价于 PhysX 内的执行器/关节速度限制；它只证明这种简单实现不能用于对齐。

## 阶段三：力矩抽查

以下不是同一时刻、同一状态的严格系统辨识，只是保存轨迹中的量级抽查；Isaac 项是 `computed_torque_estimate`，不是 PhysX 实测有效力矩。

| 引擎 | 来源 | mean | p95 | max |
|---|---|---:|---:|---:|
| MuJoCo | effective_tau | 2.027 | 9.286 | 29.314 |
| Isaac | computed_torque_estimate | 1.263 | 5.203 | 17.875 |

这个差异提示执行器/状态轨迹仍不等价，但不能单凭该表确定具体因果。

## 当前判断

- 已排除/弱化：简单步长替换、简单目标速度限制、被动 damping、frictionloss。
- 已定位：goal gap 主要体现为 0–4s 腿部运动和基座方向误差；MuJoCo 的 l_thigh 是最大单关节 gap。
- 尚未闭环：MuJoCo 显式 PD 与 PhysX 隐式 PD 的差异、质量/质心、脚底碰撞几何和接触作用点。

## 下一步建议

1. **首要实验：执行器模型对齐**。在 MuJoCo 副本里将 Python 显式 PD 与 MuJoCo position/implicit actuator 做严格 A/B，保持同一 target、state、dt、limits；记录 effective torque、joint error、base orientation。
2. **第二步：质量/质心/惯量对齐**。按 URDF/Isaac 有效参数逐刚体对齐，不要只比较总质量。
3. **第三步：接触几何与作用点对齐**。只比较 goal 的 2–4s 发生的脚底接触，输出 contact geom、位置、法向和接触力。
4. goal 的静态末帧姿态偏差约 0.11 rad 另立任务处理，不和引擎 gap 混合。

## 产物

服务器：`/data/gaoli/logs/engine_gap/`

- `divergence_analysis.json`：分叉与 per-joint gap
- `divergence_timeline.png`：时间线
- `variants.json`：单项 MuJoCo 隔离结果
- `per_joint_phase.json`：原动作段 per-joint 对照
- `torque_summary.json`：力矩量级抽查
- `analysis.log` / `variants.log`：运行日志
