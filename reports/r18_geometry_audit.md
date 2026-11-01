# 第 18 轮：v2 分阶段重建状态

## 结论摘要

本轮完成了 v2 的两个前置阶段：

- **1a 状态写入 + 5 时刻 0b gate：通过**；
- **1b ONNX/motion/action manager（不 step）：通过**。

两个 gate 的回读误差均小于 `1e-6`，并且 omega 写入敏感性测试通过，证明新 harness 的状态写入不会复现旧脚本的 omega 异常。

阶段 1c 检查时发现当前生成的脚本缺少完整的 observation/policy 实现，不能直接运行；本轮没有运行伪闭环，也没有生成答案 1/2 的无效数字。

## 1. 阶段 1a：状态写入 + 0b

产物：

```text
/data/gaoli/logs/model_align_r11/isaac_v2_1a.json
```

5 个时刻均通过：

| 时刻 | root/joint 最大回读误差 | omega 敏感性 | gate |
|---:|---:|---:|---|
| 2.0s | 1e-8 量级 | 通过 | ✅ |
| 2.5s | 3e-8 量级 | 通过 | ✅ |
| 3.0s | 1e-7 量级 | 通过 | ✅ |
| 3.5s | 2e-7 量级 | 通过 | ✅ |
| 4.0s | 1e-7 量级 | 通过 | ✅ |

omega 敏感性测试：对同一状态分别写 `R@b` 和 `R@(b+δ)`，δ=`[0.1,-0.1,0.1]`，两次 body angular velocity 回读差与 δ 一致，残差约 `1e-8~1e-7`。

### 1a 结论

```text
新 harness 的状态写入正确；root frame、joint state 和 omega 写入均通过。
```

这是【状态协议验证】，不是动力学归因。

## 2. 阶段 1b：加入 ONNX/motion/action（不 step）

产物：

```text
/data/gaoli/logs/model_align_r11/isaac_v2_1b.json
```

5 个时刻、以下步骤均保持回读误差 `<1e-6`：

```text
状态写入
ONNX session 创建
session.run 一次
action_manager.process_action
action_manager.apply_action
scene.write_data_to_sim + forward + update
```

### 1b 结论

```text
ONNX、motion、action manager 在不执行 physics step 时不会破坏状态。
```

## 3. 阶段 1c：暂停原因

检查发现当前生成的 `isaac_v2_1c.py` 还没有完整复用 1b 的 observation/policy 实现，不能保证 policy action 正确；继续运行会产生伪闭环数据。

因此本轮没有执行：

- v2 闭环 horizon=1/5/20/50；
- MuJoCo v2 对照；
- 答案 1：闭环 vs fixed action；
- 答案 2：分叉-接触逐物理步对齐。

旧脚本继续标记：

```text
deprecated；根因未定位；不得作为实验基底
```

## 4. 当前有效结论

| 项目 | 状态 | 证据 |
|---|---|---|
| 新 harness root/joint 写入 | 已通过 | 1a 五时刻 gate |
| omega 写入不是被忽略 | 已确认 | δ 敏感性测试 |
| ONNX/action manager 不改写状态 | 已确认 | 1b 五时刻 gate |
| fixed-action 0.05087 | 已确认 | 第 14 轮 A0 |
| 闭环是否缩小分叉 | 仍无法判定 | 1c 未完成 |
| 是否接触驱动 | 仍无法判定 | 阶段 B 未完成 |

## 5. 下一步

从当前通过的 1b 脚本继续添加 1c，不能再复制旧闭环：

1. 把 1b 中已验证的 `make_obs`、native/policy action mapping 原样加入；
2. 加入 decimation=4 的真实 ONNX policy step；
3. 每个 horizon step 前校验 root/joint；step 后只记录物理演化；
4. 输出 `isaac_v2_1c.json`；
5. 再写 MuJoCo v2，完成两个答案。

## 产物

- `/data/gaoli/logs/model_align_r11/isaac_v2_1a.py`
- `/data/gaoli/logs/model_align_r11/isaac_v2_1a.json`
- `/data/gaoli/logs/model_align_r11/isaac_v2_1b.py`
- `/data/gaoli/logs/model_align_r11/isaac_v2_1b.json`
- `/data/gaoli/logs/model_align_r11/report.md`
