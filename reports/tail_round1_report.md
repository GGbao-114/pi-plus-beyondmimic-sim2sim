# bigkick 收尾段实验（第 1 轮）

## 阶段 1：原链路复现

- 原 CSV：54 帧、30 Hz；原转换器输出 89 帧、50 Hz。
- 原 CSV→NPZ 全部 8 个数组与原始 NPZ 最大差为 0。
- 原文件、原权重均保留，新文件仅放在本实验目录。

## 阶段 2：尾段与一致性

- 从原 NPZ 前89帧构造同频 CSV，按关节名映射到 CSV 列。
- 1.5 秒速度连续五次多项式过渡 + 1.2 秒站姿保持，目标为默认站姿、保留末端 yaw。
- 用 MuJoCo 库 FK / geomDistance 调整支撑中点和根高度；身体轨迹和速度由原 Isaac FK 转换链路计算。
- 第一版有约6.3 mm穿地，第二版修正根高度；第三版去掉50→50重复插值以保留每帧四元数。v1/v2产物保留。
- 最终 NPZ：224帧，末帧时刻4.46秒。
- 参考足水平位移上界 1.772 cm；实际动态足滑移不能据此推断。
- 尾段最深脚底间隙 -0.0816 mm。
- 库 FK 跨平台足位置差最大 4.355e-07 m。

| 前89帧字段 | 最大绝对差 |
|---|---|
| fps | 0 |
| joint_pos | 5.82076609e-11 |
| joint_vel | 0.000457465649 |
| body_pos_w | 1.49011612e-07 |
| body_quat_w | 2.38418579e-07 |
| body_lin_vel_w | 0.0129673425 |
| body_ang_vel_w | 0.0154206306 |

末帧速度采用新的中心差分，因此第89帧邻域会变化；不是所有速度数组完全不变。位置/姿态保持近机器精度一致。
末尾50帧关节速度最大 0.000e+00 rad/s；身体线/角速度最大分别 6.260e-07 m/s / 1.490e-06 rad/s。

## 阶段 3/4：A-B-C 零复位评估

| 组 | 引擎 | 首次倒地 | 原动作段MAE | 尾段结束后站稳时长 | 重置次数 | 验收 |
|---|---|---|---|---|---|---|
| A 原策略＋原动作 | MuJoCo | 2.78 s | 0.066908 rad | 1.02 s | 0 | 未通过 |
| A 原策略＋原动作 | Isaac | 3.30 s | 0.056275 rad | 1.54 s | 0 | 未通过 |
| B | MuJoCo | 2.62 s | 0.066908 rad | 0.00 s | 0 | 未通过 |
| B | Isaac | 2.76 s | 0.056261 rad | 0.00 s | 0 | 未通过 |
| C | MuJoCo | 8 s 未倒 | 0.076680 rad | 3.54 s | 0 | 未通过 |
| C | Isaac | 8 s 未倒 | 0.068551 rad | 3.54 s | 0 | 未通过 |

验收按原动作段MAE不劣于A的110%、8秒无倒地、尾段结束后至少2秒站稳、零复位同时判断。
B 失败不能证明新尾段动态上必然不可行，也不能证明训练后必然成功；C 用同一数据微调后才能进一步区分。

## 导出与执行状态

- B：actor 对 checkpoint 最大差 3.338e-06；全部参考帧6个输出对NPZ差 0.000e+00；元数据10条；policy=/data/gaoli/logs/tail_round1/export_B/exported/model_9999.onnx。
- C：actor 对 checkpoint 最大差 2.623e-06；全部参考帧6个输出对NPZ差 0.000e+00；元数据10条；policy=/data/gaoli/logs/tail_round1/C_train/exported/model_11998.onnx。

## C 中间 checkpoint 检查

| checkpoint | 8秒是否倒地 | 原踢腿段 MuJoCo MAE | 满足10%退化上限 |
|---|---|---|---|
| model_10000.pt | 倒地 | 0.074742 rad | 否 |
| model_10500.pt | 未倒 | 0.083808 rad | 否 |
| model_11000.pt | 未倒 | 0.074054 rad | 否 |
| model_11500.pt | 未倒 | 0.075216 rad | 否 |
| model_11998.pt | 未倒 | 0.076680 rad | 否 |

5个保存的checkpoint均未同时满足MuJoCo稳定和精度条件。该检查使用checkpoint+保存归一化的PyTorch actor；最终checkpoint已另与ONNX数值对齐。
不因最后checkpoint未达精度标准就宣布微调完全无效，也不通过放宽标准宣布通过。
- 导出使用 play.py 实验副本，有限步退出、actor-only导出注册模块；完整导出强制legacy exporter避免opset降级失败。
- 先内联外部权重，再执行add_metadata.py，最后按真实关节/身体名称和名义参数修复并检查。原模型不覆盖。
- 微调使用原train.py副本，仅修复resume+自定义log目录时log_root_path未定义的问题。4096环境，恢复model_9999.pt，追加2000轮，tensorboard日志。
- C训练/导出/复评的自动续跑脚本为run_C_and_export.sh，tmux会话tail_round1_C。
- 成功标准不是训练reward，而是C双引擎零复位评估。若C仍失败，报告如实标注，不宣布收尾问题已解决。

## 文件与命令

- 原转换：bash run_pipeline_repro.sh
- 尾段生成：bash run_tail_generation.sh
- B评估：bash run_B.sh
- C训练、导出、双引擎评估：bash run_C_and_export.sh
- 生成报告/GIF：python make_report.py（服务器必须用绝对路径Python）
- GIF：A_bigkick_realtime.gif、B_bigkick.gif、A_vs_B_bigkick.gif；C完成后增加C_bigkick.gif、A_vs_C_bigkick.gif。
- A动画来自历史giffix_bigkick.gif，校正为约8秒回放；用于定性演示，不用来代替数值轨迹验收。
- 所有新数据/脚本/日志/权重在/data/gaoli/logs/tail_round1/。

**训练与评估流程完成标记：已完成。**

## 本轮最终结论

- 原策略无法直接执行新收尾（B两边失败），需要学习新的参考段。
- 追加2000轮后C两边均8秒零复位未倒，末帧后保持3.54秒；收尾稳定性目标实现。
- 原踢腿段精度代价：MuJoCo 14.60%、Isaac 21.82%退化，超过10%。
- 因而整体严格验收：未通过（稳定性通过，动作精度未通过）。
- 本轮已按2000～4000范围的下限完成；没有盲目再加2000轮。建议下一轮采用原踢腿片段重放/加权采样或旧策略动作蒸馏约束，保留收尾能力同时降低动作段退化。
- kick1015 后续可复用收尾生成流程，但不同checkpoint不能直接宣布同样通过；本轮完成的是bigkick A/B/C。