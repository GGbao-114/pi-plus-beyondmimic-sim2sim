#!/usr/bin/env python
"""在 Isaac Sim 里用 ONNX 策略跑环境 —— 验证导出是否正确。

这是关键测试：play.py 跑的是 PyTorch 策略，而 sim2sim 用的是 ONNX。
中间隔着导出步骤，从未被验证。

用法:
  python validate_onnx_in_isaac.py \
    --task Tracking-Flat-PI-Plus-Wo-v0 \
    --motion_file <npz> \
    --onnx <onnx路径> \
    --num_envs 4 --steps 500
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-PI-Plus-Wo-v0")
parser.add_argument("--motion_file", required=True)
parser.add_argument("--onnx", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import onnx
import onnxruntime
import gymnasium as gym
import whole_body_tracking.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = args.task
print("\n" + "=" * 66)
print(" 在 Isaac 里用 ONNX 策略跑环境")
print("=" * 66)

# ---- ONNX 元数据 ----
m = onnx.load(args.onnx)
meta = {p.key: p.value for p in m.metadata_props}
obs_names = meta.get("observation_names", "?").split(",")
print(f"[ONNX] 元数据 {len(meta)} 条")
print(f"[ONNX] observation_names = {obs_names}")
print(f"[ONNX] joint_names = {meta.get('joint_names','?')[:80]}...")

sess = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
in_names = [i.name for i in sess.get_inputs()]
out_names = [o.name for o in sess.get_outputs()]
print(f"[ONNX] 输入 {in_names}")
print(f"[ONNX] 输出 {out_names}")

# ---- 环境 ----
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
cfg.commands.motion.motion_file = args.motion_file
env = gym.make(TASK, cfg=cfg)

# 打印真实的观测布局（关键！）
om = env.unwrapped.observation_manager
print(f"\n[ENV] policy 观测项: {om.active_terms['policy']}")
dims = om.group_obs_dim["policy"]
print(f"[ENV] policy 观测维度: {dims}  总计 {sum(dims)}")

obs, _ = env.reset()
policy_obs = obs["policy"]
print(f"[ENV] policy obs shape: {tuple(policy_obs.shape)}")

# ---- 用 ONNX 跑 ----
device = env.unwrapped.device
n = args.num_envs
heights = []
terminated_count = np.zeros(n, dtype=bool)
terminated_step = np.full(n, -1, dtype=int)

print(f"\n[运行] {args.steps} 步 x {n} 环境")
print("=" * 66)
print(f"{'步':>5s} {'基座高度(均值)':>14s} {'已终止':>8s} {'参考高度':>10s}")
print("=" * 66)

for step in range(args.steps):
    po = policy_obs.detach().cpu().numpy().astype(np.float32)
    acts = []
    for b in range(n):
        a = sess.run(["actions"], {
            "obs": po[b:b+1],
            "time_step": np.array([[float(step)]], dtype=np.float32),
        })[0]
        acts.append(np.asarray(a).reshape(1, -1))
    action = torch.tensor(np.concatenate(acts, axis=0), dtype=torch.float32, device=device)

    obs, rew, terminated, truncated, info = env.step(action)
    policy_obs = obs["policy"]

    robot = env.unwrapped.scene["robot"]
    h = robot.data.root_pos_w[:, 2].detach().cpu().numpy()
    heights.append(h.mean())

    term = (terminated | truncated).detach().cpu().numpy()
    newly = term & (~terminated_count)
    if newly.any():
        terminated_step[newly] = step
        terminated_count |= term

    if step % 25 == 0 or step < 5:
        print(f"{step:5d} {h.mean():14.4f} {terminated_count.sum():8d}")

print("=" * 66)
heights = np.array(heights)
alive = (terminated_step < 0) | (terminated_step > args.steps - 1)
print(f"\n[结果]")
print(f"  平均基座高度(前50步) : {heights[:50].mean():.4f} m")
print(f"  平均基座高度(后50步) : {heights[-50:].mean():.4f} m")
print(f"  最低高度             : {heights.min():.4f} m")
print(f"  {args.steps} 步后仍存活 : {alive.sum()} / {n}")
for b in range(n):
    ts = terminated_step[b]
    print(f"    env[{b}]: " + (f"第 {ts} 步终止" if ts >= 0 else "存活"))
print("=" * 66)

# 高度轨迹可视化
print("\n  高度轨迹（每 25 步）:")
for i in range(0, len(heights), 25):
    bar = "#" * int(max(0, heights[i]) * 100)
    print(f"    {i:4d}  {heights[i]:.4f}  {bar}")

env.close()
simulation_app.close()
