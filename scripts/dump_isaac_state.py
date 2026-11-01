#!/usr/bin/env python
"""导出 Isaac 的真实状态 + 观测（不强行改写状态，避免缓存问题）。

用法:
  python dump_isaac_state.py --motion_file <npz> --onnx <onnx> --out <out.npz>
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-PI-Plus-Wo-v0")
parser.add_argument("--motion_file", required=True)
parser.add_argument("--onnx", required=True)
parser.add_argument("--out", default="/data/gaoli/logs/isaac_state.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import onnxruntime
import gymnasium as gym
import whole_body_tracking.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = args.task
print("\n" + "=" * 74)
print(" Isaac 真实状态导出")
print("=" * 74)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
cfg.commands.motion.motion_file = args.motion_file
env = gym.make(TASK, cfg=cfg)
env.reset()

e = env.unwrapped
robot = e.scene["robot"]
mc = e.command_manager.get_term("motion")

# 跑几步让状态稳定（此时是随机化的初始状态，没关系）
obs_dict, _ = env.reset()
for _ in range(3):
    obs_dict, *_ = env.step(torch.zeros((1, 22), device=e.device))

obs = obs_dict["policy"][0].detach().cpu().numpy().astype(np.float32)

# ---- 导出真实状态 ----
joint_names = list(robot.data.joint_names)
body_names = list(robot.data.body_names)
ts = int(mc.time_steps[0].item())

root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
joint_vel = robot.data.joint_vel[0].detach().cpu().numpy()
default_pos = robot.data.default_joint_pos[0].detach().cpu().numpy()

print(f"\n### Isaac 关节顺序（robot.data.joint_names）###")
for i, n in enumerate(joint_names):
    print(f"  [{i:2d}] {n}")

print(f"\n### Isaac 身体顺序前 8 ###")
for i, n in enumerate(body_names[:8]):
    print(f"  [{i:2d}] {n}")

print(f"\n  time_step      : {ts}")
print(f"  基座位置        : {np.round(root_pos,6)}")
print(f"  基座四元数      : {np.round(root_quat,6)}")
print(f"  base_ang_vel_b : {np.round(root_ang_vel_b,6)}")
print(f"  joint_pos 前6  : {np.round(joint_pos[:6],5)}")
print(f"  joint_vel 前6  : {np.round(joint_vel[:6],5)}")
print(f"  default_pos 前6: {np.round(default_pos[:6],5)}")
print(f"  参考joint_pos  : {np.round(mc.joint_pos[0].detach().cpu().numpy()[:6],5)}")

print(f"\n### 观测 ###")
print(f"  [0:44]   command        前6: {np.round(obs[0:6],6)}")
print(f"  [44:50]  motion_ori         : {np.round(obs[44:50],6)}")
print(f"  [50:53]  base_ang_vel        : {np.round(obs[50:53],6)}")
print(f"  [53:75]  joint_pos_rel   前6: {np.round(obs[53:59],6)}")
print(f"  [75:97]  joint_vel       前6: {np.round(obs[75:81],6)}")
print(f"  [97:119] last_action     前6: {np.round(obs[97:103],6)}")

np.savez(args.out, obs=obs, joint_names=np.array(joint_names),
         body_names=np.array(body_names), time_step=ts,
         root_pos=root_pos, root_quat=root_quat, root_ang_vel_b=root_ang_vel_b,
         joint_pos=joint_pos, joint_vel=joint_vel, default_pos=default_pos,
         ref_joint_pos=mc.joint_pos[0].detach().cpu().numpy())
print(f"\n[保存] {args.out}")
print("=" * 74)

env.close()
simulation_app.close()
