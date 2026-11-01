"""Controlled Isaac counterpart to the MuJoCo frame-zero, zero-velocity hold test.

Uses the original implicit actuators and native PhysX timestep. Direct physics
stepping intentionally bypasses RL termination, auto-reset and command updates.
The resample protocol is a separate reference-reset diagnostic, not deployment.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--runs", nargs="+", default=[
    "2026-10-03_23-12-06", "2026-10-03_23-12-15",
    "2026-10-04_09-59-28", "2026-10-04_09-59-32"])
parser.add_argument("--protocols", nargs="+", choices=["hold", "resample"], default=["hold", "resample"])
parser.add_argument("--steps_limit", type=int, default=0, help="short verification run only")
parser.add_argument("--duration", type=float, default=0.0, help="override hold duration in seconds")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch
from isaaclab_tasks.utils import parse_env_cfg
from whole_body_tracking.tasks.tracking.mdp.commands import MotionLoader
import whole_body_tracking.tasks  # noqa: F401

torch.set_num_threads(1)


def cpu(value):
    return value.detach().cpu().numpy().copy()


def quat_matrix(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def obs_numpy(robot, ids, jp, jv, bq, fi, defaults, last_action):
    q = cpu(robot.data.root_quat_w[0])
    ref = bq[fi, 0]
    # Relative quaternion conjugate(q) * ref; wxyz convention.
    w1, x1, y1, z1 = q[0], -q[1], -q[2], -q[3]
    w2, x2, y2, z2 = ref
    relative = np.array([
        w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
    obs = np.zeros((1, 119), np.float32)
    obs[0, :44] = np.concatenate((jp[fi], jv[fi]))
    obs[0, 44:50] = quat_matrix(relative)[:, :2].reshape(6)
    obs[0, 50:53] = cpu(robot.data.root_ang_vel_b[0])
    obs[0, 53:75] = cpu(robot.data.joint_pos[0, ids]) - defaults
    obs[0, 75:97] = cpu(robot.data.joint_vel[0, ids])
    obs[0, 97:] = last_action
    return obs


def main():
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    entries = [entry for entry in manifest if entry["run"] in args.runs]
    if len(entries) != len(args.runs):
        raise ValueError("requested runs not all found")
    cfg = parse_env_cfg("Tracking-Flat-PI-Plus-Wo-v0", device=args.device, num_envs=1)
    cfg.commands.motion.motion_file = entries[0]["motion"]
    cfg.commands.motion.debug_vis = False
    cfg.commands.motion.force_curriculum_enabled = False
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.observations.policy.enable_corruption = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.scene.terrain.visual_material = None
    disabled_events = []
    for name, value in list(vars(cfg.events).items()):
        if not name.startswith("_") and hasattr(value, "mode"):
            setattr(cfg.events, name, None)
            disabled_events.append(name)
    for name, value in list(vars(cfg.terminations).items()):
        if not name.startswith("_") and hasattr(value, "func"):
            setattr(cfg.terminations, name, None)
    env = gym.make("Tracking-Flat-PI-Plus-Wo-v0", cfg=cfg)
    e = env.unwrapped
    robot = e.scene["robot"]
    mc = e.command_manager.get_term("motion")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    dt = float(e.physics_dt)
    ctrl_dt = float(e.step_dt)
    if abs(ctrl_dt - 0.02) > 1e-8:
        raise ValueError(f"unexpected control period {ctrl_dt}")
    # Identity mapping is checked against the actual articulation, never guessed.
    standard_names = [
        "l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint",
        "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint",
        "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint",
        "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint",
        "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint",
        "l_ankle_roll_joint", "r_ankle_roll_joint"]
    ids = robot.find_joints(standard_names, preserve_order=True)[0]
    # ActionManager follows the articulation order; explicitly permute actions.
    native_names = list(robot.data.joint_names)
    native_to_policy = [standard_names.index(name) for name in native_names]
    policy_to_native = [native_names.index(name) for name in standard_names]
    defaults = cpu(robot.data.default_joint_pos[0, ids])
    info = dict(
        physics_dt=dt, decimation=cfg.decimation, control_dt=ctrl_dt,
        implicit_actuators=True, device=str(e.device), disabled_events=disabled_events,
        observation_noise=False, joint_names=native_names, policy_joint_names=standard_names,
        default_joint_pos=defaults.tolist(),
        stiffness=cpu(robot.data.joint_stiffness[0, ids]).tolist(),
        damping=cpu(robot.data.joint_damping[0, ids]).tolist(),
        effort_limits=cpu(robot.data.joint_effort_limits[0, ids]).tolist(),
        velocity_limits=cpu(robot.data.joint_vel_limits[0, ids]).tolist(),
        armature=cpu(robot.data.joint_armature[0, ids]).tolist(),
        material_properties=cpu(robot.root_physx_view.get_material_properties()).tolist(),
        body_masses=cpu(robot.root_physx_view.get_masses()).tolist(),
        source_note="matched observation/initial state/reference; native PhysX implicit PD and velocity limits retained",
    )
    (args.out / "config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    rows = []
    env.reset(seed=0)
    # Guard state writes during each rollout, so unrecorded teleports cannot masquerade as survival.
    raw_root_writer = robot.write_root_state_to_sim
    raw_joint_writer = robot.write_joint_state_to_sim
    writes = {"root": 0, "joint": 0}

    def root_writer(*a, **kw):
        writes["root"] += 1
        return raw_root_writer(*a, **kw)

    def joint_writer(*a, **kw):
        writes["joint"] += 1
        return raw_joint_writer(*a, **kw)

    robot.write_root_state_to_sim = root_writer
    robot.write_joint_state_to_sim = joint_writer
    for entry in entries:
        with np.load(entry["motion"]) as motion:
            jp, jv, bp, bq = [motion[key].copy() for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")]
            fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if fps != 50:
            raise ValueError(f"expected 50 Hz reference, got {fps}")
        policy = Path(entry["checkpoint"]).parent / "exported" / (Path(entry["checkpoint"]).stem + ".onnx")
        session = ort.InferenceSession(str(policy), sess_options=opts, providers=["CPUExecutionProvider"])
        mc.motion = MotionLoader(entry["motion"], mc.body_indexes, device=e.device)
        for protocol in args.protocols:
            print(f"CASE_START {entry['run']} {protocol}", flush=True)
            torch.manual_seed(0)
            e.scene.reset()
            e.action_manager.reset()
            mc.time_steps.zero_()
            root = torch.zeros((1, 13), dtype=torch.float32, device=e.device)
            root[0, :3] = torch.as_tensor(bp[0, 0], device=e.device) + e.scene.env_origins[0]
            root[0, 3:7] = torch.as_tensor(bq[0, 0] / np.linalg.norm(bq[0, 0]), device=e.device)
            q = torch.zeros_like(robot.data.joint_pos)
            q[0, ids] = torch.as_tensor(jp[0], device=e.device)
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.set_joint_velocity_target(torch.zeros_like(q))
            robot.set_joint_effort_target(torch.zeros_like(q))
            e.scene.write_data_to_sim()
            e.sim.forward()
            e.scene.update(0.0)
            initial_root = cpu(robot.data.root_state_w[0])
            initial_q = cpu(robot.data.joint_pos[0, ids])
            initial_dq = cpu(robot.data.joint_vel[0, ids])
            if np.abs(initial_q-jp[0]).max()>1e-5 or np.abs(initial_dq).max()>1e-5:
                raise ValueError("initial joint state mismatch")
            if np.abs(initial_root[:3]-(bp[0,0]+cpu(e.scene.env_origins[0]))).max()>1e-5:
                raise ValueError("initial root position mismatch")
            initial = dict(root_state=initial_root.tolist(), joint_pos=initial_q.tolist(), joint_vel=initial_dq.tolist())
            writes.update(root=0, joint=0)
            duration = args.duration if args.duration > 0 else (15.0 if "kungfu" in entry["motion"] else 8.0)
            steps = int(round(duration / ctrl_dt))
            if args.steps_limit:
                steps = min(steps, args.steps_limit)
            fi = 0
            last_action = np.zeros(22, np.float32)
            frames = {key: [] for key in (
                "time", "reference_frame", "root_pos", "base_quat", "base_omega", "root_lin_vel",
                "joint_pos", "joint_vel", "action", "obs", "computed_torque_estimate",
                "contact_net_force", "body_pos", "joint_mae", "heights")}
            resets = []
            fall = None
            obs_error_max = 0.0
            for step in range(steps):
                mc.time_steps.fill_(fi)
                obs = obs_numpy(robot, ids, jp, jv, bq, fi, defaults, last_action)
                # Check against the original no-noise ObservationManager contract.
                env_obs = cpu(e.observation_manager.compute()["policy"])
                native_obs = obs.copy()
                native_obs[0, :22] = jp[fi, native_to_policy]
                native_obs[0, 22:44] = jv[fi, native_to_policy]
                native_obs[0, 53:75] = obs[0, 53:75][native_to_policy]
                native_obs[0, 75:97] = obs[0, 75:97][native_to_policy]
                native_obs[0, 97:] = last_action[native_to_policy]
                obs_error_max = max(obs_error_max, float(np.abs(env_obs-native_obs).max()))
                if obs_error_max > 1e-4:
                    raise ValueError(f"observation implementation mismatch: {obs_error_max}")
                action = session.run(["actions"], {"obs": obs, "time_step": np.array([[fi]], np.float32)})[0][0]
                if not np.isfinite(action).all():
                    raise FloatingPointError("nonfinite policy output")
                e.action_manager.process_action(torch.as_tensor(action[native_to_policy][None], device=e.device))
                for _ in range(cfg.decimation):
                    e._sim_step_counter += 1
                    e.action_manager.apply_action()
                    e.scene.write_data_to_sim()
                    e.sim.step(render=False)
                    e.scene.update(dt)
                e.episode_length_buf += 1
                e.common_step_counter += 1
                next_fi = min(fi+1, len(jp)-1)
                height = float(robot.data.root_pos_w[0,2].item())
                t = (step+1)*ctrl_dt
                if fall is None and height<0.15:
                    fall=t
                q_now=cpu(robot.data.joint_pos[0,ids])
                dq_now=cpu(robot.data.joint_vel[0,ids])
                if not np.isfinite(q_now).all() or not np.isfinite(height):
                    raise FloatingPointError("nonfinite state")
                frames["time"].append(t)
                frames["reference_frame"].append(next_fi)
                frames["root_pos"].append(cpu(robot.data.root_pos_w[0]))
                frames["base_quat"].append(cpu(robot.data.root_quat_w[0]))
                frames["base_omega"].append(cpu(robot.data.root_ang_vel_b[0]))
                frames["root_lin_vel"].append(cpu(robot.data.root_lin_vel_w[0]))
                frames["joint_pos"].append(q_now)
                frames["joint_vel"].append(dq_now)
                frames["action"].append(action.copy())
                frames["obs"].append(obs[0].copy())
                frames["computed_torque_estimate"].append(cpu(robot.data.computed_torque[0,ids]))
                frames["contact_net_force"].append(cpu(e.scene["contact_forces"].data.net_forces_w[0]))
                frames["body_pos"].append(cpu(robot.data.body_pos_w[0,mc.body_indexes]))
                frames["joint_mae"].append(float(np.abs(q_now-jp[next_fi]).mean()))
                frames["heights"].append(height)
                last_action=action.copy()
                if protocol=="resample" and fi+1>=len(jp):
                    # Same adaptive frame sampling/state writer as training; deterministic seed,
                    # zero pose/velocity perturbation in this controlled diagnostic.
                    mc._resample_command(torch.tensor([0], device=e.device))
                    fi=int(mc.time_steps[0].item())
                    resets.append(dict(time_s=t, sampled_frame=fi))
                    e.sim.forward()
                    e.scene.update(0.0)
                else:
                    fi=next_fi
                if protocol=="hold" and (writes["root"] or writes["joint"]):
                    raise RuntimeError("unexpected state reset in hold protocol")
                if step%100==0:
                    print(f"STEP {entry['run']} {protocol} t={t:.2f} h={height:.4f} frame={fi} fall={fall}",flush=True)
            arrays={key:np.asarray(value) for key,value in frames.items()}
            np.savez_compressed(args.out/f"{entry['run']}_{protocol}.npz",**arrays)
            row=dict(run=entry['run'],motion=Path(entry['motion']).name,protocol=protocol,
                     status="OK",duration=steps*ctrl_dt,first_fall_s=fall,min_height=float(arrays['heights'].min()),
                     joint_mae=float(arrays['joint_mae'].mean()),obs_error_max=obs_error_max,
                     state_writes=writes.copy(),resample_events=resets,initial_state=initial,
                     reference_end_s=(len(jp)-1)/fps)
            rows.append(row)
            (args.out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print('CASE_RESULT '+json.dumps(row),flush=True)
    env.close()
    (args.out/'DONE').write_text('completed\n')
    print('ISAAC_COMPARISON_DONE',flush=True)


exit_code=0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code=1
finally:
    app.close()
sys.exit(exit_code)
