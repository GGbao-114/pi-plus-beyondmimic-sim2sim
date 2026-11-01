import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path("/data/gaoli/logs/model_align_r30")
XML_EXPLICIT = Path("/data/gaoli/logs/model_align_r13/pi_auto_fixedmass.xml")
XML_POSITION = ROOT / "pi_auto_fixedmass_position.xml"
BFS = ["l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint", "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint", "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint", "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint", "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint", "l_ankle_roll_joint", "r_ankle_roll_joint"]
ARM = ("_shoulder_pitch_joint", "_shoulder_roll_joint", "_upper_arm_joint", "_elbow_joint", "_wrist_joint")
SAMPLES = (2.0, 2.5, 3.0, 3.5, 4.0)
HORIZONS = (1, 5, 20, 50)
DT = 0.005
MOTIONS = {
    "goal": ("/data/gaoli/logs/model_align_r2/isaac_goal_contact/T4_goal_hold.npz", "/data/gaoli/logs/model_align_r15/isaac_dynamics_step.json"),
    "kick1015": ("/data/gaoli/logs/tail_round3/diag_isaac/A_kick1015_hold.npz", "/data/gaoli/logs/model_align_r22/isaac_kick1015/isaac_dynamics_step.json"),
    "bigkick": ("/data/gaoli/logs/model_align_r23/bigkick_B_state.npz", "/data/gaoli/logs/model_align_r23/isaac_bigkick_B/isaac_dynamics_step.json"),
}


def gains(n): return (30.0, 0.6) if n.endswith(ARM) else (80.0, 1.1)
def default(n): return -0.25 if n.endswith("_hip_pitch_joint") else 0.65 if n.endswith("_calf_joint") else -0.4 if n.endswith("_ankle_pitch_joint") else 0.0
def scale(n): return 0.25 * (20.0 if n.endswith(ARM) else 30.0) / gains(n)[0]


def run(motion_name, mode):
    state_path, isaac_path = MOTIONS[motion_name]
    state = np.load(state_path)
    xml = XML_POSITION if mode == "position" else XML_EXPLICIT
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    jid = {model.joint(i).name: i for i in range(model.njnt) if model.joint(i).name in BFS}
    qa = {n: int(model.jnt_qposadr[jid[n]]) for n in BFS}
    da = {n: int(model.jnt_dofadr[jid[n]]) for n in BFS}
    aid = {model.actuator(i).name.removesuffix("_pos_ctrl"): i for i in range(model.nu)} if mode == "position" else {}
    for n in BFS:
        dof = da[n]
        model.dof_armature[dof] = 0.008419 if n.endswith(ARM) else 0.044277
        model.dof_damping[dof] = max(model.dof_damping[dof], 0.05)
    rows = []
    for wt in SAMPLES:
        i = int(np.argmin(abs(state["time"] - wt)))
        q0, dq0 = state["joint_pos"][i], state["joint_vel"][i]
        target = state["action"][i] * np.array([scale(n) for n in BFS]) + np.array([default(n) for n in BFS])
        hs = []
        for h in HORIZONS:
            data.qpos[:] = 0; data.qvel[:] = 0
            data.qpos[:3] = state["root_pos"][i]
            data.qpos[3:7] = state["base_quat"][i] / np.linalg.norm(state["base_quat"][i])
            data.qvel[:3] = state["root_lin_vel"][i]; data.qvel[3:6] = state["base_omega"][i]
            for k, n in enumerate(BFS): data.qpos[qa[n]] = q0[k]; data.qvel[da[n]] = dq0[k]
            mujoco.mj_forward(model, data)
            for _ in range(h):
                if mode == "position":
                    data.ctrl[:] = target
                    data.qfrc_applied[:] = 0
                else:
                    q = np.array([data.qpos[qa[n]] for n in BFS]); dq = np.array([data.qvel[da[n]] for n in BFS])
                    tau = np.array([gains(n)[0] * (target[k] - q[k]) - gains(n)[1] * dq[k] for k, n in enumerate(BFS)])
                    lim = np.array([20.0 if n.endswith(ARM) else 30.0 for n in BFS]); tau = np.clip(tau, -lim, lim)
                    data.qfrc_applied[:] = 0
                    for k, n in enumerate(BFS): data.qfrc_applied[da[n]] = tau[k]
                mujoco.mj_step(model, data)
            after = np.array([data.qpos[qa[n]] for n in BFS])
            hs.append({"horizon": h, "q_after": after.tolist()})
        rows.append({"time": wt, "horizons": hs})
    isaac = json.loads(Path(isaac_path).read_text())["rows"]
    gap = [[float(np.max(np.abs(np.asarray(ir["horizons"][h]["q_after"]) - np.asarray(vr["horizons"][h]["q_after"])))) for h in range(4)] for ir, vr in zip(isaac, rows)]
    return {"motion": motion_name, "mode": mode, "rows": rows, "gap": gap}


def main():
    results = []
    for motion in MOTIONS:
        for mode in ("explicit", "position"):
            print("RUN", motion, mode, flush=True)
            results.append(run(motion, mode))
    out = ROOT / "matched_control_results.json"
    out.write_text(json.dumps({"xml_explicit": str(XML_EXPLICIT), "xml_position": str(XML_POSITION), "results": results}, indent=2))
    for x in results:
        vals = np.asarray(x["gap"])[:, 3]
        print(x["motion"], x["mode"], "50=[", ", ".join(f"{v:.6f}" for v in vals), "] mean=", f"{vals.mean():.6f}", sep="")
    print(json.dumps({"out": str(out), "count": len(results)}))


if __name__ == "__main__": main()
