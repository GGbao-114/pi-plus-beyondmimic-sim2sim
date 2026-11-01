import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path('/data/gaoli/logs/model_align_r31')
XML = Path('/data/gaoli/logs/model_align_r13/pi_auto_fixedmass.xml')
STATE = Path('/data/gaoli/logs/model_align_r2/isaac_goal_contact/T4_goal_hold.npz')
OUT = ROOT / 'mujoco_contact_diag.json'
DT = 0.005

BFS = [
    'l_hip_pitch_joint', 'l_shoulder_pitch_joint', 'r_hip_pitch_joint', 'r_shoulder_pitch_joint',
    'l_hip_roll_joint', 'l_shoulder_roll_joint', 'r_hip_roll_joint', 'r_shoulder_roll_joint',
    'l_thigh_joint', 'l_upper_arm_joint', 'r_thigh_joint', 'r_upper_arm_joint',
    'l_calf_joint', 'l_elbow_joint', 'r_calf_joint', 'r_elbow_joint',
    'l_ankle_pitch_joint', 'l_wrist_joint', 'r_ankle_pitch_joint', 'r_wrist_joint',
    'l_ankle_roll_joint', 'r_ankle_roll_joint',
]
ARM_SUFFIXES = ('_shoulder_pitch_joint', '_shoulder_roll_joint', '_upper_arm_joint', '_elbow_joint', '_wrist_joint')
SAMPLES = (2.0, 2.5, 3.0, 3.5, 4.0)
HORIZONS = (1, 5, 20, 50)


def gains(name):
    return (30.0, 0.6) if name.endswith(ARM_SUFFIXES) else (80.0, 1.1)


def default_pos(name):
    if name.endswith('_hip_pitch_joint'):
        return -0.25
    if name.endswith('_calf_joint'):
        return 0.65
    if name.endswith('_ankle_pitch_joint'):
        return -0.4
    return 0.0


def scale(name):
    kp, _ = gains(name)
    return 0.25 * (20.0 if name.endswith(ARM_SUFFIXES) else 30.0) / kp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--xml', type=Path, default=XML)
    args = parser.parse_args()
    state_path = args.state
    out_path = args.out
    xml_path = args.xml
    with np.load(state_path) as data:
        state = {key: data[key].copy() for key in data.files}
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.opt.timestep = DT
    d = mujoco.MjData(m)
    jids = {m.joint(i).name: i for i in range(m.njnt) if m.joint(i).name in BFS}
    if set(jids) != set(BFS):
        raise RuntimeError(f'missing joints: {sorted(set(BFS)-set(jids))}')
    qadr = {name: int(m.jnt_qposadr[jids[name]]) for name in BFS}
    dadr = {name: int(m.jnt_dofadr[jids[name]]) for name in BFS}
    # Match the parameters injected by the established MuJoCo evaluator.
    # The generated MJCF has no URDF armature; omitting this makes small arm
    # inertias numerically unstable and would invalidate the engine comparison.
    for name in BFS:
        dof = dadr[name]
        m.dof_armature[dof] = 0.008419 if name.endswith(ARM_SUFFIXES) else 0.044277
        if m.dof_damping[dof] < 0.01:
            m.dof_damping[dof] = 0.05

    # Use generalized applied force so both engines receive the same torque.
    # This deliberately avoids comparing two copies of an actuator formula.
    rows = []
    time_to_index = {round(float(t), 3): i for i, t in enumerate(state['time'])}
    for wanted_time in SAMPLES:
        i = time_to_index[round(wanted_time, 3)]
        q0 = state['joint_pos'][i]
        dq0 = state['joint_vel'][i]
        root_pos = state['root_pos'][i]
        root_quat = state['base_quat'][i] / np.linalg.norm(state['base_quat'][i])
        root_lin_vel = state['root_lin_vel'][i]
        root_ang_vel_body = state['base_omega'][i]
        action = state['action'][i]
        target = np.array([action[k] * scale(name) + default_pos(name) for k, name in enumerate(BFS)])
        requested = np.array([
            gains(name)[0] * (target[k] - q0[k]) - gains(name)[1] * dq0[k]
            for k, name in enumerate(BFS)
        ])
        limits = np.array([20.0 if name.endswith(ARM_SUFFIXES) else 30.0 for name in BFS])
        tau = np.clip(requested, -limits, limits)
        horizon_rows = []
        for horizon in HORIZONS:
            d.qpos[:] = 0.0
            d.qvel[:] = 0.0
            d.qpos[:3] = root_pos
            d.qpos[3:7] = root_quat
            d.qvel[:3] = root_lin_vel
            d.qvel[3:6] = root_ang_vel_body
            for k, name in enumerate(BFS):
                d.qpos[qadr[name]] = q0[k]
                d.qvel[dadr[name]] = dq0[k]
            mujoco.mj_forward(m, d)
            q_before = np.array([d.qpos[qadr[name]] for name in BFS])
            dq_before = np.array([d.qvel[dadr[name]] for name in BFS])
            closed_loop_torques = []
            step_frames = []
            for _ in range(horizon):
                # Match Isaac's fixed-action protocol: the target stays fixed,
                # while implicit PD recomputes effort from each engine's own
                # current q/dq on every physics step.
                current_q = np.array([d.qpos[qadr[name]] for name in BFS])
                current_dq = np.array([d.qvel[dadr[name]] for name in BFS])
                current_requested = np.array([
                    gains(name)[0] * (target[k] - current_q[k]) - gains(name)[1] * current_dq[k]
                    for k, name in enumerate(BFS)
                ])
                current_tau = np.clip(current_requested, -limits, limits)
                d.qfrc_applied[:] = 0.0
                for k, name in enumerate(BFS):
                    d.qfrc_applied[dadr[name]] = current_tau[k]
                closed_loop_torques.append(current_tau.copy())
                mujoco.mj_step(m, d)
                pairs=[]
                for ci in range(d.ncon):
                    c=d.contact[ci]; f=np.zeros(6); mujoco.mj_contactForce(m,d,ci,f)
                    pairs.append({'geom1':m.geom(c.geom1).name,'geom2':m.geom(c.geom2).name,'point':np.asarray(c.pos).tolist(),'normal_force':float(abs(f[0]))})
                step_frames.append({'step':len(step_frames)+1,'q':np.array([d.qpos[qadr[name]] for name in BFS]).tolist(),'dq':np.array([d.qvel[dadr[name]] for name in BFS]).tolist(),'contacts':pairs,'ncon':int(d.ncon)})
            q_after = np.array([d.qpos[qadr[name]] for name in BFS])
            dq_after = np.array([d.qvel[dadr[name]] for name in BFS])
            contacts = []
            for ci in range(d.ncon):
                contact = d.contact[ci]
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, force)
                contacts.append({
                    'geom1': m.geom(contact.geom1).name,
                    'geom2': m.geom(contact.geom2).name,
                    'dist': float(contact.dist),
                    'point': np.asarray(contact.pos).tolist(),
                    'normal_force_contact_frame': float(abs(force[0])),
                })
            horizon_rows.append({
                'horizon_steps': horizon,
                'dt': DT,
                'actual_step_count': horizon,
                'q_before': q_before.tolist(),
                'dq_before': dq_before.tolist(),
                'q_after': q_after.tolist(),
                'dq_after': dq_after.tolist(),
                'root_pos_after': d.qpos[:3].copy().tolist(),
                'root_quat_after': d.qpos[3:7].copy().tolist(),
                'root_lin_vel_after': d.qvel[:3].copy().tolist(),
                'root_ang_vel_after': d.qvel[3:6].copy().tolist(),
                'q_delta_from_initial': (q_after - q0).tolist(),
                'dq_delta_from_initial': (dq_after - dq0).tolist(),
                'closed_loop_torque_first': closed_loop_torques[0].tolist(),
                'closed_loop_torque_last': closed_loop_torques[-1].tolist(),
                'ncon_after': int(d.ncon),
                'contacts_after': contacts,
                'step_frames': step_frames,
            })
        rows.append({
            'time': wanted_time,
            'source_index': i,
            'dt': DT,
            'action': action.tolist(),
            'target': target.tolist(),
            'requested_tau': requested.tolist(),
            'applied_tau': tau.tolist(),
            'joint_names': BFS,
            'horizons': horizon_rows,
        })
    result = {
        'experiment': 'mujoco_same_state_same_torque_step',
        'strict_dynamics_test': True,
        'xml': str(xml_path),
        'state_npz': str(state_path),
        'dt': DT,
        'step_count_per_horizon': list(HORIZONS),
        'state_write_count': 0,
        'torque_source': 'closed-loop fixed target/action: each engine recomputes clipped PD effort from its own current q/dq; no MuJoCo actuator formula comparison',
        'rows': rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out_path), 'rows': len(rows), 'dt': DT}, indent=2))


if __name__ == '__main__':
    main()
