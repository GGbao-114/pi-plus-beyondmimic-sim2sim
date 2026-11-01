import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path("/data/gaoli/logs/model_align_r26")
XML = Path("/data/gaoli/logs/model_align_r13/pi_auto_fixedmass.xml")
MOTION_ROOT = Path("/data/gaoli/projects/Mini-Pi-Plus_BeyondMimic/source/motion/hightorque/pi_plus/npz")
DT = 0.005
DEC = 4
FALL_Z = 0.15
BFS = ["l_hip_pitch_joint", "l_shoulder_pitch_joint", "r_hip_pitch_joint", "r_shoulder_pitch_joint", "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint", "r_shoulder_roll_joint", "l_thigh_joint", "l_upper_arm_joint", "r_thigh_joint", "r_upper_arm_joint", "l_calf_joint", "l_elbow_joint", "r_calf_joint", "r_elbow_joint", "l_ankle_pitch_joint", "l_wrist_joint", "r_ankle_pitch_joint", "r_wrist_joint", "l_ankle_roll_joint", "r_ankle_roll_joint"]
ARM = ("_shoulder_pitch_joint", "_shoulder_roll_joint", "_upper_arm_joint", "_elbow_joint", "_wrist_joint")
MOTIONS = {
    "goal": (MOTION_ROOT / "pi_plus_goal.npz", ROOT.parent / "model_align_r24/goal_dr_run/exported/model_9999.onnx", ROOT.parent / "model_align_r25/baseline_goal/exported/model_9999.onnx"),
    "kick1015": (MOTION_ROOT / "kick1015.npz", ROOT.parent / "model_align_r24/dr_run_kick1015/exported/model_9999.onnx", ROOT.parent / "model_align_r25/baseline_kick1015/exported/model_9999.onnx"),
    "bigkick": (MOTION_ROOT / "pi_plus_right_big_kick.npz", ROOT.parent / "model_align_r24/dr_run_pi_plus_right_big_kick/exported/model_9999.onnx", ROOT.parent / "model_align_r25/baseline_bigkick/exported/model_9999.onnx"),
    "kungfu": (MOTION_ROOT / "pi_plus_kungfu.npz", ROOT.parent / "model_align_r24/dr_run_pi_plus_kungfu/exported/model_9999.onnx", ROOT.parent / "model_align_r25/baseline_kungfu/exported/model_9999.onnx"),
    "dance": (MOTION_ROOT / "pi_plus_dance1_subject2.npz", ROOT.parent / "model_align_r24/dr_run_pi_plus_dance1_subject2/exported/model_9999.onnx", ROOT.parent / "model_align_r25/baseline_dance/exported/model_9999.onnx"),
}


def gains(n): return (30.0, 0.6) if n.endswith(ARM) else (80.0, 1.1)
def default(n): return -0.25 if n.endswith("_hip_pitch_joint") else 0.65 if n.endswith("_calf_joint") else -0.4 if n.endswith("_ankle_pitch_joint") else 0.0


def obs(q, dq, root_quat, omega, ref_q, ref_dq, ref_quat, last):
    root_q = root_quat / np.linalg.norm(root_quat); r = ref_quat / np.linalg.norm(ref_quat)
    w1,x1,y1,z1=root_q[0],-root_q[1],-root_q[2],-root_q[3]; w2,x2,y2,z2=r
    rel=np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2]); rel/=np.linalg.norm(rel)
    w,x,y,z=rel
    mat=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    out=np.zeros(119,np.float32); out[:44]=np.r_[ref_q,ref_dq]; out[44:50]=mat[:,:2].reshape(6); out[50:53]=omega; out[53:75]=q-np.array([default(n) for n in BFS]); out[75:97]=dq; out[97:]=last; return out


def run(motion_name, policy_name, policy_path):
    motion=np.load(MOTIONS[motion_name][0]); meta={x.key:x.value for x in onnx.load(str(policy_path)).metadata_props}; scale=np.array([float(x) for x in meta['action_scale'].split(',')]); defaults=np.array([float(x) for x in meta['default_joint_pos'].split(',')]); sess=ort.InferenceSession(str(policy_path),providers=['CPUExecutionProvider'])
    nframes=len(motion['joint_pos']); duration=15.0 if motion_name=='kungfu' else 8.0; steps=int(round(duration/0.02)); m=mujoco.MjModel.from_xml_path(str(XML)); m.opt.timestep=DT; d=mujoco.MjData(m); ids={m.joint(i).name:i for i in range(m.njnt) if m.joint(i).name in BFS}; qa={n:int(m.jnt_qposadr[ids[n]]) for n in BFS}; da={n:int(m.jnt_dofadr[ids[n]]) for n in BFS}
    for n in BFS: m.dof_armature[da[n]]=0.008419 if n.endswith(ARM) else 0.044277; m.dof_damping[da[n]]=0.0
    d.qpos[:]=0; d.qvel[:]=0; d.qpos[:3]=motion['body_pos_w'][0,0]; d.qpos[3:7]=motion['body_quat_w'][0,0]/np.linalg.norm(motion['body_quat_w'][0,0]); d.qvel[:3]=motion['body_lin_vel_w'][0,0]; d.qvel[3:6]=motion['body_ang_vel_w'][0,0]
    for k,n in enumerate(BFS): d.qpos[qa[n]]=motion['joint_pos'][0,k]; d.qvel[da[n]]=motion['joint_vel'][0,k]
    mujoco.mj_forward(m,d); last=np.zeros(22,np.float32); errors=[]; heights=[]; fall=None
    for step in range(steps):
        fi=min(step//DEC,nframes-1); q=np.array([d.qpos[qa[n]] for n in BFS]); dq=np.array([d.qvel[da[n]] for n in BFS]); o=obs(q,dq,d.qpos[3:7],d.qvel[3:6],motion['joint_pos'][fi],motion['joint_vel'][fi],motion['body_quat_w'][fi,0],last); act=sess.run(['actions'],{'obs':o[None,:],'time_step':np.array([[fi]],np.float32)})[0][0].astype(np.float32); last=act.copy(); target=act*scale+defaults
        for _ in range(DEC):
            cq=np.array([d.qpos[qa[n]] for n in BFS]); cdq=np.array([d.qvel[da[n]] for n in BFS]); tau=np.array([gains(n)[0]*(target[k]-cq[k])-gains(n)[1]*cdq[k] for k,n in enumerate(BFS)]); tau=np.clip(tau,[-20 if n.endswith(ARM) else -30 for n in BFS],[20 if n.endswith(ARM) else 30 for n in BFS]); d.qfrc_applied[:]=0
            for k,n in enumerate(BFS): d.qfrc_applied[da[n]]=tau[k]
            mujoco.mj_step(m,d)
        nxt=min(fi+1,nframes-1); q_now=np.array([d.qpos[qa[n]] for n in BFS]); errors.append(float(np.mean(np.abs(q_now-motion['joint_pos'][nxt])))); heights.append(float(d.qpos[2]));
        if fall is None and d.qpos[2]<FALL_Z: fall=(step+1)*0.02
    return {'motion':motion_name,'policy':policy_name,'duration':duration,'joint_mae':float(np.mean(errors)),'max_joint_mae':float(np.max(errors)),'first_fall_s':fall,'min_height':float(np.min(heights)),'survival_fraction':1.0 if fall is None else fall/duration,'steps':steps,'protocol':'frame0_motion_hold','control_note':'MuJoCo applied-torque PD; Isaac counterpart uses native PhysX implicit PD'}


def main():
    out=[]
    for motion,(path,dr,base) in MOTIONS.items():
        for policy,p in [('baseline',base),('dr',dr)]: print('RUN',motion,policy,flush=True); out.append(run(motion,policy,p))
    p=ROOT/'mujoco_hold_results.json'; p.write_text(json.dumps({'xml':str(XML),'dt':DT,'decimation':DEC,'results':out},indent=2)); print(json.dumps({'out':str(p),'count':len(out)}))


if __name__=='__main__': main()
