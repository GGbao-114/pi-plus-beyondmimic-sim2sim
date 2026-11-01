"""Analyze phase-separated traces without treating post-fall data as onset evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def unit_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("zero quaternion")
    return q / norm


def up_vector(q):
    w, x, y, z = unit_quat(q).T
    return np.stack((2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)), axis=-1)


def analyze_trace(path, ref, joint_limits, ref_end):
    with np.load(path) as data:
        trace = {k: data[k].copy() for k in data.files}
    t = trace["time"]
    height = trace["heights"]
    fi = np.minimum(np.rint(t*50).astype(int), len(ref["joint_pos"])-1)
    error = np.abs(trace["joint_pos"]-ref["joint_pos"][fi]).mean(axis=1)
    actual_up = up_vector(trace["base_quat"])
    ref_up = up_vector(ref["body_quat_w"][fi, 0])
    tilt = np.degrees(np.arccos(np.clip(actual_up[:, 2], -1, 1)))
    reference_up_error = np.degrees(np.arccos(np.clip((actual_up*ref_up).sum(axis=1), -1, 1)))
    crossed = np.flatnonzero(height < 0.15)
    fall_time = float(t[crossed[0]]) if len(crossed) else None
    before_fall = t < fall_time if fall_time is not None else np.ones(len(t), dtype=bool)
    in_reference = t <= ref_end + 1e-9
    phases = {
        "reference_prefall": in_reference & before_fall,
        "hold_prefall": ~in_reference & before_fall,
        "after_first_fall": ~before_fall,
    }
    summaries = {}
    for name, mask in phases.items():
        if not mask.any():
            summaries[name] = None
            continue
        req = np.abs(trace["requested_tau"][mask])
        eff = np.abs(trace["effective_tau"][mask])
        summaries[name] = dict(
            samples=int(mask.sum()), start_s=float(t[mask][0]), end_s=float(t[mask][-1]),
            joint_mae=float(error[mask].mean()), height_min=float(height[mask].min()),
            tilt_max_deg=float(tilt[mask].max()), up_direction_error_max_deg=float(reference_up_error[mask].max()),
            requested_tau_saturation_sample_fraction=float((req>=0.98*joint_limits).mean()),
            effective_tau_saturation_sample_fraction=float((eff>=0.98*joint_limits).mean()),
            effective_tau_max=float(eff.max()), joint_speed_max=float(np.abs(trace["joint_vel"][mask]).max()),
            base_omega_max=float(np.linalg.norm(trace["base_omega"][mask],axis=1).max()),
            no_contacts_sample_fraction=float((trace["ncon"][mask]==0).mean()),
            contact_force_norm_peak=float(trace["contact_force_norm_max"][mask].max()),
        )
    end_indices=np.flatnonzero(t<=ref_end+1e-9)
    end_index=int(end_indices[-1]) if len(end_indices) and t[-1]>=ref_end-1e-9 else None
    row=dict(trace=path.name, reference_end_s=ref_end, sim_end_s=float(t[-1]), first_fall_s=fall_time,
             reference_end_reached=end_index is not None, phases=summaries)
    if end_index is not None:
        row["at_reference_end"]=dict(time_s=float(t[end_index]),height=float(height[end_index]),
            tilt_deg=float(tilt[end_index]),up_direction_error_deg=float(reference_up_error[end_index]),
            base_omega=float(np.linalg.norm(trace["base_omega"][end_index])),contacts=int(trace["ncon"][end_index]))
    points=[]
    if fall_time is not None:
        for at in [ref_end, ref_end+0.2, ref_end+0.4, fall_time-0.4, fall_time-0.2, fall_time]:
            idx=int(np.argmin(np.abs(t-at)))
            points.append(dict(time_s=float(t[idx]),height=float(height[idx]),tilt_deg=float(tilt[idx]),
                               up_error_deg=float(reference_up_error[idx]),joint_mae=float(error[idx]),
                               contacts=int(trace["ncon"][idx]),base_omega=float(np.linalg.norm(trace["base_omega"][idx]))))
    row["endgame_samples"]=points
    return row


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=here/"sim2sim复测_20261007/results")
    parser.add_argument("--motions",type=Path,default=here/"Mini-Pi-Plus_BeyondMimic/source/motion/hightorque/pi_plus/npz")
    args=parser.parse_args()
    manifest=json.loads((args.results/"manifest.json").read_text())
    xml_info=json.loads((args.results/"model_info.json").read_text())
    # The XML's unit-geared actuator order is audited from the original model.
    import xml.etree.ElementTree as ET
    xml_path=args.results.parent/"pi_auto.xml"
    if not xml_path.exists():
        raise FileNotFoundError("copy the evaluated pi_auto.xml beside results before analysis")
    xml=ET.parse(xml_path).getroot()
    names=[node.get("joint") for node in xml.find("actuator")]
    arm_suffix=("_shoulder_pitch_joint","_shoulder_roll_joint","_upper_arm_joint","_elbow_joint","_wrist_joint")
    limits=np.array([20.0 if name.endswith(arm_suffix) else 30.0 for name in names])
    assert len(names)==xml_info["nu"]
    rows=[]
    for entry in manifest:
        motion=args.motions/Path(entry["motion"]).name
        if hashlib.sha256(motion.read_bytes()).hexdigest()!=entry["motion_sha256"]:
            raise ValueError(f"motion hash mismatch: {motion}")
        with np.load(motion) as data:
            ref={k:data[k].copy() for k in ("joint_pos","joint_vel","body_quat_w")}
        end=(len(ref["joint_pos"])-1)/50
        variants=["updated_raw_gyro","updated_15s"]
        if entry["run"]=="2026-10-03_23-12-12":
            variants.append("complete_dance_plus_hold")
        for variant in variants:
            row=analyze_trace(args.results/f'{entry["run"]}_{variant}.npz',ref,limits,end)
            row.update(run=entry["run"],motion=motion.name,variant=variant,
                       terminal_reference_velocity_max=float(np.abs(ref["joint_vel"][-1]).max()))
            rows.append(row)
            print(json.dumps(row),flush=True)
    (args.results/"endgame_analysis_corrected.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print("CORRECTED_ANALYSIS_DONE")


if __name__=="__main__":
    main()
