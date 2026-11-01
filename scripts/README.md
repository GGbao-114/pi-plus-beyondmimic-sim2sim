# Scripts index

These are the tools I wrote for this project, published as-run. Each script's docstring states the
question it answers; several also state which **evidence class** the result belongs to, because
conclusions from the two classes are not interchangeable:

- **`[dynamics comparison]`** — identical initial state, identical action, compare state after real
  solver steps. Answers "do the two engines integrate the same dynamics?"
- **`[trajectory replay]`** — run the closed-loop policy against a reference motion. Answers "does the
  robot survive the task?"

A parameter can look irrelevant under one class and decisive under the other. Mixing them silently is
how I wasted a month.

Paths inside the scripts point at `/data/gaoli/logs/...` from the original machine; adapt before use.

---

## Establish ground truth before comparing

| script | question it answers |
|---|---|
| `judge_joint_order.py` | Is the npz joint order the CSV/URDF order or the PhysX BFS order? Decided by FK error against reference body poses (0.0914 m vs 0.0002 m). Needs no Isaac. |
| `find_body_order.py` | What is the real body order in `npz body_pos_w`? Poses MuJoCo at frame 0, then matches each npz body index to the nearest MuJoCo body. |
| `check_kinematics.py` | Does MJCF forward kinematics agree with the URDF-generated motion data? Catches joint-axis / link-offset mismatches, the most common and most hidden sim2sim failure. |
| `final_diag.py` | Three-in-one: MJCF-vs-URDF inertia tensors (after rotation), joint-axis vectors, static stance test. |
| `check_gyro.py` | Which frame does the MuJoCo `gyro` sensor report in? Compares `mj_objectVelocity` world vs local against the sensor reading — whichever matches is the answer. |
| `audit_formulas.py` | AST-level audit of the conversion formulas in the analysis code (hashing + `ast`), to make sure a fixed formula stays fixed. |
| `inspect_ckpt.py` | Dump what is actually inside a `.pt` checkpoint. |

## Export and verify the policy

| script | question it answers |
|---|---|
| `add_metadata.py` | Patches the missing ONNX metadata: joint names **in BFS order**, body names, npz→body index map, gains, dt, embedded reference output. |
| `validate_onnx_in_isaac.py` | Is the exported ONNX actually equivalent to the PyTorch policy? `play.py` runs the `.pt`; sim2sim runs the `.onnx` — the export step was never validated before this. |
| `dump_isaac_state.py` | Export the real Isaac state + observations without forcing state writes (avoids the physics cache trap). |
| `compare_obs.py` | Feed the dumped Isaac state into MuJoCo's observation builder and diff every component against the configured noise band. |

## Replay and compare

| script | question it answers |
|---|---|
| `sim2sim_headless.py` | Headless MuJoCo replay of the ONNX policy; reads joint order from the model instead of hard-coding it, and reports torque saturation. Emits GIF + metrics. |
| `isaac_hold_compare.py` | The Isaac counterpart of the frame-zero, zero-velocity hold test, using the original implicit actuators and native PhysX timestep. |
| `mujoco_hold_r26.py` | MuJoCo side of the same hold test, matched to the Isaac run. |
| `isaac_contact_api_probe.py` | What per-contact data does Isaac actually expose? Written to establish the R2/R3 evidence boundary honestly. |
| `mj_contact_diag_base.py` | MuJoCo contact diagnostics on the matched scenario (contact count, normal force, penetration). |
| `matched_control_r30.py` | Are the explicit-PD MJCF and the position-actuator MJCF equivalent under matched state and matched action? (R1 in code.) |
| `static_alignment_legs.py` | Static leg-pose alignment check between the two models. |

## Build the models

| script | question it answers |
|---|---|
| `auto_convert.py` | Generate an MJCF from the URDF, discovering `urdf2mjcf`'s actual API instead of assuming it. |
| `auto_sim2sim.py` | URDF → MJCF → sim2sim end to end with fault tolerance for the traps actually hit (empty URDF material names, relative mesh paths, etc.). |
| `sweep_mjcf.py` | Parameter ablation: change exactly one MJCF item, run a fixed duration, record time-to-fall. |
| `ablate_fixed.py` | Re-run the ablation on the *repaired* MJCF — the earlier ablation was run on a broken model (no armature), so its conclusions were void. |
| `ablate_dt.py` | Timestep refinement 0.002 → 0.0005 to find where the result converges (0.001 already behaves very differently from 0.002). |
| `pipeline.py` | Grid runner: enumerates parameter combinations and dispatches runs in parallel processes. |

## Repair the motion and analyse the outcome

| script | question it answers |
|---|---|
| `round2_distill_wrapper.py` | Resume PPO with an added old-policy action-distillation term restricted to the original frames. This is the step that restores fidelity. |
| `analyze_endgame.py` | Phase-separated trace analysis that refuses to treat post-fall samples as onset evidence. |
| `analyze_isaac_hold_compare.py` | Validate and summarise matched, no-reset Isaac/MuJoCo traces. |
| `make_A_vs_D.py` | Build the side-by-side "original vs repaired" GIF for the report. |

---

## Note on code quality

These are research scripts, not a packaged library: absolute paths, hard-coded run directories, and
per-round copies (`r26`, `r30`, `r31` suffixes) are deliberate — each round had to stay byte-identical
to what produced its report, so a conclusion could always be re-derived. Consolidating them into a
config-driven CLI is the obvious next refactor, not something I pretended to have already done.
