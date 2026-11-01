# 04 · Four silent mismatches, and how each was proved

Every one of these breaks cross-engine comparison **without raising an error anywhere**. They are the
reason the first month of "physics analysis" was measuring my own plumbing.

General principle learned: in cross-engine work, a difference that is large and *systematic* is almost
never physics — physics differences are small, motion-dependent, and noisy.

---

## 1. PhysX articulation joint order is BFS, not URDF declaration order

**Symptom:** the ONNX policy is fed the wrong joint angles in MuJoCo; the robot collapses instantly.

**Cause:** `pi_plus_20dof.urdf` and `pi_20dof.xml` declare joints grouped by limb —

```
left leg (6) → right leg (6) → left arm (5) → right arm (5)
```

— but Isaac/PhysX exposes an articulation's joints in **breadth-first tree order**:

```
[0]  l_hip_pitch          [1]  l_shoulder_pitch
[2]  r_hip_pitch          [3]  r_shoulder_pitch
[4]  l_hip_roll           [5]  l_shoulder_roll
[6]  r_hip_roll           [7]  r_shoulder_roll
[8]  l_thigh              [9]  l_upper_arm
[10] r_thigh              [11] r_upper_arm
[12] l_calf               [13] l_elbow
[14] r_calf               [15] r_elbow
[16] l_ankle_pitch        [17] l_wrist
[18] r_ankle_pitch        [19] r_wrist
[20] l_ankle_roll         [21] r_ankle_roll
```

**How it was proved** (forward kinematics against reference body poses, `scripts/judge_joint_order.py`):

```
assume CSV order  → mean error 0.0914 m
assume BFS order  → mean error 0.0002 m   ← 450× smaller
```

Independently confirmed by dumping the articulation joint names directly from Isaac:

```
['l_hip_pitch_joint', 'l_shoulder_pitch_joint', 'r_hip_pitch_joint', ...]
```

**Why it is dangerous:** the training pipeline is fully self-consistent under either ordering, so
nothing surfaces it. It only appears the moment a second engine reads the file.

---

## 2. Body-index mapping between the motion file and the tracked subset

**Symptom:** subtler — the policy gets slightly wrong reference body poses and behaves "almost right",
which is far more expensive to debug than a collapse.

**Cause:** the motion `.npz` carries `body_pos_w` for **23 bodies** (all of them), while training only
tracks the **13** listed in `cfg.body_names`. The index map is not the identity.

**How it was proved:** matched against the ONNX policy's own embedded reference data; residual
`0.000000`.

```python
NPZ_IDX = [0, 5, 13, 21, 7, 15, 22, 6, 14, 18, 8, 16, 20]
```

---

## 3. Wrong `base_link` inertia → 18.755 kg instead of 11.0002 kg

**Symptom:** none, during training. A 70 % mass error is completely invisible there, because the policy
simply adapts to whatever mass it is handed.

**Cause:** the URDF `base_link` inertial block was invalid.

**Impact:** this is the single most important correction in the project, because every cross-engine
force comparison before it was meaningless. After correction, URDF (11.0002 kg) and MJCF (11.18 kg)
agree to **1.6 %** — good enough that the residual gap can be attributed to contact rather than mass.

**Lesson:** validate total mass and per-link inertia against the real robot *before* any sim-to-sim
or sim-to-real analysis. Cheap to check, fatal to skip.

---

## 4. Torque limits configured but never enabled

**Symptom:** the repaired policy reached 45 N·m while the config claimed 30 N·m; a whole class of
"the policy is too aggressive" explanations became available that were actually a config bug.

**Cause:** the auto-generated MJCF had every actuator/joint force-range flag set to `false`. The
existing script edited the numeric ranges and never flipped the switch, so the limits were decorative.

**Fix:** real, active limits — legs 30 N·m, arms 20 N·m — then re-run all comparisons.

---

## Bonus: two export-level defects

**5. ONNX metadata silently empty.** `play.py` calls `attach_onnx_metadata`, and the exported model had
zero metadata entries. Root cause not fully isolated (suspected external-weights behaviour in
`onnx.save`); solved with `scripts/add_metadata.py`, which writes joint/body names, gains, dt and the
reference output directly. Verified afterwards: actor vs checkpoint < 4e-6, metadata count 10.

**6. Quaternion → rotation-matrix sign error.** A hand-written conversion flipped the sign, so the
robot interpreted orientation wrongly. Fixed, and a rotation-matrix orthonormality assertion was added
to the comparison tooling so it cannot silently return.

---

## Checklist this produced

Before trusting any sim-to-sim number:

- [ ] total mass and per-link inertia match the real robot
- [ ] joint order established **empirically** (FK, not by reading the file)
- [ ] body index map recovered from the policy's own embedded data
- [ ] torque limits actually active, verified by measured torque
- [ ] quaternion conventions asserted orthonormal
- [ ] observation diff inside the declared noise band, component by component
- [ ] training protocol == evaluation protocol
