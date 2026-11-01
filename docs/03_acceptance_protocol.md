# 03 · Motion tail repair and the dual-engine acceptance gate

**The bug:** `bigkick`, `goal`, `kick1015` fell over after 2–6 s. The standing hypothesis was
"PhysX and MuJoCo contact solvers are fundamentally different". It was wrong, and it took five
findings to disprove it.

---

## Findings, in the order they forced themselves on me

**1. The MJCF torque limit was configured but never enabled.**
Every actuator/joint force-range flag in the auto-generated MJCF was `false`; the old script edited
the numeric range without flipping the switch. Measured post-segment torque reached **45 N·m against a
30 N·m setpoint**. Fixed to real, active limits: legs 30 N·m / arms 20 N·m.

**2. The "failure" happened *after* the motion segment, not during it.**
Per-control-step analysis: all three motions track the reference well inside the segment
(MAE 0.06–0.13 rad) and fall **0.7–1.0 s after the reference ends**.

**3. Isaac fails too — the same way, slightly later.**
Retested in Isaac under the identical protocol (frozen final frame, zero resets, zero noise):

| motion | MuJoCo fall | Isaac fall |
|---|---:|---:|
| `bigkick` | 2.78 s | 3.30 s |
| `goal` | 5.56 s | 6.08 s |
| `kick1015` | 2.64 s | 3.02 s |

Both engines fall. The earlier "Isaac is stable for 9.8 s" was an artefact: it evaluated `kungfu`, a
12 s motion that never reached its end, in an environment that auto-resets.

**The root cause was a protocol mismatch inside the training env:**
`commands.py::_resample_command` teleports to a *new* motion the instant the reference ends, so the
policy was never trained on "freeze the last frame and hold" — and that is exactly what the evaluator
demanded. **Evaluation protocol ≠ training protocol.**

**4. Appending a tail without retraining makes things *worse*.**
Given a new tail reference (transition to default stance), the original policy cannot track it
(transition error 0.26+ rad) and falls *earlier* than with a frozen frame. **The ability to settle must
be trained, not spliced on.**

**5. Naive fine-tuning buys stability by destroying arm accuracy.**
2 000 iterations of fine-tuning: stable for 8 s, zero resets, 3.54 s hold — but original-segment MAE
regressed by 14.6 % (MuJoCo) / 21.8 % (Isaac), exceeding the 10 % gate. Per-joint decomposition:
**the entire regression is in the arms**; legs net ≈ 0, and the kicking leg actually got more accurate.
That is a structural trade caused by the changed sampling distribution, and it is what motivated
distillation.

---

## The three-step repair

### Step 1 — Tail data generation

- From the motion's final frame, a **1.5 s quintic polynomial** transition to the default stance,
  then **1.2 s hold**.
- Body trajectories still produced by the Isaac FK conversion chain (50 → 50 Hz passthrough); the foot
  side cross-checked with MuJoCo's own FK.
- Seam quality: the first 89 frames match the original data to < 2.4e-7 in position and orientation,
  and velocity is continuous across the seam.

### Step 2 — Staged retraining (5 000 iterations, resumed from the original checkpoint)

| stage | content | iterations | purpose |
|---|---|---:|---|
| T1 | train on the tail motion only | 2 000 | learn to stand |
| T2 | original segment re-weighted, p = 0.5 | 1 000 | recover the original distribution |
| T3 | original segment re-weighted, p = 0.7 | 1 000 | rebalance across engines |
| T4 | **distillation** from the original policy (teacher = original, λ = 0.1, original 89 frames only) | 1 000 | **restore fidelity** |

Why distillation is the load-bearing step: T1–T3 fix stability and damage the arms. Distilling the
original policy back on the original frames pins the behaviour that the fine-tuning was eroding, so
stability and fidelity stop trading against each other.

### Step 3 — Acceptance

Gate, applied **per engine**:

- 8 s without falling,
- ≥ 2 s hold after the reference ends,
- **0 state resets**,
- original-segment MAE ≤ 1.10 × baseline.

---

## Acceptance results

### `bigkick` (round 2, `D_distill_clean`)

| engine | original-segment MAE | threshold | post-segment hold | resets | verdict |
|---|---:|---:|---:|---:|---|
| MuJoCo | 0.0607 | 0.0736 | 3.54 s | 0 | ✅ |
| Isaac | 0.0528 | 0.0619 | 3.54 s | 0 | ✅ |

### `kick1015` (round 3, `T4_distill`)

| engine | original-segment MAE | threshold | post-segment hold | resets | verdict |
|---|---:|---:|---:|---:|---|
| MuJoCo | 0.0658 | 0.0665 | 3.54 s | 0 | ✅ |
| Isaac | 0.0530 | 0.0534 | 3.54 s | 0 | ✅ |

> **Honest caveat:** `kick1015` passes with a very thin margin (≈1 % on MuJoCo, ≈0.7 % on Isaac). It is
> a pass, but not a comfortable one; the next iteration would compress the MAE further or tune λ.

Export chain validated end to end for both: ONNX actor vs checkpoint < 4e-6, reference output
identical to the npz, 10 metadata entries.

---

## Visual evidence

| file | content |
|---|---|
| [`../assets/A_vs_D_bigkick.gif`](../assets/A_vs_D_bigkick.gif) | left: original policy (falls at 2.78 s) · right: repaired (8 s) |
| [`../assets/A_vs_T4_kick1015.gif`](../assets/A_vs_T4_kick1015.gif) | same comparison for `kick1015` |
| [`../assets/round4_analysis.png`](../assets/round4_analysis.png) | 4-panel: error curves / per-joint regression / per-limb-per-phase / training curves |

---

## The lesson I would keep

> A MuJoCo fall is not evidence about MuJoCo. Confirm that training and evaluation ask the robot for
> the same thing before attributing anything to the simulator.

This is also why the earlier ablation table in
[`02_cross_engine_diagnosis.md`](02_cross_engine_diagnosis.md) shows **no** parameter having any
effect: the failure was outside the reference segment, where no dynamics parameter can act.
