# 02 · Cross-engine diagnosis: R1–R5

**Question this document answers:** after every plumbing mismatch is fixed, *why* do the same ONNX
policy and the same MJCF/URDF-driven state produce different joint trajectories in Isaac Sim (PhysX)
and MuJoCo?

**Answer, in one line:** the difference is in the foot-contact solving mechanism, not in any parameter
a user can tune — and the sweep proves the optimum is motion-dependent.

An important framing note first, because it cost me the most time: **most of what initially looked
like an engine difference was not one.** See [`03_acceptance_protocol.md`](03_acceptance_protocol.md)
for the evaluation-protocol bug that masqueraded as an engine difference, and
[`04_pitfalls.md`](04_pitfalls.md) for the four index/model bugs corrected before this analysis began.

---

## Baseline

`q gap` = max |Δq| between the two engines over a matched, zero-noise, zero-reset replay of the same
policy and the same reference motion.

| motion | baseline mean q gap |
|---|---:|
| `goal` | 0.035192 |
| `kick1015` | (motion-dependent; see R5 table) |

The specific numbers in R5 come from a *fixed test scene and parameter grid*, which is a slightly
different statistical basis from the general table above. Compare **before/after within each
experiment**; do not mix the two tables as if they were one metric.

---

## R1 — Controller implementation

**Hypothesis:** MuJoCo's explicit PD loop differs from PhysX's `position` actuator.
**Method:** drive the same joint targets through MuJoCo's explicit PD and through a `position`
actuator; compare.
**Result:** max joint difference `1.86e-12 rad`.
**Verdict:** numerically identical. Not the cause.

---

## R2 — Collision geometry

**Hypothesis:** the compiled mesh, the source STL and the convex hull used by each engine differ in
shape or placement.
**Method:** compare AABBs and closest-point queries across compiled mesh / raw STL / convex hull.
**Result:** AABB deviation `1.32e-9 m`; closest-point deviation `4.02e-9 m`.
**Verdict:** negligible. Ankle-roll mesh-vs-box substitution was tested separately (R23) and made the
divergence **worse** (+133.7 %), which is evidence that foot geometry is *sensitive* — but it does not
show that the original mesh explains the gap.

**Evidence boundary:** PhysX's runtime convex-hull vertices were never exported, so this comparison is
between the *declared* geometries, not the runtime ones.

---

## R3 — Contact timing

**Hypothesis:** the two engines disagree about *when* the foot touches the ground.
**Method:** Isaac foot `ContactSensor` state vs a MuJoCo foot proxy quantity.
**Result:** on several motions the contact onset instant and the contact duration in frames differ.
**Verdict:** real difference — but see the boundary.

**Evidence boundary:** the two data sources are not the same physical quantity. This is a per-foot
*proxy* comparison, not a one-to-one measurement of native contact points. At the time, Isaac
exposed only per-foot `ContactSensor` aggregates, not PhysX-native per-contact reports. Treat the
firing times as indicative, not as ground truth on contact point identity.

---

## R4 — Contact force magnitude

**Hypothesis:** the contact timing can be near-identical while the *force* is not.
**Method:** at matched timestamps, compare contact frame counts and normal-force peaks for `goal` and
`kick1015`.

**Result — this is the sharpest single number in the whole project:**

| moment | Isaac | MuJoCo |
|---|---:|---:|
| `kick1015`, t ≈ 3 s | ≈ 1112 N | ≈ 387 N |

For `goal`, some timestamps agree on contact-frame count while MuJoCo reports a higher peak force.
Other timestamps reverse the trend.

**Verdict:** the disagreement is real and **motion-dependent in sign** — there is no single scalar
offset to correct.

---

## R5 — Is there a parameter set that fixes both trajectories and contacts?

**Hypothesis:** 108 combinations of `solref`, `solimp`, `condim`, friction and `margin`, across two
motions, will contain at least one combination that is good for both.

**Result: there is no universal optimum.**

| motion | baseline mean q gap | best trajectory combo | improvement | character of the winner |
|---|---:|---:|---:|---|
| `goal` | 0.035192 | 0.022163 | −37.0 % | lower contact-force error, but trajectory error does **not** drop the same way |
| `kick1015` | — | — | — | trajectory-optimal and contact-optimal combos **do not coincide** |

The winners are not even close to each other:

| motion | trajectory-optimal candidate |
|---|---|
| `goal` | `solref = 0.02`, `condim = 4`, friction ≈ 1.0 |
| `kick1015` | `solref = 0.005`, `condim = 6`, friction ≈ 0.7 |

**Interpretation.** A single robot has one physical foot. If two motions demand opposite contact
stiffness and dimensionality to match a *different* simulator, the parameter is compensating for a
solver-mechanism difference rather than describing the robot. Fitting per-motion is overfitting the
metric.

**Verdict:** general-purpose contact tuning is saturated. Per-motion candidates are retained and
labelled with their applicable motion and metric; they are **not** presented as robot parameters.

---

## Ablations that ruled out the remaining "easy" explanations

Single-change MuJoCo ablations, recording time-to-fall:

| configuration | time to fall |
|---|---:|
| ① baseline | 0.82 s |
| ③ contact stiffness `solref` 0.02 → 0.002 | 0.84 s |
| ④ friction 1.0 → 1.5 | 0.82 s |
| ⑤ torque limit 20 → 30 | 0.82 s |
| ⑥ remove joint dry friction (`frictionloss`) | 0.82 s |
| ⑦ disable IMU low-pass filter | 0.82 s |
| ⑧ all of the above together | 0.82 s |

No physical parameter change moved the outcome — which, at the time, was read as "engine-level
difference". The correct reading (found later) is that the failure was **not in the reference
segment at all**, so no dynamics parameter could have fixed it.

Single-factor isolation on the matched-state step comparison (R23/R25):

| variant | 3.5 s / 50-step max \|Δq\| | vs baseline | conclusion |
|---|---:|---:|---|
| baseline | 0.048892 | 0 | — |
| MuJoCo implicit integrator | 0.047668 | −2.5 % | small improvement only |
| passive `damping = 0` | 0.048892 | 0 % | no visible effect |
| ankle-roll mesh → box | 0.114244 | +133.7 % | clearly worse |

Integrator and passive damping are **not** the main cause. Contact geometry is highly sensitive, but
the correct match has not been found.

---

## Static-stance control: is the model itself broken?

Locking all joints at the default stance and letting physics run, without the policy:

| time | base height | max \|τ\| |
|---|---:|---:|
| 0.00 s | 0.3511 m | 0.00 N·m |
| 0.50 s | 0.3596 m | 1.30 N·m |
| 1.00 s | 0.3597 m | 1.27 N·m |
| … | … | … |
| 5.50 s | 0.3597 m | 1.22 N·m |

Six seconds, no fall, no torque saturation. **The MJCF model plus PD gains can hold the robot up** —
there is no fundamental model defect. This control is what let me stop suspecting the model and start
suspecting the task definition.

---

## Final attribution

| aspect | Isaac Sim (PhysX) | MuJoCo |
|---|---|---|
| contact solve | GPU position/velocity iterations (8/4) | convex optimisation + soft constraints |
| actuator | implicit PD (solved inside the solver) | explicit PD |
| integrator | implicit | semi-implicit Euler |
| timestep | 0.005 s | 0.002 s |
| joint dry friction | not supported | supported (MJCF has 0.2) |

**What is established:** the gap is in the contact mechanism; it is motion-dependent; no tunable
parameter set removes it.

**What is *not* established:** a contact-point-level proof, and whether this gap actually matters on
real hardware. Both require either PhysX-native per-contact reporting or real-robot data — which is
where this project deliberately stops and hands over.
