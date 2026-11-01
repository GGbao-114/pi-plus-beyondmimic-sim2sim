# 01 · Pipeline: environment, training, export, replay

Everything here was executed on a shared Ubuntu 22.04 box (4 × RTX 4090, 96 cores, 503 GB RAM).
The point of this document is that the numeric results in the README are reproducible from a
described state, not from "it looked stable".

---

## 1. Environment

| Component | Version |
|---|---|
| Isaac Sim | 5.0.0 |
| Isaac Lab | 2.2.0 |
| rsl_rl | 2.3.1 |
| Python | 3.11 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| MuJoCo | 3.14 |
| ONNX / ONNX Runtime | 1.30 |

### Four packaging traps (each cost me a day)

**1. Isaac Sim version strings have four digits.**
`pip install isaacsim==5.0.0` resolves to nothing useful; the correct pin is `5.0.0.0`.

**2. `pypi.nvidia.com` must be the *primary* index, not an extra.**
With a mirror as `--index-url`, pip cannot resolve NVIDIA's private packages:

```bash
--index-url https://pypi.nvidia.com \
--extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**3. Extension packages are *optional extras* — the most expensive trap.**
A plain `pip install isaacsim==5.0.0.0` installs a 164 KB shell with no extensions, because
`isaacsim-extscache-*` / `isaacsim-app` are all gated behind `extra == "extscache"` / `extra == "all"`.
The result is a missing `isaacsim.simulation_app` at import time. Correct form:

```bash
pip install "isaacsim[extscache,all]==5.0.0.0" \
  --index-url https://pypi.nvidia.com \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Verification that the install is real: environment size ≈ 17 GB (a healthy install is 15–18 GB),
`exts/` contains 92 extensions, startup ≈ 17 s.

**4. CMake 4 breaks `cmake_minimum_required(<3.5)`** (`egl_probe` fails to configure):
`conda install -c conda-forge "cmake=3.31"`.

Also: `flatdict` needs `setuptools==69.5.1` and `--no-build-isolation`, and all caches must be
redirected (`PIP_CACHE_DIR`, `CONDA_PKGS_DIRS`, `CONDA_ENVS_PATH`, `XDG_CACHE_HOME`, `HF_HOME`,
`TORCH_HOME`) because the system disks had < 30 GB free.

---

## 2. Training

| Setting | Value |
|---|---|
| Task | `Tracking-Flat-PI-Plus-Wo-v0` |
| Parallel envs | 4096 |
| Control rate | 50 Hz (`sim.dt = 0.005`, `decimation = 4`) |
| Network | actor-critic `[512, 256, 128]` |
| PPO | lr 1e-3, γ 0.99, λ 0.95 |
| Time per iteration | 3.06 s |
| Iterations | 10 000 per motion |

```bash
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-PI-Plus-Wo-v0 \
  --motion_file source/motion/hightorque/pi_plus/npz/pi_plus_kungfu.npz \
  --num_envs 4096 --headless --max_iterations 10000
```

Motion set: `goal`, `kick1015`, `bigkick`, `kungfu`, `dance`
(the very first run also produced a 20 000-iteration `dance_long` variant).

**Convergence evidence (kungfu, the representative case):**

| metric | value |
|---|---|
| final reward | 39.8 (last 40 iterations flat at 38–40, no upward trend) |
| `Mean episode length` | 482 / 496 / 493 / 496 / 481 (cap 500 = 10 s) |
| `error_body_pos` | 0.09 → 0.026–0.033 m |

Reward and training curves: [`../assets/training_rewards.png`](../assets/training_rewards.png),
[`../assets/training_progress.gif`](../assets/training_progress.gif).

---

## 3. Export to ONNX (+ the metadata that was silently empty)

```bash
python scripts/rsl_rl/play.py \
  --task=Tracking-Flat-PI-Plus-Wo-v0 \
  --checkpoint logs/rsl_rl/pi_plus_flat/<run>/model_9999.pt \
  --num_envs=1 \
  --motion_file source/motion/hightorque/pi_plus/npz/pi_plus_kungfu.npz \
  --headless
```

Output: `model_9999.onnx` (24 KB) + `model_9999.onnx.data` (1.4 MB external weights).

`play.py` calls `attach_onnx_metadata`, but the exported file had **0 metadata entries**. Root cause
was never fully isolated (suspected to be how `onnx.save` handles external weight files), so I wrote a
patch that writes the metadata directly (`scripts/add_metadata.py`):

- 22 joint names **in BFS order** (see [`04_pitfalls.md`](04_pitfalls.md)),
- 13 tracked body names + the npz→body index map,
- default joint positions, PD gains, action scale, control dt,
- the reference motion output (so a replayer can cross-check the model bit-for-bit).

Validation of the export: ONNX actor vs checkpoint max deviation **< 4e-6**; embedded reference output
identical to the source npz; 10 metadata entries present.

---

## 4. Replay in both engines

The two engines must start from an *identical* state or the comparison means nothing. The harness
therefore:

1. loads the ONNX policy and the same reference motion,
2. sets both simulators to the reference frame-0 pose (or a dumped Isaac state),
3. steps with the same control period (50 Hz), no noise, no reset,
4. logs `q`, `qd`, torque, foot contact state/force per control step into `.npz` + `.json`.

Key scripts: `sim2sim_headless.py` (MuJoCo, headless, emits GIF + metrics), `isaac_hold_compare.py`
(Isaac counterpart, native PhysX timestep and implicit actuators), `dump_isaac_state.py` +
`compare_obs.py` (observation-by-observation diff).

---

## 5. Why the observation diff is the strongest single piece of evidence

Before blaming physics, I proved the two sides compute the *same numbers*. Dumping the real state
from Isaac, feeding it to the MuJoCo observation builder, and diffing per component:

| component | max observed diff | configured observation noise | verdict |
|---|---:|---|---|
| command | 0.000000 | — | ✅ |
| motion_ori | 0.043345 | ±0.05 | ✅ inside |
| base_ang_vel | 0.182446 | ±0.2 | ✅ inside |
| joint_pos_rel | 0.009583 | ±0.01 | ✅ inside |
| joint_vel | 0.481287 | ±0.5 | ✅ inside |
| last_action | 0.000000 | — | ✅ |

Every difference falls **exactly inside the noise range the training config declares**. The
observations agree; whatever diverges afterwards is genuine dynamics, not a plumbing bug.

Nine independent checks were run in total: joint order, body index, gyro frame, observation layout,
observation values, forward kinematics, joint axes, total mass, torque limits.
