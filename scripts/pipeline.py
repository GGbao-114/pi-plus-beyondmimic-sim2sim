import itertools
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path("/data/gaoli/logs/model_align_r23")
PY = "/data/gaoli/conda/envs/isaaclab/bin/python"
sys.path.insert(0, str(ROOT))
import sweep_harness  # noqa: E402

XML = ROOT / "pi_auto_fixedmass.xml"
PROJ = Path("/data/gaoli/projects/Mini-Pi-Plus_BeyondMimic")
MOTION_DIR = PROJ / "source/motion/hightorque/pi_plus/npz"

MOTIONS = {
    "goal": {"state": "/data/gaoli/logs/model_align_r2/isaac_goal_contact/T4_goal_hold.npz",
             "isaac": "/data/gaoli/logs/model_align_r15/isaac_dynamics_step.json",
             "motion": str(MOTION_DIR / "pi_plus_goal.npz"), "need_isaac": False},
    "kick1015": {"state": "/data/gaoli/logs/tail_round3/diag_isaac/A_kick1015_hold.npz",
                 "isaac": "/data/gaoli/logs/model_align_r22/isaac_kick1015/isaac_dynamics_step.json",
                 "motion": str(MOTION_DIR / "kick1015.npz"), "need_isaac": False},
    "bigkick_B": {"state": str(ROOT / "bigkick_B_state.npz"),
                  "isaac": str(ROOT / "isaac_bigkick_B" / "isaac_dynamics_step.json"),
                  "motion": str(MOTION_DIR / "pi_plus_right_big_kick.npz"), "need_isaac": True},
    "bigkick_C": {"state": str(ROOT / "bigkick_C_state.npz"),
                  "isaac": str(ROOT / "isaac_bigkick_C" / "isaac_dynamics_step.json"),
                  "motion": str(MOTION_DIR / "pi_plus_right_big_kick.npz"), "need_isaac": True},
}

SOLREF = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
SOLIMP = {
    "base": [0.99, 0.999, 1e-5, 0.5, 2.0],
    "mid": [0.95, 0.99, 0.001, 0.5, 2.0],
    "soft": [0.9, 0.95, 0.001, 0.5, 2.0],
}
CONDIM = [3, 4, 6]
FRICTION = [1.0, 0.7]
MARGIN = [0.0, 0.001]


def build_grid():
    grid = []
    for sr, (sname, si), cd, fr, mg in itertools.product(SOLREF, SOLIMP.items(), CONDIM, FRICTION, MARGIN):
        name = f"sr{sr}_si{sname}_cd{cd}_fr{fr}_mg{mg}".replace(".", "p")
        grid.append({"variant": name, "params": {
            "solref": [sr, 1.0], "solimp": si, "condim": cd,
            "friction0": fr, "margin": mg, "scope": "all"}})
    return grid


def run_isaac(motion_key):
    info = MOTIONS[motion_key]
    out_dir = ROOT / f"isaac_{motion_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ROOT / f"isaac_{motion_key}.log"
    cmd = [PY, str(ROOT / "isaac_perstep.py"), "--project", str(PROJ),
           "--out", str(out_dir), "--state_npz", info["state"],
           "--motion", info["motion"], "--device", "cuda:1", "--headless"]
    with log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    json_path = out_dir / "isaac_dynamics_step.json"
    return motion_key, rc, json_path.exists() and json_path.stat().st_size > 0


def task(args):
    motion_key, variant, params = args
    spec = {"motion": motion_key, "variant": variant, "state_npz": MOTIONS[motion_key]["state"],
            "xml": str(XML), "params": params,
            "out": str(ROOT / "runs" / f"{motion_key}__{variant}.json")}
    try:
        result = sweep_harness.run(spec)
        out = Path(spec["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result))
        return motion_key, variant, spec["out"], None
    except Exception as exc:  # noqa: BLE001
        return motion_key, variant, spec["out"], repr(exc)


def table_from_result(result, isaac_rows):
    out = []
    for ir, vr in zip(isaac_rows, result["rows"]):
        out.append([float(np.max(np.abs(np.asarray(ih["q_after"]) - np.asarray(vh["q_after"]))))
                    for ih, vh in zip(ir["horizons"], vr["horizons"])])
    return out


def classify(table, base):
    r = np.asarray(table) / np.asarray(base) - 1
    c = r[:, 3]
    return {"improved_times": int(np.sum(c < 0)), "max_increase_pct": float(np.max(c) * 100),
            "min_change_pct": float(np.min(c) * 100),
            "consistent": bool(int(np.sum(c < 0)) >= 4 and float(np.max(c)) <= 0.02)}


def main():
    (ROOT / "runs").mkdir(exist_ok=True)
    report_stage = {}

    # Stage 1: Isaac fixed-action references for motions that still need them.
    for key, info in MOTIONS.items():
        if info["need_isaac"] and not Path(info["isaac"]).exists():
            mk, rc, ok = run_isaac(key)
            report_stage[f"isaac_{mk}"] = {"rc": rc, "ok": ok}

    isaac = {}
    for key, info in MOTIONS.items():
        p = Path(info["isaac"])
        isaac[key] = json.loads(p.read_text())["rows"] if p.exists() else None

    # Stage 2: sweep.
    grid = build_grid()
    tasks = [(m, "noop", {"scope": "all"}) for m in MOTIONS]
    tasks += [(m, g["variant"], g["params"]) for m in MOTIONS for g in grid]
    results = {}
    errors = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        futs = {ex.submit(task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            motion_key, variant, out, err = fut.result()
            done += 1
            if err:
                errors.append({"motion": motion_key, "variant": variant, "error": err})
            else:
                results[(motion_key, variant)] = out
            if done % 200 == 0:
                print(f"progress {done}/{len(tasks)} errors={len(errors)}", flush=True)

    # Stage 3: aggregate.
    baseline = {}
    for m in MOTIONS:
        p = results.get((m, "noop"))
        baseline[m] = table_from_result(json.loads(Path(p).read_text()), isaac[m]) if p else None

    variants = {}
    for g in grid:
        v = g["variant"]
        per_motion = {}
        n_consistent = 0
        s350 = []
        for m in MOTIONS:
            p = results.get((m, v))
            if not p:
                per_motion[m] = {"missing": True}
                continue
            t = table_from_result(json.loads(Path(p).read_text()), isaac[m])
            cls = classify(t, baseline[m])
            cls["endpoint_3.5s_50"] = t[3][3]
            per_motion[m] = cls
            if cls["consistent"]:
                n_consistent += 1
            s350.append(t[3][3] / baseline[m][3][3] - 1)
        variants[v] = {"params": g["params"], "n_motions_consistent": n_consistent,
                       "mean_3.5s50_change_pct": float(np.mean(s350) * 100) if s350 else None,
                       "per_motion": per_motion}

    ranked = sorted([(v, d["n_motions_consistent"], d["mean_3.5s50_change_pct"], v) for v, d in variants.items() if d["mean_3.5s50_change_pct"] is not None],
                    key=lambda x: (-x[1], x[2]))
    summary = {"stage": report_stage, "motions": list(MOTIONS), "grid_size": len(grid),
               "runs_ok": len(results), "errors": errors, "baseline": baseline,
               "variants": variants, "ranked": [r[0] for r in ranked[:20]],
               "errors_count": len(errors)}
    (ROOT / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"grid": len(grid), "runs_ok": len(results), "errors": len(errors),
                      "top": ranked[:10]}, indent=2))


if __name__ == "__main__":
    main()
