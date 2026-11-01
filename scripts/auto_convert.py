#!/usr/bin/env python
"""自动探测并调用 urdf2mjcf，生成 MJCF。

urdf2mjcf 的 API 名称不确定，本脚本会:
  1. 找 console script（~/.local/bin 等）
  2. 探测包的公开函数与签名
  3. 逐个尝试调用
  4. 报告成功/失败

用法:
  python auto_convert.py --urdf <urdf> --out <mjcf>
"""
import argparse
import glob
import importlib
import inspect
import os
import pkgutil
import subprocess
import sys

import numpy as np  # noqa


def find_console_script():
    cands = []
    for d in ["~/.local/bin", "/opt/anaconda3/bin",
              "/data/gaoli/conda/envs/isaaclab/bin", "/usr/local/bin"]:
        d = os.path.expanduser(d)
        for f in glob.glob(os.path.join(d, "*urdf*")):
            if os.path.isfile(f) and os.access(f, os.X_OK):
                cands.append(f)
    return cands


def probe_package():
    try:
        pkg = importlib.import_module("urdf2mjcf")
    except Exception as e:
        print(f"[!] 无法导入 urdf2mjcf: {e}")
        return None, []
    print(f"[OK] 包位置: {getattr(pkg, '__file__', '?')}")
    names = [n for n in dir(pkg) if not n.startswith("_")]
    print(f"[i] 公开名称: {names}")
    callables = []
    for n in names:
        obj = getattr(pkg, n)
        if callable(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(?)"
            print(f"    {n}{sig}")
            callables.append((n, obj))
    subs = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    print(f"[i] 子模块: {subs}")
    return pkg, callables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    urdf = os.path.abspath(args.urdf)
    out = os.path.abspath(args.out)
    print("=" * 78)
    print(" urdf2mjcf 自动探测")
    print("=" * 78)
    print(f"  URDF: {urdf}")
    print(f"  输出: {out}")
    print(f"  URDF 存在: {os.path.isfile(urdf)}")

    # ---------- 1. console script ----------
    scripts = find_console_script()
    print(f"\n[1] 找到的可执行文件: {scripts}")
    for s in scripts:
        for form in ([s, urdf, out], [s, urdf, "-o", out], [s, urdf]):
            print(f"    尝试: {' '.join(form)}")
            try:
                r = subprocess.run(form, capture_output=True, text=True, timeout=300)
                guess = os.path.splitext(urdf)[0] + ".mjcf"
                got = None
                if os.path.isfile(out):
                    got = out
                elif os.path.isfile(guess):
                    got = guess
                if got:
                    print(f"    ✅ 生成: {got}")
                    if got != out:
                        import shutil
                        shutil.move(got, out)
                    print(f"    → 已保存到 {out}")
                    return
                if r.returncode != 0:
                    print(f"    退出码 {r.returncode}: {(r.stderr or r.stdout).strip()[:150]}")
            except Exception as e:
                print(f"    异常: {e}")

    # ---------- 2. Python API ----------
    print(f"\n[2] 探测 Python API")
    pkg, callables = probe_package()

    tried = []
    if pkg is not None:
        # 先试包级函数
        for name, fn in callables:
            if name in ("np", "os", "sys"):
                continue
            for form_args in [(urdf, out), (urdf,), (out, urdf)]:
                label = f"{name}{form_args}"
                if label in tried:
                    continue
                tried.append(label)
                try:
                    print(f"    尝试 {name}{form_args} ...")
                    fn(*form_args)
                    if os.path.isfile(out):
                        print(f"    ✅ 成功 → {out}")
                        return
                except Exception as e:
                    print(f"      失败: {type(e).__name__}: {str(e)[:120]}")

        # 再试子模块里的 main/CLI
        for m in pkgutil.iter_modules(pkg.__path__):
            try:
                sub = importlib.import_module(f"urdf2mjcf.{m.name}")
            except Exception:
                continue
            for n in dir(sub):
                obj = getattr(sub, n)
                if callable(obj) and n in ("main", "cli", "run", "convert",
                                           "urdf_to_mjcf", "urdf2mjcf",
                                           "convert_urdf_to_mjcf"):
                    try:
                        print(f"    尝试 urdf2mjcf.{m.name}.{n} ...")
                        try:
                            obj(urdf, out)
                        except TypeError:
                            obj(urdf)
                        if os.path.isfile(out):
                            print(f"    ✅ 成功 → {out}")
                            return
                    except Exception as e:
                        print(f"      失败: {type(e).__name__}: {str(e)[:120]}")

    # ---------- 3. 兜底：调用 convert_urdf_to_mjcf 类名 ----------
    print(f"\n[3] 兜底尝试常见函数名")
    for modname in ["urdf2mjcf", "urdf2mjcf.convert", "urdf2mjcf.main",
                    "urdf2mjcf.urdf2mjcf"]:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for fn_name in ["urdf2mjcf", "convert", "convert_urdf_to_mjcf",
                        "urdf_to_mjcf", "main", "run"]:
            fn = getattr(mod, fn_name, None)
            if fn is None or not callable(fn):
                continue
            try:
                print(f"    尝试 {modname}.{fn_name} ...")
                try:
                    fn(urdf, out)
                except TypeError:
                    fn(urdf)
                if os.path.isfile(out):
                    print(f"    ✅ 成功 → {out}")
                    return
            except Exception as e:
                print(f"      失败: {type(e).__name__}: {str(e)[:120]}")

    print("\n" + "=" * 78)
    print(" [!] 全部尝试失败")
    print("=" * 78)
    print(" 请手动运行下面命令看用法，然后把输出发我:")
    print(f"   ls ~/.local/bin/ | grep -i urdf")
    print(f"   python -c \"import urdf2mjcf, inspect, pkgutil; ")
    print(f"       print(urdf2mjcf.__file__); ")
    print(f"       print([m.name for m in pkgutil.iter_modules(urdf2mjcf.__path__)])\"")
    print("=" * 78)


if __name__ == "__main__":
    main()
