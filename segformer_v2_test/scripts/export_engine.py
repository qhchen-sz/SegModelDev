# -*- coding: utf-8 -*-
"""Export a trained best/ to ONNX and a TensorRT engine (fp16 by default).

Uses the frozen pipeline/export_best_onnx.py and trtexec with the same shapes as
deploy_cpp/tools/build_engine.bat:
    pixel_values: min 1x3x640x640, opt/max 16x3x640x640, fixed 640x640.

Usage:
    python export_engine.py --config <DS>/config_sw_s4.yaml --run sw_s4 \
        --deploy-dir D:/qhchen/NewModel/newsiliec9/deploy_cpp/models [--fp32] [--force]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PIPE = HERE / "pipeline"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", default="sw_s4")
    ap.add_argument("--name", default=None,
                    help="model/engine name; default = dataset folder name")
    ap.add_argument("--deploy-dir", default=None,
                    help="optional dir to copy the engine into (e.g. deploy_cpp/models)")
    ap.add_argument("--trt", default=r"E:\Tensorrt\TensorRT-8.6.1.6")
    ap.add_argument("--cuda", default=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    ds = Path(cfg["project_root"])
    name = args.name or ds.name
    best = Path(cfg["out_dir"]) / args.run / "best"
    if not (best / "config.json").exists():
        print(f"[ERROR] best model not found: {best}")
        return 1

    export_dir = ds / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    prec = "fp32" if args.fp32 else "fp16"
    onnx = export_dir / f"segformer_{name}_b1_640.onnx"
    engine = export_dir / f"segformer_{name}_b1_{prec}_640.engine"

    if engine.exists() and not args.force:
        print(f"[skip] engine already exists: {engine} (use --force to rebuild)")
    else:
        trtexec = Path(args.trt) / "bin" / "trtexec.exe"
        if not trtexec.exists():
            print(f"[ERROR] trtexec not found: {trtexec}")
            return 1

        print(f"[1/2] exporting ONNX: {best} -> {onnx}")
        cmd = [sys.executable, str(PIPE / "export_best_onnx.py"),
               "--src", str(best), "--out", str(onnx), "--opset", str(args.opset)]
        subprocess.run(cmd, cwd=str(PIPE), check=True)

        print(f"[2/2] building {prec} engine: {engine}")
        env = os.environ.copy()
        env["PATH"] = (str(Path(args.trt) / "lib") + os.pathsep +
                       str(Path(args.trt) / "bin") + os.pathsep +
                       str(Path(args.cuda) / "bin") + os.pathsep +
                       env.get("PATH", ""))
        cmd = [str(trtexec), f"--onnx={onnx}", f"--saveEngine={engine}"]
        if not args.fp32:
            cmd.append("--fp16")
        cmd += [
            "--minShapes=pixel_values:1x3x640x640",
            "--optShapes=pixel_values:16x3x640x640",
            "--maxShapes=pixel_values:16x3x640x640",
            "--memPoolSize=workspace:4096",
        ]
        subprocess.run(cmd, check=True, env=env)
        if not engine.exists():
            print(f"[ERROR] trtexec reported success but engine is missing: {engine}")
            return 1

    deployed = None
    if args.deploy_dir:
        deploy_dir = Path(args.deploy_dir)
        deploy_dir.mkdir(parents=True, exist_ok=True)
        deployed = deploy_dir / engine.name
        if deployed.resolve() != engine.resolve():
            shutil.copy2(engine, deployed)
        print(f"[copy] {engine} -> {deployed}")

    report = {
        "name": name,
        "run": args.run,
        "precision": prec,
        "best": str(best),
        "onnx": str(onnx),
        "onnx_bytes": onnx.stat().st_size,
        "onnx_sha256": sha256(onnx),
        "engine": str(engine),
        "engine_bytes": engine.stat().st_size,
        "engine_sha256": sha256(engine),
        "deployed": str(deployed) if deployed else None,
        "trtexec": str(Path(args.trt) / "bin" / "trtexec.exe"),
        "shapes": "pixel_values: min 1x3x640x640, opt/max 16x3x640x640",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (export_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {engine}")
    print(f"[OK] report: {export_dir / 'export_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
