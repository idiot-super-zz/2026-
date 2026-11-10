"""补齐 runs/*/benchmark.json 中缺失的 FLOPs 字段。

场景：训练机上没有安装 thop 时，benchmark.py 会把 macs_g / flops_g 记为 null，
而延迟数据是真实测得的、值得保留。本脚本只补这两个字段，不动 latency/throughput。

计算量与设备无关，可在任意装有 thop 的机器上运行：
    python fill_flops.py
"""

import glob
import json
import os

import torch

from models.resnet import build_model

try:
    from thop import profile
except ImportError:
    raise SystemExit("需要 thop：pip install thop")


def main():
    paths = sorted(glob.glob(os.path.join("runs", "*", "config.json")))
    if not paths:
        raise SystemExit("未找到 runs/*/config.json")

    for cfg_path in paths:
        run_dir = os.path.dirname(cfg_path)
        name = os.path.basename(run_dir)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        img_size = cfg.get("img_size", 32)
        model = build_model(cfg["model"]).eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(torch.randn(1, 3, img_size, img_size),), verbose=False)

        bench_path = os.path.join(run_dir, "benchmark.json")
        if os.path.exists(bench_path):
            with open(bench_path, "r", encoding="utf-8") as f:
                bench = json.load(f)
        else:
            bench = {"exp_name": name, "model": cfg["model"], "img_size": img_size,
                     "params": cfg.get("params"), "latency": {}, "throughput": {}}

        bench["macs_g"] = round(macs / 1e9, 4)
        bench["flops_g"] = round(macs * 2 / 1e9, 4)

        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(bench, f, ensure_ascii=False, indent=2)

        print("%-12s MACs=%.4f G  FLOPs=%.4f G" % (name, macs / 1e9, macs * 2 / 1e9))


if __name__ == "__main__":
    main()
