"""效率评测：参数量、MACs/FLOPs、单图延迟、吞吐、显存。

用法（两种，二选一）：
    # 1) 直接指定结构开关
    python benchmark.py --exp-name E0_baseline
    python benchmark.py --exp-name E1_cbam --use-cbam

    # 2) 从训练产出的 checkpoint 读取结构配置
    python benchmark.py --ckpt runs/E1_cbam/best.pth

输出：runs/{exp_name}/benchmark.json，并打印到终端。
"""

import argparse
import json
import os
import time

import torch

from models.resnet import build_model, resnet18
from utils.misc import count_params, save_json


def parse_args():
    p = argparse.ArgumentParser("效率评测")
    p.add_argument("--exp-name", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--stem", type=str, default="cifar", choices=["cifar", "imagenet"])
    p.add_argument("--norm", type=str, default="bn", choices=["bn", "gn"])
    p.add_argument("--gn-groups", type=int, default=32)
    p.add_argument("--use-cbam", action="store_true")
    p.add_argument("--use-dw", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args()


def measure_latency(model, device, img_size, batch_size, warmup, iters):
    """返回 (ms/batch, 吞吐 images/s)。"""
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt * 1000, batch_size / dt


def main():
    args = parse_args()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["config"]["model"])
        exp_name = args.exp_name or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
        img_size = ckpt["config"].get("img_size", args.img_size)
    else:
        cfg = dict(num_classes=args.num_classes, stem=args.stem, norm=args.norm,
                   gn_groups=args.gn_groups, use_cbam=args.use_cbam, use_dw=args.use_dw)
        exp_name = args.exp_name or "benchmark"
        img_size = args.img_size

    model = build_model(cfg)
    n_params = count_params(model)

    macs = None
    try:
        from thop import profile
        macs, _ = profile(model.eval(), inputs=(torch.randn(1, 3, img_size, img_size),), verbose=False)
    except ImportError:
        print("[警告] 未安装 thop，跳过 FLOPs 统计")

    result = {
        "exp_name": exp_name,
        "model": cfg,
        "img_size": img_size,
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "macs_g": round(macs / 1e9, 4) if macs else None,
        "flops_g": round(macs * 2 / 1e9, 4) if macs else None,
        "latency": {},
        "throughput": {},
    }

    # CPU 延迟（部署视角）
    model_cpu = model.to("cpu")
    ms, thr = measure_latency(model_cpu, torch.device("cpu"), img_size, 1, args.warmup, max(args.iters // 2, 10))
    result["latency"]["cpu_bs1_ms"] = round(ms, 3)
    result["throughput"]["cpu_bs1_ips"] = round(thr, 2)

    # GPU 延迟（训练/推理视角）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_gpu = model.to(device)
        torch.cuda.reset_peak_memory_stats()
        for bs in (1, 64):
            ms, thr = measure_latency(model_gpu, device, img_size, bs, args.warmup, args.iters)
            result["latency"][f"gpu_bs{bs}_ms"] = round(ms, 3)
            result["throughput"][f"gpu_bs{bs}_ips"] = round(thr, 1)
        result["peak_infer_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    else:
        print("[提示] 未检测到 CUDA，仅统计 CPU 延迟")

    out_dir = os.path.join(args.out_dir, exp_name)
    save_json(result, os.path.join(out_dir, "benchmark.json"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
