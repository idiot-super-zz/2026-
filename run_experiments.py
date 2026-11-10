"""按预设表依次启动实验（单进程串行）。

两套预设：
    cifar    : CIFAR-100（32×32），只跑基线 E0 作为"低分辨率小样本"的对照
    crack    : 混凝土裂缝（224×224），基线 + 三项改造 + 组合，作为主消融实验

设计说明：**单进程串行**，直接在当前进程内调用 train.main(argv)，不起子进程，
Ctrl+C 一次即可全部停止，不会残留 DataLoader worker。

用法：
    python run_experiments.py --preset cifar --dry-run
    python run_experiments.py --preset cifar --only E0_baseline
    python run_experiments.py --preset crack --data-dir data/crack --epochs 40
    python run_experiments.py --preset crack --data-dir data/crack --only F1_cbam --force

断点续跑：runs/{exp}/summary.json 已存在则跳过（--force 强制重跑）。
"""

import argparse
import os
import time

from train import main as train_main

CIFAR_EXPS = {
    "E0_baseline": [],
    "E1_cbam": ["--use-cbam"],
    "E2_dw": ["--use-dw"],
    "E3_gn": ["--norm", "gn"],
    "E4_combo": ["--use-cbam", "--use-dw", "--norm", "gn"],
    "E5_bn_bs16": ["--batch-size", "16"],
    "E6_gn_bs16": ["--batch-size", "16", "--norm", "gn"],
}

# bs=16 时每 epoch 迭代数是 bs=128 的 8 倍，附录实验单独设定 epoch
CIFAR_EPOCH_OVERRIDES = {"E5_bn_bs16": 100, "E6_gn_bs16": 100}

PRESETS = {
    "cifar": {"exps": CIFAR_EXPS, "overrides": CIFAR_EPOCH_OVERRIDES,
              "common": [], "epochs": 150},
}


def build_argv(name, preset, extra, args):
    cfg = PRESETS[preset]
    epochs = cfg["overrides"].get(name, args.epochs)
    argv = ["--exp-name", name, "--epochs", str(epochs), "--workers", str(args.workers)]
    argv += cfg["common"]
    if "--dataset" in cfg["common"] and args.data_dir:
        argv += ["--data-dir", args.data_dir]
    argv += cfg["exps"][name]
    argv += list(extra or [])
    return argv


def main():
    p = argparse.ArgumentParser("批量运行实验（单进程串行）")
    p.add_argument("--preset", type=str, default="cifar", choices=list(PRESETS))
    p.add_argument("--data-dir", type=str, default=None,
                   help="imagefolder 预设的数据根目录（含 train/ 与 val/）")
    p.add_argument("--epochs", type=int, default=None, help="缺省：cifar=150, crack=40")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="追加给 train.py 的参数，例如 --extra --data-root /root/autodl-tmp/data")
    args = p.parse_args()

    if args.epochs is None:
        args.epochs = PRESETS[args.preset]["epochs"]

    exps = PRESETS[args.preset]["exps"]
    if args.preset == "crack" and not args.data_dir:
        raise SystemExit("--preset crack 需要指定 --data-dir（含 train/ 与 val/ 的目录）")

    names = args.only or list(exps.keys())
    todo = []
    for name in names:
        if name not in exps:
            raise SystemExit(f"未知实验名: {name}，可选: {list(exps)}")
        if not args.force and os.path.exists(os.path.join("runs", name, "summary.json")):
            print(f"[跳过] {name} 已完成（加 --force 可重跑）", flush=True)
            continue
        todo.append(name)

    if not todo:
        print("没有需要执行的实验。")
        return

    print(f"预设 {args.preset}：待执行 {len(todo)} 个实验 {todo}", flush=True)
    print("按 Ctrl+C 可随时中断，已完成的实验不会丢失。", flush=True)

    t0 = time.time()
    for i, name in enumerate(todo, 1):
        argv = build_argv(name, args.preset, args.extra, args)
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {name}\n"
              f"$ python train.py {' '.join(argv)}\n{'=' * 70}", flush=True)
        if args.dry_run:
            continue
        try:
            train_main(argv)
        except KeyboardInterrupt:
            print(f"\n用户中断，停在 {name}。已完成的实验保留在 runs/ 下。", flush=True)
            raise SystemExit(130)

    print(f"\n全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟。"
          f"\n生成表格：python make_tables.py --latex report_tables.tex", flush=True)


if __name__ == "__main__":
    main()
