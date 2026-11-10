"""训练脚本（CIFAR-100 主实验 / 场景数据微调通用）。

每次实验在 runs/{exp_name}/ 下产出：
    config.json    完整超参与结构配置（保证可追溯）
    metrics.jsonl  每个 epoch 的训练/验证指标
    summary.json   最终汇总（best top-1、参数量、FLOPs、耗时、显存），供 make_tables.py 自动出表
    best.pth       验证集最优权重
    last.pth       最后一轮权重

示例：
    python train.py --exp-name E0_baseline --epochs 150
    python train.py --exp-name E1_cbam --use-cbam --epochs 150
    python train.py --exp-name smoke --epochs 2 --subset 0.05
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from models import resnet18
from utils.data import build_loaders
from utils.misc import (
    AverageMeter,
    EpochTimer,
    accuracy,
    append_jsonl,
    count_params,
    peak_gpu_mem_mb,
    save_json,
    set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser("ResNet-18 结构改造实验：训练")
    # 实验管理
    p.add_argument("--exp-name", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=42)

    # 数据
    p.add_argument("--dataset", type=str, default="cifar100",
                   choices=["cifar100", "cifar10", "imagefolder"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--data-dir", type=str, default=None, help="imagefolder 数据根目录（含 train/val）")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--cutout", type=int, default=16, help="Cutout 边长，0 表示关闭")
    p.add_argument("--subset", type=float, default=0.0, help="只取部分数据用于冒烟测试")

    # 结构开关
    p.add_argument("--stem", type=str, default="cifar", choices=["cifar", "imagenet"])
    p.add_argument("--norm", type=str, default="bn", choices=["bn", "gn"])
    p.add_argument("--gn-groups", type=int, default=32)
    p.add_argument("--use-cbam", action="store_true")
    p.add_argument("--use-dw", action="store_true")

    # 优化
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=5, help="线性 warmup 轮数，所有变体统一")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--print-freq", type=int, default=200)
    p.add_argument("--progress", type=str, default="auto", choices=["auto", "on", "off"],
                   help="进度条：auto 表示输出重定向到文件时自动关闭（避免日志被 \\r 刷爆）")
    return p.parse_args(argv)


def build_optimizer(args, model):
    decay, no_decay = model.weight_groups()
    return optim.SGD(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, momentum=args.momentum, nesterov=True,
    )


def build_scheduler(args, optimizer):
    """线性 warmup + 余弦退火。

    warmup 对所有变体统一开启：GroupNorm 在训练初期（仿射参数尚未稳定）对较大学习率
    更敏感，固定高学习率会出现损失先上升再回落的现象；统一 warmup 既能稳定 GN，
    也保证各变体的训练配方完全一致，消融结果仍然受控。
    """
    import math

    warmup = max(args.warmup_epochs, 0)
    total = max(args.epochs, 1)
    min_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(train_loader, model, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    use_amp = scaler is not None

    # 输出被重定向到文件时（nohup ... > train.log）自动关闭进度条，避免 \r 把日志刷成几万行
    pbar = tqdm(train_loader, desc=f"train e{epoch}", leave=False, dynamic_ncols=True,
                file=sys.stdout, disable=not args.show_bar)
    for i, (images, target) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            output = model(images)
            loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc1, acc5 = accuracy(output.detach().float(), target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        if args.show_bar and i % args.print_freq == 0:
            pbar.set_postfix(loss=f"{losses.avg:.3f}", top1=f"{top1.avg:.2f}")

    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def evaluate(val_loader, model, criterion, device, show_bar=True):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for images, target in tqdm(val_loader, desc="  val", leave=False, dynamic_ncols=True,
                               file=sys.stdout, disable=not show_bar):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
    return losses.avg, top1.avg, top5.avg


def main(argv=None):
    """argv 为空时从 sys.argv 解析；传入列表时供 run_experiments.py 在进程内调用。"""
    args = parse_args(argv)
    # auto：只有输出是终端时才显示进度条；nohup 重定向到日志文件时自动关闭
    args.show_bar = (sys.stdout.isatty() if args.progress == "auto" else args.progress == "on")
    set_seed(args.seed)
    run_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        # 同一进程连续跑多个实验时，峰值显存必须逐实验重置，否则会继承上一个实验的值
        torch.cuda.reset_peak_memory_stats()

    train_loader, val_loader, num_classes, class_names = build_loaders(args)

    model_kwargs = dict(
        num_classes=num_classes,
        stem=args.stem,
        norm=args.norm,
        gn_groups=args.gn_groups,
        use_cbam=args.use_cbam,
        use_dw=args.use_dw,
    )
    model = resnet18(**model_kwargs).to(device)
    n_params = count_params(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    config = {
        "exp_name": args.exp_name,
        "model": model_kwargs,
        "dataset": args.dataset,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "cutout": args.cutout,
        "img_size": args.img_size,
        "seed": args.seed,
        "amp": use_amp,
        "device": str(device),
        "torch": torch.__version__,
        "params": n_params,
    }
    save_json(config, os.path.join(run_dir, "config.json"))

    print(f"[{args.exp_name}] device={device} params={n_params:,} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} amp={use_amp}")

    best_top1 = -1.0
    best_epoch = -1
    timer = EpochTimer(args.epochs)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    for epoch in range(args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            train_loader, model, criterion, optimizer, scaler, device, epoch, args)
        val_loss, val_top1, val_top5 = evaluate(val_loader, model, criterion, device, args.show_bar)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": round(tr_loss, 4),
            "train_top1": round(tr_top1, 4),
            "train_top5": round(tr_top5, 4),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 4),
            "val_top5": round(val_top5, 4),
            "gpu_mem_mb": round(peak_gpu_mem_mb(), 1),
        }
        append_jsonl(record, metrics_path)

        if val_top1 > best_top1:
            best_top1, best_epoch = val_top1, epoch
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "val_top1": val_top1}, os.path.join(run_dir, "best.pth"))

        print(f"e{epoch:03d} lr={lr_now:.4f} tr_loss={tr_loss:.3f} tr@{1:.0f}={tr_top1:.2f} "
              f"| val_loss={val_loss:.3f} val@1={val_top1:.2f} val@5={val_top5:.2f} "
              f"| {timer.step(epoch)}", flush=True)

    torch.save({"model": model.state_dict(), "config": config, "epoch": args.epochs - 1,
                "val_top1": val_top1}, os.path.join(run_dir, "last.pth"))

    summary = {
        "exp_name": args.exp_name,
        "dataset": args.dataset,
        "model": model_kwargs,
        "params": n_params,
        "params_m": round(n_params / 1e6, 3),
        "best_top1": round(best_top1, 2),
        "best_epoch": best_epoch,
        "final_top1": round(val_top1, 2),
        "final_top5": round(val_top5, 2),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_time_min": round(timer.total_time / 60, 2),
        "sec_per_epoch": round(timer.total_time / max(args.epochs, 1), 2),
        "peak_gpu_mem_mb": round(peak_gpu_mem_mb(), 1),
        "device": str(device),
        "amp": use_amp,
        "seed": args.seed,
    }
    save_json(summary, os.path.join(run_dir, "summary.json"))

    # 及时释放：run_experiments.py 会在同一进程里连续跑多个实验
    del model, optimizer, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"\n[{args.exp_name}] best_top1={best_top1:.2f} @epoch{best_epoch} "
          f"time={summary['train_time_min']:.1f}min")
    print(f"结果已保存到 {run_dir}")


if __name__ == "__main__":
    main()
