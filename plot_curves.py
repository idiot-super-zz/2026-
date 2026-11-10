"""从 runs/*/metrics.jsonl 绘制训练曲线（验证准确率 / 损失），供报告插图使用。

用法：
    python plot_curves.py                       # 画 runs/ 下全部实验
    python plot_curves.py --exp E0_baseline E1_cbam --out figures/acc.png

输出：figures/ 下的 png（同时把原始曲线数据导出为 csv，便于在报告里定量描述）。
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser("绘制训练曲线")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--exp", nargs="*", default=None, help="指定实验名，缺省为全部")
    p.add_argument("--out-dir", type=str, default="figures")
    p.add_argument("--smooth", type=int, default=1, help="滑动平均窗口，1 表示不平滑")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.runs_dir, "*", "metrics.jsonl")))
    if not paths:
        print(f"{args.runs_dir}/ 下没有 metrics.jsonl，请先训练。")
        return

    fig_acc, ax_acc = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig_loss, ax_loss = plt.subplots(figsize=(7, 4.5), dpi=150)

    for path in paths:
        name = os.path.basename(os.path.dirname(path))
        if args.exp and name not in args.exp:
            continue
        rows = read_metrics(path)
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        val_top1 = [r["val_top1"] for r in rows]
        val_loss = [r["val_loss"] for r in rows]

        if args.smooth > 1:
            def smooth(xs):
                out = []
                for i in range(len(xs)):
                    lo = max(0, i - args.smooth + 1)
                    out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
                return out
            val_top1, val_loss = smooth(val_top1), smooth(val_loss)

        label = name.lstrip("_") or name  # matplotlib 会忽略以下划线开头的图例标签
        ax_acc.plot(epochs, val_top1, label=label, linewidth=1.6)
        ax_loss.plot(epochs, val_loss, label=label, linewidth=1.6)

        with open(os.path.join(args.out_dir, f"curve_{name}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_top1", "val_top1", "lr"])
            for r in rows:
                writer.writerow([r["epoch"], r["train_loss"], r["val_loss"],
                                 r["train_top1"], r["val_top1"], r["lr"]])

    for ax, ylabel, title in [
        (ax_acc, "Validation Top-1 (%)", "Validation accuracy"),
        (ax_loss, "Validation loss", "Validation loss"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    acc_path = os.path.join(args.out_dir, "val_top1.png")
    loss_path = os.path.join(args.out_dir, "val_loss.png")
    fig_acc.tight_layout()
    fig_acc.savefig(acc_path)
    fig_loss.tight_layout()
    fig_loss.savefig(loss_path)
    print(f"已保存 {acc_path}\n已保存 {loss_path}\n曲线 csv 也在 {args.out_dir}/")


if __name__ == "__main__":
    main()
