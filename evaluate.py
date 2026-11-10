"""在测试集上评测某个 checkpoint，可选输出逐类准确率与混淆矩阵。

示例：
    python evaluate.py --ckpt runs/E0_baseline/best.pth
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet import build_model
from utils.data import build_transforms
from utils.misc import accuracy, save_json


def parse_args():
    p = argparse.ArgumentParser("评测 checkpoint")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10", "imagefolder"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--confusion", action="store_true", help="输出混淆矩阵到 ckpt 同目录")
    p.add_argument("--max-batches", type=int, default=0,
                   help="只评测前 N 个 batch（冒烟测试用），0 表示全量")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]["model"]
    img_size = ckpt["config"].get("img_size", args.img_size)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_tf = build_transforms(args.dataset, img_size, 0, train=False)
    if args.dataset == "imagefolder":
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_tf)
    else:
        cls = torchvision.datasets.CIFAR100 if args.dataset == "cifar100" else torchvision.datasets.CIFAR10
        test_set = cls(args.data_root, train=False, transform=val_tf, download=True)

    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=torch.cuda.is_available())

    n_cls = len(test_set.classes)
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long)
    top1_sum, top5_sum, total = 0.0, 0.0, 0

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader, desc="test", dynamic_ncols=True, file=sys.stdout)):
            if args.max_batches and i >= args.max_batches:
                break
            images, target = images.to(device), target.to(device)
            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_sum += acc1 * target.size(0)
            top5_sum += acc5 * target.size(0)
            total += target.size(0)
            if args.confusion:
                for t, p in zip(target.view(-1), output.argmax(1).view(-1)):
                    conf[t.long(), p.long()] += 1

    result = {"ckpt": args.ckpt, "test_top1": round(top1_sum / total, 2),
              "test_top5": round(top5_sum / total, 2), "num_samples": total}
    print(f"test_top1={result['test_top1']:.2f}  test_top5={result['test_top5']:.2f}  n={total}")

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    save_json(result, os.path.join(out_dir, "test_metrics.json"))
    if args.confusion:
        import csv
        with open(os.path.join(out_dir, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred"] + [str(i) for i in range(n_cls)])
            for i in range(n_cls):
                writer.writerow([str(i)] + conf[i].tolist())
        print(f"混淆矩阵已保存到 {os.path.join(out_dir, 'confusion_matrix.csv')}")


if __name__ == "__main__":
    main()
