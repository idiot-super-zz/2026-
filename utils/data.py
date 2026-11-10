"""数据集与 DataLoader 构建。

支持：
    cifar100 / cifar10 : 通过 torchvision.datasets 自动下载到 --data-root（无需单独的仓库）
    imagefolder        : 场景应用数据集（如 SDNET2018 裂缝分类、NEU-CLS），按 ImageFolder 组织
"""

import os

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Cutout:
    """随机遮挡一个方形区域（DeVries & Taylor, 2017），作为轻量正则化。"""

    def __init__(self, size: int = 16):
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[1], img.shape[2]
        cy = torch.randint(h, (1,)).item()
        cx = torch.randint(w, (1,)).item()
        half = self.size // 2
        y1, y2 = max(cy - half, 0), min(cy + half, h)
        x1, x2 = max(cx - half, 0), min(cx + half, w)
        img[:, y1:y2, x1:x2] = 0.0
        return img


def build_transforms(dataset: str, img_size: int, cutout: int, train: bool):
    if dataset in ("cifar100", "cifar10"):
        mean, std = (CIFAR100_MEAN, CIFAR100_STD) if dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)
        if train:
            ops = [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
            if cutout > 0:
                ops.append(Cutout(cutout))
            return T.Compose(ops)
        return T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # imagefolder：把短边缩放到 img_size 的 1.14 倍后随机裁 img_size，与 ImageNet 训练惯例一致
    resize = int(round(img_size * 1.14))
    if train:
        ops = [
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if cutout > 0:
            ops.append(Cutout(cutout))
        return T.Compose(ops)
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(args):
    """返回 (train_loader, val_loader, num_classes, class_names)。"""
    train_tf = build_transforms(args.dataset, args.img_size, args.cutout, train=True)
    val_tf = build_transforms(args.dataset, args.img_size, args.cutout, train=False)

    if args.dataset in ("cifar100", "cifar10"):
        cls = torchvision.datasets.CIFAR100 if args.dataset == "cifar100" else torchvision.datasets.CIFAR10
        train_set = cls(args.data_root, train=True, transform=train_tf, download=True)
        val_set = cls(args.data_root, train=False, transform=val_tf, download=True)
        num_classes = 100 if args.dataset == "cifar100" else 10
        class_names = train_set.classes if hasattr(train_set, "classes") else None
    elif args.dataset == "imagefolder":
        if not args.data_dir:
            raise ValueError("imagefolder 数据集必须提供 --data-dir")
        train_set = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
        val_set = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_tf)
        num_classes = len(train_set.classes)
        class_names = train_set.classes
    else:
        raise ValueError(f"未知数据集: {args.dataset}")

    # 冒烟测试用：只取一小部分样本，用于验证耗时而不影响正式实验
    if getattr(args, "subset", 0) and args.subset > 0:
        n = max(int(len(train_set) * args.subset), args.batch_size * 2)
        train_set = Subset(train_set, list(range(min(n, len(train_set)))))
        n_val = max(int(len(val_set) * args.subset), args.batch_size)
        val_set = Subset(val_set, list(range(min(n_val, len(val_set)))))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=max(args.batch_size, 100), shuffle=False, num_workers=args.workers,
        pin_memory=pin, persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader, num_classes, class_names
