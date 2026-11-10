"""训练/评测共用的工具函数。"""

import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """返回 top-k 准确率（百分比）。

    类别数少于 max(topk) 时（如二分类场景）自动截断，避免 topk 越界；
    此时 top-5 与 top-k 等价。
    """
    num_classes = output.size(1)
    maxk = min(max(topk), num_classes)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        kk = min(k, maxk)
        correct_k = correct[:kk].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class AverageMeter:
    """记录滑动平均与累计值。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class EpochTimer:
    """记录每 epoch 耗时并给出剩余时间估计，方便核对算力预算。"""

    def __init__(self, total_epochs: int):
        self.total = total_epochs
        self.start = time.time()
        self.epoch_times = []

    def step(self, epoch: int) -> str:
        now = time.time()
        self.epoch_times.append(now - self.start)
        self.start = now
        avg = sum(self.epoch_times) / len(self.epoch_times)
        remain = avg * (self.total - epoch - 1)
        return f"epoch_time={self.epoch_times[-1]:.1f}s avg={avg:.1f}s eta={remain / 60:.1f}min"

    @property
    def total_time(self) -> float:
        return sum(self.epoch_times)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def peak_gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0
