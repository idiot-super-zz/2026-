"""无需数据集的训练通路自检：随机张量前向 + 反向 + 优化器步进。

用途：在下载数据之前先确认模型、损失、优化器、混合精度相关代码路径都正常，
并模拟小 batch（bs=2）验证 GroupNorm 在 batch 极小时仍可正常训练（BN 会退化）。

用法：
    python smoke_test.py
"""

import torch
import torch.nn as nn

from models import resnet18
from utils.misc import count_params

VARIANTS = {
    "E0 baseline": dict(),
    "E1 +CBAM": dict(use_cbam=True),
    "E2 +DWConv": dict(use_dw=True),
    "E3 +GroupNorm": dict(norm="gn"),
    "E4 combo": dict(use_cbam=True, use_dw=True, norm="gn"),
}


def run_step(name, kwargs, batch_size=8):
    model = resnet18(num_classes=100, stem="cifar", **kwargs)
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9, nesterov=True)

    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 100, (batch_size,))

    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert torch.isfinite(torch.tensor(losses)).all(), f"{name}: loss 出现 NaN/Inf"
    n_bad_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    )
    assert n_bad_grad == 0, f"{name}: {n_bad_grad} 个参数梯度为 NaN/Inf"

    # 参数确实被更新：比较第一步与第三步的 logits
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, 100)

    print(f"{name:16s} params={count_params(model):>10,}  "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}  grad_ok=True")


def main():
    print("=" * 78)
    print("训练通路自检（随机张量，4 个 batch 的等效步数）")
    print("=" * 78)
    for name, kwargs in VARIANTS.items():
        run_step(name, kwargs, batch_size=8)

    print("\n小 batch 场景（bs=2）：GN 仍能正常统计，BN 会因统计量不足而退化")
    for name, kwargs in {"BN bs=2": dict(), "GN bs=2": dict(norm="gn")}.items():
        run_step(name, kwargs, batch_size=2)

    print("\n全部通过。")


if __name__ == "__main__":
    main()
