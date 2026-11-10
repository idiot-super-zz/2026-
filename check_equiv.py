"""结构等价性自检：证明"手写的 ResNet-18"就是"标准 ResNet-18"。

不需要 clone 任何官方仓库 —— torchvision 随 PyTorch 一起安装，直接当参照物：

  Test A  stem=imagenet：用 strict=True 加载 torchvision 的 state_dict（键名必须完全一致），
          再比较 224x224 输入下的输出 logits，误差应为 0。
  Test B  stem=cifar：把参照模型的 stem 换成 3x3/stride-1 并去掉 maxpool，
          加载本文权重后比较 32x32 输入下的输出，误差应为 0。

同时打印各变体的参数量，校验与手工推算/文献值一致。

用法：
    python check_equiv.py
"""

import copy

import torch
import torch.nn as nn
import torchvision

from models import resnet18

ATOL = 1e-5


def _tv_reference(num_classes: int = 100) -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    return model.eval()


def test_a_imagenet_stem():
    """224x224 输入、ImageNet 标准 stem：state_dict 键名 + 输出必须完全一致。"""
    tv = _tv_reference(100)
    mine = resnet18(num_classes=100, stem="imagenet").eval()

    tv_keys = set(tv.state_dict().keys())
    my_keys = set(mine.state_dict().keys())
    assert tv_keys == my_keys, (
        f"state_dict 键名不一致\n仅 torchvision 有: {sorted(tv_keys - my_keys)}\n"
        f"仅本实现有: {sorted(my_keys - tv_keys)}"
    )

    missing = mine.load_state_dict(tv.state_dict(), strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y_tv = tv(x)
        y_mine = mine(x)
    diff = (y_tv - y_mine).abs().max().item()
    assert diff < ATOL, f"输出不一致，max|diff|={diff}"
    print(f"[Test A] stem=imagenet  224x224  keys={len(tv_keys)}  max|logit diff|={diff:.2e}  PASS")
    return diff


def test_b_cifar_stem():
    """32x32 输入、CIFAR 适配 stem：参照模型同步替换 stem 后输出必须完全一致。"""
    tv = _tv_reference(100)
    tv.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    tv.maxpool = nn.Identity()

    mine = resnet18(num_classes=100, stem="cifar").eval()
    missing = mine.load_state_dict(tv.state_dict(), strict=True)
    assert not missing.missing_keys, missing.missing_keys

    # 反向再验一次：把本文权重灌回参照模型
    tv_back = copy.deepcopy(tv)
    tv_back.load_state_dict(mine.state_dict(), strict=True)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y_tv = tv_back(x)
        y_mine = mine(x)
    diff = (y_tv - y_mine).abs().max().item()
    assert diff < ATOL, f"输出不一致，max|diff|={diff}"
    print(f"[Test B] stem=cifar     32x32    max|logit diff|={diff:.2e}  PASS")
    return diff


def report_params():
    """参数量核对：基线与 torchvision、各改造变体之间的差值应符合手工推算。"""
    rows = []

    tv_1k = _tv_reference(1000)
    tv_100 = _tv_reference(100)
    n_tv_1k = sum(p.numel() for p in tv_1k.parameters())
    n_tv_100 = sum(p.numel() for p in tv_100.parameters())
    assert n_tv_1k == 11689512, n_tv_1k
    assert n_tv_100 == 11227812, n_tv_100

    variants = {
        "E0 baseline (cifar stem)": dict(),
        "E1 +CBAM": dict(use_cbam=True),
        "E2 +DWConv": dict(use_dw=True),
        "E3 +GroupNorm": dict(norm="gn"),
        "E4 combo (CBAM+DW+GN)": dict(use_cbam=True, use_dw=True, norm="gn"),
    }
    for name, kwargs in variants.items():
        model = resnet18(num_classes=100, stem="cifar", **kwargs)
        rows.append((name, sum(p.numel() for p in model.parameters())))

    n_base = rows[0][1]
    # 手工推算：11,227,812（torchvision, 100 类） - 7x7 stem 9,408 + 3x3 stem 1,728
    assert n_base == 11220132, n_base
    assert rows[3][1] == n_base, "GN 与 BN 的仿射参数个数应完全相同"

    print("\n参数量核对：")
    print(f"  torchvision resnet18 (1000 类) : {n_tv_1k:,}")
    print(f"  torchvision resnet18 (100 类)  : {n_tv_100:,}")
    for name, n in rows:
        print(f"  {name:26s}: {n:>12,}  ({n / n_base * 100:5.1f}% of baseline)")
    return rows


def report_flops():
    """FLOPs：用 thop 在 32x32 输入下统计（1 MAC = 2 FLOPs，此处打印 MACs）。"""
    try:
        from thop import profile
    except ImportError:
        print("\n[跳过] 未安装 thop，无法统计 FLOPs：pip install thop")
        return

    print("\n计算量（输入 1x3x32x32，单位 GMACs）：")
    variants = {
        "E0 baseline": dict(),
        "E1 +CBAM": dict(use_cbam=True),
        "E2 +DWConv": dict(use_dw=True),
        "E4 combo": dict(use_cbam=True, use_dw=True, norm="gn"),
    }
    base = None
    for name, kwargs in variants.items():
        model = resnet18(num_classes=100, stem="cifar", **kwargs).eval()
        macs, _ = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
        base = base or macs
        print(f"  {name:14s}: {macs / 1e9:.3f} G  ({macs / base * 100:5.1f}% of baseline)")


if __name__ == "__main__":
    print("=" * 78)
    print("ResNet-18 结构等价性自检（参照物：torchvision.models.resnet18）")
    print("=" * 78)
    test_a_imagenet_stem()
    test_b_cifar_stem()
    report_params()
    report_flops()
    print("\n全部检查通过。")
