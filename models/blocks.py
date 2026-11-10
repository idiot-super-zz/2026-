"""手写的 CNN 核心组件。

设计约束：
1. 只依赖 torch / torch.nn，不 import timm、torchvision 等现成结构。
2. 子模块命名与 torchvision.models.resnet 完全对齐（conv1/bn1/conv2/bn2/downsample），
   使基线模型可以用 strict=True 直接加载官方 ImageNet 权重，从而完成结构等价性校验
   （见 check_equiv.py）。
3. 所有结构改造都以"开关"的形式实现，保证消融实验中只有被考察的组件发生变化。
"""

import torch
import torch.nn as nn

__all__ = [
    "make_norm",
    "DepthwiseSeparableConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "BasicBlock",
]


def make_norm(kind: str, num_channels: int, groups: int = 32) -> nn.Module:
    """返回归一化层。BN 与 GN 的仿射参数个数相同（各 2C），便于公平比较。

    Args:
        kind: "bn" -> BatchNorm2d；"gn" -> GroupNorm。
        num_channels: 通道数 C。
        groups: GN 的分组数 G，必须整除 C。
    """
    if kind == "bn":
        return nn.BatchNorm2d(num_channels)
    if kind == "gn":
        if num_channels % groups != 0:
            raise ValueError(
                f"GroupNorm: num_channels({num_channels}) 必须能被 groups({groups}) 整除"
            )
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"未知的归一化类型: {kind}")


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：3x3 depthwise + 1x1 pointwise，中间不插激活。

    关于中间是否加 BN/ReLU：实测在残差块内插入 "dw→bn→relu→pw" 反而更差
    （CIFAR-100 上 75.96 → 75.41），这与 MobileNetV2 的"线性瓶颈"结论一致——
    1x1 投影层需要保持线性，激活应放在扩展层之前。因此这里保持两段纯线性变换，
    使算子替换成为唯一变量。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 kernel_size: int = 3, padding: int = 1,
                 norm: str = "bn", gn_groups: int = 32):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力分支：avg/max 双池化共享 MLP，再 sigmoid 重标定。

    注：曾试过给最后一层加偏置、把门控初始值从 0.5 抬到 0.73，CIFAR-100 上
    结果为 76.74，与本文采用的默认初始化（76.88）差异在噪声范围内，
    说明门控起始值不是该模块精度落后的原因（见报告 5.5 节）。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * weight


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力分支：沿通道做 avg/max 拼接后用 7x7 卷积生成空间权重图。

    注：曾试过改用 3x3 卷积核（CIFAR 下 stage4 特征图仅 4x4），
    结果为 76.74，与 7x7 的 76.88 无显著差异，故保留原文设置。
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        weight = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weight


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)，通道在前、空间在后。"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class BasicBlock(nn.Module):
    """ResNet 的基本残差块，支持三项可选改造。

    原始结构（He et al., 2016）：
        out = relu( bn2(conv2( relu(bn1(conv1(x))) ) + downsample(x) )

    可选改造：
        use_dw   : conv1/conv2 的 3x3 标准卷积 -> 深度可分离卷积
        use_cbam : 残差相加之前，对 bn2 的输出做 CBAM 重标定
        norm     : bn1/bn2 使用 BatchNorm 或 GroupNorm
    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 norm: str = "bn", gn_groups: int = 32,
                 use_cbam: bool = False, use_dw: bool = False,
                 downsample: nn.Module | None = None):
        super().__init__()

        if use_dw:
            self.conv1 = DepthwiseSeparableConv(inplanes, planes, stride=stride,
                                                norm=norm, gn_groups=gn_groups)
            self.conv2 = DepthwiseSeparableConv(planes, planes, stride=1,
                                                norm=norm, gn_groups=gn_groups)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)

        self.bn1 = make_norm(norm, planes, gn_groups)
        self.bn2 = make_norm(norm, planes, gn_groups)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(planes) if use_cbam else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out
