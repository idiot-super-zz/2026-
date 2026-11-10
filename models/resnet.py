"""ResNet-18（手写实现，带结构改造开关）。

与 torchvision.models.resnet 的对应关系：
    - 参数量、state_dict 键名、前向通路完全一致（stem="imagenet" 时可用 strict=True
      加载官方权重，见 check_equiv.py）。
    - stem="cifar" 时把 7x7/stride-2 卷积 + 3x3/stride-2 maxpool 换成 3x3/stride-1 卷积，
      用于 32x32 的 CIFAR 输入（沿用 He et al. 在 CIFAR 实验中的低分辨率适配方式）。

三项可选改造（对应消融实验的 E1-E3）：
    use_cbam : 在每个 BasicBlock 的残差相加前插入 CBAM
    use_dw   : 把 BasicBlock 内的 3x3 标准卷积替换为深度可分离卷积
    norm     : "bn" | "gn"（GN 固定 groups=32）
"""

import torch
import torch.nn as nn

from .blocks import BasicBlock, make_norm

__all__ = ["ResNet", "resnet18"]


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes: int = 100,
                 stem: str = "cifar", norm: str = "bn", gn_groups: int = 32,
                 use_cbam: bool = False, use_dw: bool = False,
                 zero_init_residual: bool = False):
        super().__init__()
        if stem not in ("cifar", "imagenet"):
            raise ValueError(f"未知的 stem 类型: {stem}")

        self.stem = stem
        self.norm_kind = norm
        self.gn_groups = gn_groups
        self.use_cbam = use_cbam
        self.use_dw = use_dw
        self._inplanes = 64

        if stem == "cifar":
            # 32x32 输入：单层 3x3/stride-1 卷积，不做下采样
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.maxpool = nn.Identity()
        else:
            # ImageNet 标准：7x7/stride-2 卷积 + 3x3/stride-2 maxpool
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.bn1 = make_norm(norm, 64, gn_groups)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(block, zero_init_residual)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self._inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self._inplanes, planes * block.expansion, 1, stride, bias=False),
                make_norm(self.norm_kind, planes * block.expansion, self.gn_groups),
            )

        layers = [
            block(
                self._inplanes, planes, stride=stride,
                norm=self.norm_kind, gn_groups=self.gn_groups,
                use_cbam=self.use_cbam, use_dw=self.use_dw,
                downsample=downsample,
            )
        ]
        self._inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self._inplanes, planes, stride=1,
                    norm=self.norm_kind, gn_groups=self.gn_groups,
                    use_cbam=self.use_cbam, use_dw=self.use_dw,
                    downsample=None,
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self, block, zero_init_residual: bool):
        """与 torchvision 的 ResNet 初始化保持一致（kaiming_normal_ fan_out）。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    last_norm = m.bn2
                    if last_norm.weight is not None:
                        nn.init.constant_(last_norm.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def weight_groups(self):
        """训练时不对 BN/GN 的仿射参数做 weight decay（与常见做法一致）。"""
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay


def resnet18(num_classes: int = 100, **kwargs) -> ResNet:
    """ResNet-18：4 个 stage，每个 stage 2 个 BasicBlock，通道 64/128/256/512。"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def build_model(model_cfg: dict) -> ResNet:
    """从 config 字典构建模型，供训练/评测/测速脚本共用。"""
    cfg = dict(model_cfg)
    num_classes = cfg.pop("num_classes", 100)
    arch = cfg.pop("arch", "resnet18")
    if arch != "resnet18":
        raise ValueError(f"当前仅实现 resnet18，收到 arch={arch}")
    return resnet18(num_classes=num_classes, **cfg)
