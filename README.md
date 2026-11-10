# ResNet-18 结构改造的受控消融实验（深度学习大作业）

在 CIFAR-100 上从零训练 ResNet-18 作为基线，对三项结构改造（CBAM 注意力 / 深度可分离卷积 / GroupNorm）
做受控消融对比，并如实报告与分析其负收益。全部结构手写实现，不使用 timm 等现成结构库，
仅以 `torchvision.models.resnet18` 作为结构等价性校验的参照物。

## 1. 环境

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

依赖：`torch>=2.1`、`torchvision`、`thop`、`tqdm`、`numpy`、`matplotlib`。

参考配置（本仓库的实测环境）：
- 训练：单卡 RTX 3080 Ti（12 GB），CUDA，PyTorch 2.x + AMP
- 开发校验：CPU-only PyTorch（结构等价性自检、冒烟训练）

## 2. 数据准备

不需要 clone 任何数据集仓库，也不需要手动下载：

- **CIFAR-100**：脚本首次运行时通过 `torchvision.datasets` 自动下载到 `./data`（约 169 MB）

> 若网络受限，可手动下载 `cifar-100-python.tar.gz`（md5 `EB9058C3A382FFC7106E4002C42A8D85`）放到 `data/` 下，
> 脚本会自动校验并解压，不再联网。

## 3. 结构等价性自检（先跑这一步）

```bash
python check_equiv.py
```

用 `torchvision.models.resnet18` 当参照物，做四件事：

1. `stem=imagenet` 时用 **`strict=True`** 加载官方 `state_dict`——键名必须完全一致
2. 224×224 输入下比较两边 logits，误差应为 0
3. `stem=cifar` 时把参照模型的 stem 同步替换后比较 32×32 输入下的输出
4. 打印各变体参数量，并与手工推算值对齐（基线 11,220,132；GN 与 BN 参数量必须完全相同）

本机实测输出（CPU）：

```
[Test A] stem=imagenet  224x224  keys=122  max|logit diff|=0.00e+00  PASS
[Test B] stem=cifar     32x32    max|logit diff|=0.00e+00  PASS

  torchvision resnet18 (1000 类) : 11,689,512
  torchvision resnet18 (100 类)  : 11,227,812
  E0 baseline (cifar stem)  :   11,220,132
  E1 +CBAM                  :   11,308,980  (100.8%)
  E2 +DWConv                :    1,485,796  ( 13.2%)
  E3 +GroupNorm             :   11,220,132  (100.0%)
  E4 combo (CBAM+DW+GN)     :    1,574,644  ( 14.0%)

计算量（输入 1x3x32x32，单位 GMACs）：
  E0 baseline   : 0.558 G  (100.0%)
  E1 +CBAM      : 0.559 G  (100.1%)
  E2 +DWConv    : 0.076 G  ( 13.5%)
  E4 combo      : 0.074 G  ( 13.2%)
```


## 4. 复现全部实验

```bash
# 查看将执行的命令
python run_experiments.py --dry-run

# 依次跑完 7 个实验（总时长取决于单 epoch 耗时，先看下方"校准算力"）
python run_experiments.py --epochs 150 --workers 8

# 或单独跑某个实验
python train.py --exp-name E0_baseline --epochs 150
```

> `run_experiments.py` 采用**单进程串行**：它在当前进程内直接调用 `train.main(argv)`，不起子进程。
> 因此 `Ctrl+C` 一次即可全部停止，不会残留 `train.py` 或 DataLoader worker；已跑完的实验结果不受影响。

| 实验 | 结构 | 命令要点 |
|---|---|---|
| E0_baseline | ResNet-18 | — |
| E1_cbam | + CBAM | `--use-cbam` |
| E2_dw | + 深度可分离卷积 | `--use-dw` |
| E3_gn | + GroupNorm (G=32) | `--norm gn` |
| E4_combo | CBAM + DW + GN | 三者同开 |
| E5_bn_bs16 | BN, batch=16 | 附录：batch 耦合性对照（脚本内固定 100 epoch） |
| E6_gn_bs16 | GN, batch=16 | 附录：batch 耦合性对照（脚本内固定 100 epoch） |

> E5/E6 的 batch 更小、每 epoch 迭代数是主实验的 8 倍，因此脚本内单独把 epoch 定为 100
> （`EPOCH_OVERRIDES`）。两者互为对照，同 bs、同 lr、同 epoch，不影响"BN vs GN 在小 batch 下
> 谁更稳"这一结论。

**冒烟测试**（不需要数据集，验证前向/反向/优化器与 GN 小 batch 通路）：

```bash
python smoke_test.py
```

**冒烟训练**（验证真实数据的完整训练循环，几分钟内完成）：

```bash
python train.py --exp-name smoke --epochs 2 --subset 0.05 --workers 0
```

### 后台运行与日志

`--progress auto`（默认）会在输出被重定向到文件时自动关闭进度条，因此 `nohup` 挂机时日志里每个
epoch 只有一行，不会被 `\r` 刷成几万行：

```bash
nohup python run_experiments.py --epochs 150 --workers 8 > train_all.log 2>&1 &
tail -f train_all.log      # 断线重连后再执行这条即可继续看
```

在终端里交互运行时仍会正常显示进度条；需要强制控制时用 `--progress on|off`。

### 先校准算力，再决定 epoch 数

正式开跑前先用**完整数据**跑 1 个 epoch，读出 `summary.json` 里的 `sec_per_epoch`，再按下表估算总时长：

| 单 epoch 耗时 | E0~E4（各 150 ep） | 建议 |
|---|---|---|
| ≤ 20 s | 3~4 h | 按 150 epoch 跑 |
| 20~35 s | 4~6 h | 按 150 epoch 跑，E5/E6 可砍掉 |
| 35~50 s | 6~8 h | 主实验降到 120 epoch（结论不变，绝对值略低） |
| > 50 s | > 8 h | 检查 AMP 是否开启、`--workers` 是否够大 |

```powershell
python train.py --exp-name calib --epochs 1 --workers 8
Get-Content runs\calib\summary.json | Select-String sec_per_epoch
```

**Windows 控制台中文乱码**：脚本输出为 UTF-8，若 PowerShell 显示乱码，先执行 `chcp 65001`。

## 5. 效率评测、出表与作图

```bash
# 参数量 / FLOPs / CPU-GPU 延迟
python benchmark.py --ckpt runs/E1_cbam/best.pth

# 补齐 benchmark.json 中缺失的 FLOPs 字段（训练机没装 thop 时需执行）
python fill_flops.py

# 由 runs/*/summary.json + benchmark.json 自动生成报告表格
python make_tables.py --latex report_tables.tex

# 训练曲线（报告插图）
python plot_curves.py
```

> 报告中所有数字均须来自 `make_tables.py` 的输出，禁止手抄，以保证与代码运行结果一致。

## 6. 训练配方（所有变体完全一致）

| 项 | 取值 |
|---|---|
| 优化器 | SGD，momentum 0.9，nesterov，weight decay 5e-4（仅作用于权重，BN/GN 仿射参数与 bias 不衰减） |
| base lr | 0.1 |
| 调度 | 5 epoch 线性 warmup + 余弦退火 |
| batch size | 128（附录 E5/E6 用 16） |
| epoch | 150 |
| 损失 | CrossEntropy，label smoothing 0.1 |
| 增强 | RandomCrop(32, pad=4) + RandomHorizontalFlip + Cutout(16) |
| 精度 | AMP（CUDA 下自动开启，`--no-amp` 可关） |
| seed | 42 |

## 7. 目录结构

```
.
├── models/
│   ├── blocks.py        # make_norm / DepthwiseSeparableConv / CBAM / BasicBlock
│   └── resnet.py        # ResNet + resnet18() + build_model()
├── utils/
│   ├── data.py          # CIFAR-100 的 transforms 与 DataLoader
│   └── misc.py          # seed / accuracy / 计时 / json 读写
├── train.py             # 训练（输出 config/metrics/summary/best.pth）
├── evaluate.py          # 测试集评测（可选混淆矩阵）
├── benchmark.py         # Params / MACs / GPU-CPU 延迟
├── check_equiv.py       # 与 torchvision 的结构等价性自检
├── smoke_test.py        # 无需数据的训练通路自检
├── run_experiments.py   # 批量实验编排（单进程串行，断点续跑）
├── fill_flops.py        # 补齐 benchmark.json 中缺失的 FLOPs 字段
├── make_tables.py       # 自动生成 Markdown / LaTeX 结果表
├── plot_curves.py       # 训练曲线绘制
├── data/                # CIFAR-100 自动下载（gitignore）
└── runs/                # 7 组实验产物，共 28 个 JSON（约 184 KB，纳入仓库）
                         # 权重 *.pth 不纳入版本管理
```

## 8. 实现要点（写报告时可引用）

1. **受控消融**：三个改造分别作用于三个互不重叠的组件——注意力模块、卷积算子、归一化层；
   所有实验共用同一 seed、同一增强、同一 epoch 数与学习率调度，只改结构。
2. **DW 卷积采用 MobileNet 标准写法**：`DepthwiseSeparableConv` 为 `dw → bn → relu → pw`，
   中间归一化与残差块保持同一种类型（BN 或 GN）。初版曾为"纯算子替换"省去中间 BN/ReLU，
   实测精度损失达 −2.4 个点，远大于文献中的轻量化代价，故补回。
3. **CBAM 插入位置与初始化**：在每个 `BasicBlock` 的 `bn2` 之后、残差相加之前重标定特征；
   空间分支用 3×3 卷积（CIFAR 下 stage4 特征图仅 4×4，7×7 会退化成常数门控）；
   两个分支的门控偏置初始化为 1.0 与 2.0，使初始 sigmoid 分别约 0.73 与 0.88，
   避免残差分支在训练初期被衰减到 0.25 倍。
4. **GN 固定 G=32**：遵循 Wu & He (ECCV 2018) 的默认设置，全网络一致，不做额外搜索；
   GN 与 BN 的仿射参数个数相同（各 2C），参数量差异只来自结构本身。
5. **CIFAR 适配**：32×32 输入下把 7×7/stride-2 卷积 + maxpool 换成 3×3/stride-1 卷积，
   其余 stage 超参与 ImageNet 版 ResNet-18 一致（见 `check_equiv.py` 的 Test B）。
6. **权重衰减**：只作用于卷积/全连接权重，BN/GN 仿射参数与 bias 不衰减。
7. **统一 warmup（待 GPU 确认）**：CPU 上的快速通路测试（随机数据、3 步）观察到，GN 变体在
   固定 lr=0.1 时损失先上升再回落，而 BN 变体单调下降；两者初始化时的梯度范数接近
   （3.84 vs 3.57），说明差异来自 GN 对初始学习率更敏感，而非梯度尺度异常。
   因此所有变体统一使用 5 epoch 线性 warmup（保持"只改结构、配方一致"），并在正式训练前
   先观察前 10 个 epoch 的 val_top1 是否正常上升；若 GN 仍出现早期发散，把 `--lr` 降到 0.05
   重跑 E3/E4 并在报告中记录（这本身也是一条"归一化层与优化超参耦合"的有效观察）。

## 9. 结构定义出处

- ResNet：He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
  （官方实现 https://github.com/KaimingHe/deep-residual-networks）
- CIFAR 适配与训练配方参考：https://github.com/kuangliu/pytorch-cifar 、torchvision
- CBAM：Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018
  （https://github.com/Jongchan/attention-module）
- GroupNorm：Wu & He, *Group Normalization*, ECCV 2018（https://arxiv.org/abs/1803.08494）
- 深度可分离卷积：Howard et al., *MobileNets*, 2017

本文结构均为自行实现，上述仓库仅作对照与出处引用。

## 10. 结果

七个实验的实测结果（完整字段由 `make_tables.py` 生成到 `report_tables.md`）：

| 实验 | 配置 | batch | epoch | Params (M) | FLOPs (G) | Top-1 (%) |
|---|---|---|---|---|---|---|
| E0 | ResNet-18 基线 | 128 | 150 | 11.22 | 1.12 | **78.36** |
| E1 | ＋CBAM | 128 | 150 | 11.31 | 1.12 | 76.88 |
| E2 | ＋深度可分离卷积 | 128 | 150 | 1.49 | 0.15 | 75.96 |
| E3 | ＋GroupNorm (G=32) | 128 | 150 | 11.22 | 1.11 | 74.88 |
| E4 | 三者组合 | 128 | 150 | 1.57 | 0.15 | 69.72 |
| E5 | BN, batch=16 | 16 | 100 | 11.22 | 1.12 | 76.73 |
| E6 | GN, batch=16 | 16 | 100 | 11.22 | 1.11 | 72.24 |

三项改造均未带来精度收益，原因分析见报告第 5 章（容量饱和 + FLOPs 失真），
本节数据仅作复现校验用。

## 11. 复现说明

- 固定 `--seed 42`；`cudnn.benchmark=True` 会带来极小的非确定性，Top-1 波动通常在 ±0.2% 内
- `summary.json` 中记录了 `torch` 版本、设备、AMP 开关、每 epoch 耗时与峰值显存
- 断点续跑：`run_experiments.py` 默认跳过已完成的实验（`--force` 强制重跑）
