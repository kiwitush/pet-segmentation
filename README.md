# U-Net 图像分割 — 损失函数工程实验

从零搭建 U-Net 语义分割模型，在 Oxford-IIIT Pet 数据集上进行三分类分割（前景、背景、边界），对比三种损失函数的性能。

## 项目结构

```
img_segment/
├── train.py             # 训练入口
├── unet.py              # U-Net 手写实现
├── loss.py              # Dice Loss + Cross-Entropy Loss + 组合损失
├── dataset.py           # Oxford-IIIT Pet 数据集加载与预处理
├── metrics.py           # mIoU 等评估指标
├── experiments/
│   ├── run_ce.py        # 实验1：仅 Cross-Entropy Loss
│   ├── run_dice.py      # 实验2：仅 Dice Loss
│   └── run_combined.py  # 实验3：CE + Dice 组合损失
├── eval/
│   ├── evaluate.py      # 验证集评估（mIoU / Dice / 混淆矩阵）
│   └── visualize.py     # 预测结果可视化
├── checkpoints/         # 保存模型权重
├── data/                # 数据集目录（首次自动下载）
├── requirements.txt
└── README.md
```

## 环境配置

创建虚拟环境并安装依赖

```bash
pip install -r requirements.txt
```
依赖说明：
- `torch`, `torchvision` — 深度学习框架
- `numpy`, `pillow` — 数据处理
- `swanlab` — 实验跟踪与可视化


## 训练

### 从项目根目录运行

**实验1 — 仅 Cross-Entropy Loss：**
```bash
python experiments/run_ce.py
```

**实验2 — 仅 Dice Loss：**
```bash
python experiments/run_dice.py
```

**实验3 — 组合损失 (CE + Dice)：**
```bash
python experiments/run_combined.py
```

### 自定义参数

所有实验脚本支持通过命令行覆盖默认参数：

```bash
python experiments/run_combined.py --epochs 100 --batch-size 16 --lr 5e-4 --weight-decay 5e-3
```

或者直接使用通用训练入口：

```bash
python train.py --loss combined --epochs 80 --batch-size 8 --lr 5e-4 --weight-decay 5e-3
```

### 切换损失函数

通过 `--loss` 参数切换：

| 参数值 | 损失函数 |
|--------|----------|
| `ce` | nn.CrossEntropyLoss |
| `dice` | 手写 DiceLoss |
| `combined` | 0.5 * CE + 0.5 * Dice |

## 评估

加载已训练模型在验证集上评估，输出 mIoU、Pixel Accuracy、逐类 IoU/Dice 和混淆矩阵：

```bash
python eval/evaluate.py --checkpoint checkpoints/unet_ce_best.pt
```

预测结果可视化：

```bash
python eval/visualize.py --checkpoint checkpoints/unet_ce_best.pt --num-samples 6
```

## 模型架构 (U-Net)

- **编码器**: 64→128→256→512→1024 通道下采样
- **瓶颈**: 1024 通道
- **解码器**: 1024→512→256→128→64 通道上采样
- **Skip Connection**: 编码器各层特征与解码器对应层拼接
- **输出**: 1×1 卷积 → 3 通道 logits
- **上采样方式**: 转置卷积

## 实验设置

| 项目 | 配置 |
|------|------|
| 数据集 | Oxford-IIIT Pet（80% 训练 / 20% 验证） |
| 输入尺寸 | 256×256 |
| 网络结构 | U-Net（随机初始化，无预训练） |
| 输出类别 | 3（前景/背景/边界） |
| Batch Size | 8 |
| 学习率 | 5e-4 |
| 优化器 | AdamW (weight_decay=5e-3) |
| 学习率调度 | 余弦衰减 |
| Max epoch | 80 |
| 评价指标 | mIoU |

## 训练监控

训练过程中 SwanLab 会实时记录：
- 训练集 loss 曲线
- 验证集 loss 曲线
- 验证集 mIoU 曲线
- 学习率变化曲线

训练开始会自动输出 swanlab 链接，可实时监控训练曲线。