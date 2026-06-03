# Project Init Record — Restormer (CVPR 2022 Oral)

## 基本信息
- **仓库**: lack9921/Restormer (forked from swz30/Restormer)
- **分支**: main (commit: eb7c83c)
- **初始化时间**: 2026-06-03 ~21:25 GMT+8
- **版本**: 1.2.0
- **项目技术栈**: Python 3.10 + PyTorch + CUDA
- **项目类型**: 图像复原 Transformer (Deraining / Deblurring / Denoising / LoViF All-in-One)
- **原文**: [Restormer: Efficient Transformer for High-Resolution Image Restoration, CVPR 2022](https://arxiv.org/abs/2111.09881)

## 本项目特点（fork 版本改动）

此 fork 在官方 Restormer 基础上有以下关键改动：

1. **✅ 新增 LoViF 2026 训练/验证配置** — `LoViF_Restoration/` 目录
2. **✅ 支持 Progressive Learning** — `basicsr/train.py` 中实现 `mini_batch_sizes` / `gt_sizes` / `iters` 动态切换 patch_size & batch_size
3. **✅ ConcatLoViFDataset** — `basicsr/data/concat_lovif_dataset.py` 将 5 个 degradation track (Blur/Haze/Lowlight/Rain/Snow) 拼接成一个数据集
4. **✅ LPIPS 评估指标** — `basicsr/metrics/psnr_ssim.py` 集成了 `calculate_lpips`
5. **✅ Gradient Accumulation** — `basicsr/utils/gradient_accum.py` + `ImageCleanModel` 支持 `gradient_accumulation_steps`
6. **✅ CosineAnnealingRestartCyclicLR** — 两段重启余弦学习率调度
7. **✅ LoViF 验证脚本** — `split_lovif_val.py` 将训练集最后 N 张切出做验证
8. **✅ Sanity check at iter 0** — `basicsr/train.py` 在训练启动时先做一轮验证

## 项目架构总览

```
Restormer/
├── basicsr/                    # ⭐ 核心框架（基于 BasicSR 裁剪）
│   ├── train.py                # 训练入口（含渐进学习）
│   ├── test.py                 # 测试入口
│   ├── data/                   # 数据集和数据加载
│   │   ├── __init__.py         # 自动注册 dataset
│   │   ├── paired_image_dataset.py  # 标准成对图像数据集
│   │   ├── concat_lovif_dataset.py  # LoViF 多track拼接数据集
│   │   ├── data_sampler.py     # EnlargedSampler
│   │   ├── transforms.py       # 数据增强
│   │   └── ...
│   ├── models/
│   │   ├── __init__.py         # 自动扫描 _model.py 注册模型
│   │   ├── base_model.py       # 基础模型类（含 nondist_validation）
│   │   ├── image_restoration_model.py  # ⭐ ImageCleanModel（核心训练/验证逻辑）
│   │   ├── archs/
│   │   │   ├── __init__.py     # 自动扫描 _arch.py 注册架构
│   │   │   ├── restormer_arch.py  # ⭐ Restormer 网络定义（MDTA + GDFN）
│   │   │   └── arch_util.py
│   │   └── losses/
│   │       ├── __init__.py
│   │       ├── losses.py       # L1Loss, MSELoss 等
│   │       └── loss_util.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── psnr_ssim.py        # PSNR, SSIM, LPIPS 计算
│   │   └── metric_util.py
│   └── utils/
│       ├── options.py          # YAML 配置解析
│       ├── logger.py           # TensorBoard + wandb 日志
│       ├── img_util.py         # tensor2img 等
│       ├── gradient_accum.py   # GradientAccumWrapper
│       └── ...
├── Deraining/                  # 去雨任务
│   ├── test.py                 # 独立测试脚本
│   ├── utils.py                # PSNR/SSIM 计算（RGB 空间）
│   └── Options/
│       └── Deraining_Restormer.yml
├── Motion_Deblurring/          # 运动去模糊任务
│   ├── test.py                 # 独立测试脚本
│   └── Options/
│       └── Deblurring_Restormer.yml
├── Defocus_Deblurring/         # 散焦去模糊任务
│   ├── test_single_image_defocus_deblur.py
│   ├── test_dual_pixel_defocus_deblur.py
│   └── Options/
├── Denoising/                  # 去噪任务
│   └── evaluate_*.py
├── LoViF_Restoration/          # ⭐ LoViF 2026 比赛配置
│   └── Options/
│       ├── LoViF_Restoration_Restormer.yml            # 通用训练配置
│       └── LoViF_Restoration_Restormer_workstation.yml # 工作站训练配置
├── demo.py                     # 通用推理演示脚本
├── setup.py / setup.cfg        # Python 包安装
├── train.sh                    # 分布式训练启动脚本
└── split_lovif_val.py          # LoViF 数据切分脚本
```

### 各模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| 🚂 训练入口 | `basicsr/train.py` | 主训练循环，含 Progressive Learning、自动恢复、sanity validation |
| 🧪 测试入口 | `basicsr/test.py` | YAML → 数据加载 → 模型加载 → validation 全流程 |
| 🧠 模型架构 | `basicsr/models/archs/restormer_arch.py` | Restormer: MDTA + GDFN + U-shape encoder-decoder |
| 🏭 模型工厂 | `basicsr/models/image_restoration_model.py` | ImageCleanModel: 训练/验证/保存/加载/EMA |
| 📊 评估指标 | `basicsr/metrics/psnr_ssim.py` | PSNR, SSIM, LPIPS (AlexNet) |
| ⚙️ 配置引擎 | `basicsr/utils/options.py` | YAML 解析 + 路径自动生成 |
| 📦 数据加载 | `basicsr/data/paired_image_dataset.py` | 文件夹/LMDB/meta_info 三种模式读图 |
| 🔗 LoViF 数据 | `basicsr/data/concat_lovif_dataset.py` | 5 个 track 拼接为一个数据集 |
| 🖼️ 通用推理 | `demo.py` | 6 种任务推理 + tile 切块防 OOM |

## 数据流 / 执行链路

### 训练流程

```
train.py -opt LoViF_Restoration_Restormer.yml
  │
  ├── parse_options() → YAML → opt dict
  │
  ├── 自动恢复检测 (experiments/{name}/training_states/*.state)
  │
  ├── make_exp_dirs() → experiments/{name}/
  │
  ├── create_train_val_dataloader()
  │   ├── ConcatLoViFDataset (5 tracks merged)
  │   └── val_loaders (5 separate track loaders)
  │
  ├── create_model(opt) → ImageCleanModel
  │   ├── define_network(opt['network_g']) → Restormer(**params)
  │   ├── 加载预训练权重 (pretrain_network_g)
  │   └── init_training_settings(): 损失 / 优化器 / EMA / 梯度累积
  │
  ├── [Sanity validation at iter 0]
  │
  └── 训练循环（while current_iter ≤ total_iters）
      ├── Progressive Learning: 根据 iter 切换 gt_size / batch_size
      ├── feed_train_data → lq + gt (可选 MixingAugment)
      ├── optimize_parameters → L1Loss → backward + clip_grad + step
      ├── 每隔 print_freq: 打印日志
      ├── 每隔 save_checkpoint_freq: 保存模型
      └── 每隔 val_freq: 5 个 track 分别 validation（PSNR/SSIM/LPIPS）
```

### Progressive Learning 机制

```yaml
mini_batch_sizes: [2, 2, 1]     # 每个阶段的 batch_size
gt_sizes: [128, 192, 256]       # 每个阶段的 patch_size
iters: [100000, 100000, 100000] # 每个阶段的 iter 数
```

训练过程中自动根据 `current_iter` 落在哪个区间切换 patch 大小和 batch_size（小patch先训，大patch精调）。

### 推理流程

```
demo.py --task Deraining --input_dir ./input/ --result_dir ./output/
  │
  ├── get_weights_and_parameters() → 写死权重路径 + 模型参数
  ├── Restormer(**parameters)
  ├── 加载 checkpoint['params']
  ├── 对每张图：
  │   ├── 8 对齐 padding (reflect)
  │   ├── 可选 tile 切块（大图防 OOM）
  │   ├── model(input) → restored
  │   ├── unpadded + clamp(0,1)
  │   └── save
  └── 输出到结果目录
```

## Restormer 架构核心

```
输入 (HWC, 0-1)
  ↓
OverlapPatchEmbed (3×3 Conv, in→dim)
  ↓
Encoder Level 1: [TransformerBlock × 4]  (dim=48, heads=1)
  ↓ Downsample (Conv + PixelUnshuffle)
Encoder Level 2: [TransformerBlock × 6]  (dim=96, heads=2)
  ↓ Downsample
Encoder Level 3: [TransformerBlock × 6]  (dim=192, heads=4)
  ↓ Downsample
Latent:         [TransformerBlock × 8]  (dim=384, heads=8)
  ↓ Upsample (Conv + PixelShuffle)
Decoder Level 3: [TransformerBlock × 6]  (↓ skip-concat with enc L3)
  ↓ Upsample
Decoder Level 2: [TransformerBlock × 6]  (↓ skip-concat with enc L2)
  ↓ Upsample
Decoder Level 1: [TransformerBlock × 4]  (↓ skip-concat with enc L1)
  ↓
Refinement: [TransformerBlock × 4]
  ↓
Output Conv (3×3, dim*2 → out_channels) + Global Residual
  ↓
输出
```

### 关键模块

| 模块 | 说明 |
|------|------|
| **MDTA** (Multi-DConv Head Transposed Self-Attention) | 通道维度的自注意力（非空间维度），降低计算复杂度；1×1 conv + 3×3 depthwise conv → QKV → 转置注意力 |
| **GDFN** (Gated-Dconv Feed-Forward Network) | 门控机制：两个并行分支，GELU 激活后逐元素相乘；3×3 depthwise conv 编码空间信息 |
| **LayerNorm** | 两种模式：`WithBias`（标准 LN）和 `BiasFree`（无偏置，用于降噪任务） |
| **Global Residual** | 最终输出 = output_conv(out) + input_img |

## 配置中心

### LoViF 训练配置 (`LoViF_Restoration_Restormer.yml`)

| 配置项 | 值 | 作用 |
|--------|------|------|
| `name` | LoViF_AllTracks_Restormer | 实验名称，决定 exp 目录 |
| `model_type` | ImageCleanModel | 使用的模型类 |
| `scale` | 1 | 超分倍率（复原任务=1） |
| `num_gpu` | 1 | GPU 数量 |
| `datasets.train.type` | ConcatLoViFDataset | 5 track 拼接数据集 |
| `datasets.train.batch_size_per_gpu` | 2 | 基础 batch size |
| `datasets.train.gt_size` | 128 | 基础 patch 大小 |
| `datasets.train.mini_batch_sizes` | [2] | Progressive learning batch sizes |
| `datasets.train.gt_sizes` | [128] | Progressive learning patch sizes |
| `datasets.train.iters` | [300000] | 各阶段 iter 数 |
| `network_g.type` | Restormer | 网络架构 |
| `network_g.dim` | 48 | 基础通道数 |
| `network_g.num_blocks` | [4,6,6,8] | 各层 TransformerBlock 数量 |
| `network_g.heads` | [1,2,4,8] | 各层注意力头数 |
| `train.total_iter` | 300000 | 总迭代数 |
| `train.gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `train.optim_g.type` | AdamW | 优化器 |
| `train.optim_g.lr` | 3e-4 | 学习率 |
| `train.scheduler.type` | CosineAnnealingRestartCyclicLR | 两阶段重启余弦退火 |
| `train.pixel_opt.type` | L1Loss | 损失函数 |
| `val.window_size` | 8 | 验证时 8 对齐 |
| `val.val_freq` | 4000 | 验证频率 |
| `val.metrics` | psnr + ssim + lpips | 评估指标 |
| `logger.print_freq` | 1000 | 打印频率 |
| `logger.save_checkpoint_freq` | 4000 | 保存频率 |

## 编码风格约定

- **命名**: snake_case（Python 标准），类名 PascalCase
- **import**: 绝对 import（`basicsr.models.archs.restormer_arch`）
- **配置**: YAML 驱动，通过 `options.parse()` 加载为 OrderedDict
- **模型注册**: 自动扫描 `basicsr/models/` 下 `*_model.py` + `archs/*_arch.py`
- **日志**: `logging` 模块（`get_root_logger`），可选 TensorBoard + wandb
- **指标**: `basicsr/metrics/` 下的独立函数（`calculate_psnr`, `calculate_ssim`, `calculate_lpips`）
- **检查点**: 保存 `net_g` + `optimizer` + `epoch/iter` state，`.pth` 文件用 `params` 作为 key

## 关键设计决策

1. **基于 BasicSR 裁剪而不是重写** — 复用 BasicSR 的成熟架构（配置驱动、模型注册、dataloader 工厂），减少重复造轮子
2. **YAML 配置驱动** — 所有超参数、网络结构、数据路径都在 YAML 中声明，不硬编码
3. **Progressive Learning** — 小 patch 开始训练 → 逐步增大，兼顾早期收敛速度和后期精度
4. **Gradient Accumulation** — 在低显存设备上模拟大 batch，RTX 5060 8GB 也能训
5. **ConcatLoViFDataset** — 5 个 degradation track 拼接的轻量实现，每个 batch 随机从任一 track 采样
6. **自动恢复** — 检测 `experiments/{name}/training_states/` 中最新的 `.state` 文件自动恢复训练
7. **EMA** — 可选指数移动平均，测试时使用 EMA 权重提升稳定性
8. **CosineAnnealingRestartCyclicLR** — 两段重启余弦：先大学习率快速收敛，再小学习率精细调优

## 数据集结构

```
LoViF 2026 (FoundIR subset, 512×512)
├── Train/                 # 4900 pairs per track × 5 = 24,500 总
│   ├── Blur/GT/ LQ/
│   ├── Haze/GT/ LQ/
│   ├── Lowlight/GT/ LQ/
│   ├── Rain/GT/ LQ/
│   └── Snow/GT/ LQ/
├── Val/                   # 100 pairs per track × 5 = 500 总
│   └── (same structure)
└── Test/                  # 100 pairs per track × 5 = 500 总
    └── (same structure)
```

## 📌 待深入理解的区域

- `ConcatLoViFDataset` 的采样策略是均匀的还是按比例的？是否需要进行类别平衡？
- `CosineAnnealingRestartCyclicLR` 的两段周期（92000/208000）与 `total_iter=300000` 的对应关系
- `base_model.py` 中 `nondist_validation` 的 fully rewritten version 与 `ImageCleanModel.nondist_validation` 是哪个生效？（模型类覆盖了 base 版本）
- `dataset_enlarge_ratio: 1` 在训练中的作用（目前没用起来）
- 8GB 显存下 `gt_size: 128` + `batch_size: 2` + `gradient_accum: 8` 的 OOM 边界在哪里

## 🧹 清理建议

- `basicsr/models/base_model.py` 中的 `nondist_validation` 看起来是一个简化版实现，而 `ImageCleanModel` 中有完整的 metric 计算版本，可能有代码冗余
- `LoViF_Restoration_Restormer_workstation.yml` 和 `LoViF_Restoration_Restormer.yml` 内容相似，可以合并并参数化
- `train.sh` 写死了 8 GPU 分布式，单卡训练需要单独写启动命令
