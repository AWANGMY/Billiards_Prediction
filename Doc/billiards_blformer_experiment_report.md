# 台球开球预测 BLFormer 实验报告

## 1. 摘要

本报告整理 paper40 clean 口径下的台球开球预测实验。评估对象包括四类基线模型和 BLFormer:

- MLP: 多层感知机基线。
- Transformer: token-level Transformer 基线。
- BLCNN: 论文风格卷积基线。
- Attention: Spatial Attention 基线。
- BLFormer: 基于局面 set 表示和几何 attention bias 的 Transformer 模型。

三个预测任务为:

| Task | 含义 | 类别数 |
|---|---|---:|
| `clear` | 开球后是否清台 | 2 |
| `win` | 开球后是否最终赢局 | 2 |
| `potted_after_break` | 开球后进球数量 | 10 |

本报告不使用 `Dataset/processed/billiards_layout.pt`。该文件是项目已有 processed 产物，不能直接对应论文文字中的 40% train / 60% held-out evaluation，也不利于审计 raw data 到样本和 split 的生成过程。本文统一使用 `Output/reproduction/billiards_layout_paper40.pt`，该文件由公开 raw layout 数据通过 `ClassesData/PreprocessBilliards.py --split-method paper40` 重新预处理得到，共 1940 条 accepted samples，并固定为 776 train / 1164 held-out test。

在该口径下，基线结果明显低于论文表格中的 BLCNN 数值。BLFormer 的主要实验结论是: 仅扩大 independent-head 统一模型容量，对 `clear` 的改善有限；flat joint-head 更适合同时建模 `clear`、`win`、`potted_after_break` 三项任务之间的标签关系。其中 `joint_d80_clsmean` 取得 `71.391755 / 70.189005 / 65.378004` 的三任务 held-out accuracy，是本报告记录的 BLFormer 主结果。

## 2. 数据与评估协议

复现实验统一使用:

```text
Output/reproduction/billiards_layout_paper40.pt
```

数据规模:

| Item | Value |
|---|---:|
| samples | 1940 |
| train | 776 |
| held-out test | 1164 |
| validation | 0, 基线复现不使用 validation |

BLFormer 训练时在 776 条 train 内部再切出 validation，用于模型选择:

| Split | Count |
|---|---:|
| internal train | 660 |
| internal validation | 116 |
| final train | 776 |
| held-out test | 1164 |

评估原则:

- 所有正式数值均来自固定 held-out test。
- 不使用 held-out test 做 checkpoint selection、阈值选择或超参选择。
- 基线结果以 `Output/reproduction/formal/` 和 `Output/reproduction/formal_other_methods/` 中的可追踪 artifact 为准。
- BLFormer 结果以 `Output/blformer_paper40/` 中保存的 `config.json`、`history.csv`、`search_results.csv`、`summary.json` 和 test prediction 文件为准。

## 3. 基线复现结果

下表为 paper40 clean held-out test accuracy，单位为百分比。

| Model | Weight decay | clear | win | potted_after_break |
|---|---:|---:|---:|---:|
| BLCNN | 0.001 | 71.219931 | 66.580756 | 61.254296 |
| BLCNN | 0.0001 | 71.477663 | 66.752577 | 61.254296 |
| MLP | 0.001 | 69.158076 | 67.268041 | 55.068729 |
| Transformer | 0.001 | 65.807560 | 66.408935 | 40.120275 |
| Attention | 0.001 | 65.893471 | 66.408935 | 48.024055 |

结果来源:

```text
Output/reproduction/formal/BLCNN_formal_combined_summary.csv
Output/reproduction/formal_other_methods/OtherMethods_paper40_clean_combined_summary.csv
Doc/billiards_reproduction_report.md
```

观察:

- `clear` 上 BLCNN 最强，weight decay `0.0001` 略高于 `0.001`。
- `win` 上 MLP 与 BLCNN 接近，MLP `weight_decay=0.001` 的结果最高。
- `potted_after_break` 上 BLCNN 明显优于 MLP、Transformer 和 Attention。
- 公开 raw data 重新预处理后的 held-out 结果无法复现论文表格中 BLCNN 的 `89.69 / 86.56 / 80.94`。`Doc/billiards_reproduction_report.md` 中已记录 release-code-parity 口径存在 train/test overlap 和 test-set checkpoint selection，因此本报告不采用该口径作为 clean held-out 结论。

## 4. BLFormer 方法

BLFormer 与四类基线的主要区别是输入表示和 attention bias。它不直接把 `x_paper` 的 10x27 离散特征作为主输入，而是从连续布局 `x` 构造可变长 set:

```text
[CLS] + active ball tokens + 6 pocket tokens
```

每个 token 包含:

- token type embedding: `CLS`、ball、pocket、padding。
- ball id embedding: 区分 1 到 10 号球。
- coordinate encoder: 将 ball/pocket 的二维坐标映射到 `d_model`。

每对 token 之间计算 10 维 pair feature，并通过小型 MLP 映射为每个 attention head 的 bias:

| Pair feature group | 含义 |
|---|---|
| distance / dx / dy | token 间几何关系 |
| angle | ball-ball 到袋角度或 ball-pocket 入袋角 |
| blocked | 路径是否被其它球遮挡 |
| edge type one-hot | CLS/pad、ball-ball、ball-pocket、pocket-pocket |

模型主体为带几何 attention bias 的 Transformer encoder。输出端支持两类设计:

- Independent-head: `clear`、`win`、`potted_after_break` 分别由独立 head 预测。
- Flat joint-head: 将三个标签合成为 40 类 joint label，训练 40-way logits，再通过边缘化恢复三个任务的预测。

joint label 定义为:

```text
joint_label = clear * 20 + win * 10 + potted_after_break
```

报告中的 BLFormer 主线使用 `cls_mean` pooling，即拼接 CLS 表征和有效 token 均值表征。该设计在小数据下比只使用 CLS 更稳定。

## 5. BLFormer 实验结果

### 5.1 Independent-Head Capacity Baseline

该消融保持统一 BLFormer encoder 和 `cls_mean` pooling，但使用独立任务 head。`potted_after_break` 使用 class + ordinal hybrid loss。实验用于判断扩大统一模型容量是否能稳定改善三任务表现。

| Trial | Params | clear | win | potted_after_break | Mean acc |
|---|---:|---:|---:|---:|---:|
| `hybrid_d80_clsmean_ord0.25` | 114,986 | 69.673538 | 70.446736 | 66.151202 | 68.757159 |
| `hybrid_d88_clsmean_ord0.25` | 138,402 | 69.243985 | 70.446736 | 64.862543 | 68.184421 |

结果来源:

```text
Output/blformer_paper40/blformer_unified_capacity_20260706/search_results.csv
Output/blformer_paper40/blformer_unified_capacity_20260706/summary.json
```

观察: `d80` 和 `d88` 两个配置在 `win` 与 `potted_after_break` 上表现较强，但 `clear` 没有随参数量增加而稳定上升。该结果说明模型容量不是唯一限制因素。

### 5.2 Flat Joint-Head 消融

该消融使用同一套 encoder 和 `cls_mean` pooling，但把三任务标签合成为 40 类联合标签。推理时对 joint logits 做边缘化，得到 `clear`、`win`、`potted_after_break` 的预测。

| Trial | Params | clear | win | potted_after_break | Mean acc |
|---|---:|---:|---:|---:|---:|
| `joint_d64_clsmean` | 79,906 | 69.931269 | 70.532644 | 65.378004 | 68.613972 |
| `joint_d64_clsmean_marg0.5` | 79,906 | 70.274913 | 70.446736 | 65.378004 | 68.699884 |
| `joint_d80_clsmean` | 121,506 | 71.391755 | 70.189005 | 65.378004 | 68.986255 |
| `joint_d80_clsmean_marg0.5` | 121,506 | 70.360827 | 70.189005 | 66.838485 | 69.129439 |

结果来源:

```text
Output/blformer_paper40/blformer_joint_head_20260706/search_results.csv
Output/blformer_paper40/blformer_joint_head_20260706/summary.json
```

观察:

- 从 `d64` 提升到 `d80` 后，joint-head 的 `clear` 从 69.931269 提升到 71.391755。
- `joint_d80_clsmean` 在 `clear` 上是该组实验中最高的单模型结果。
- `joint_marginal_weight=0.5` 提高了 `potted_after_break`，但同时降低了 `clear`。因此主结果采用无 marginal 辅助 loss 的 `joint_d80_clsmean`，以保持三任务表现更均衡。

### 5.3 设计观察

历史探索结果显示:

- `cls_mean` pooling 比单独使用 CLS 更稳，能改善三任务平衡表现。
- `potted_after_break` 在 paper40 clean 中不是简单单峰有序分布，class/hybrid head 比纯 ordinal head 更适合 accuracy evaluation。
- Independent-head 的 class/hybrid potted head 能在 `potted_after_break` 上取得较高 accuracy，但对 `clear` 的改善有限。
- joint-head 通过显式建模 `clear`、`win`、`potted_after_break` 的联合标签关系，改善了 `clear` 的表现。

相关结果来源:

```text
Output/blformer_paper40/paper40_improve_20260706/search_results.csv
Output/blformer_paper40/paper40_clear_focus_20260706/search_results.csv
Doc/BLFormer_exploration_report.md
```

## 6. 结论

paper40 clean 口径下，公开 raw layout 数据重新预处理后的基线结果与论文表格数值存在明显差异。考虑到 release-code-parity 口径包含 train/test overlap 和 test-set checkpoint selection，本报告只讨论固定 held-out test 上的 clean evaluation。

BLFormer 的主要贡献是把台球局面建模为 set，并把距离、角度、遮挡、edge type 注入 attention bias。实验结果表明:

- Independent-head BLFormer 增大容量后，`win` 和 `potted_after_break` 表现较强，但 `clear` 没有稳定受益。
- Flat joint-head 能利用三任务标签之间的联合结构，`joint_d80_clsmean` 在 `clear` 上达到本组 BLFormer 单模型实验的最高值。
- `joint_d80_clsmean_marg0.5` 提高了 `potted_after_break`，但削弱了 `clear`；因此是否使用 marginal auxiliary loss 取决于更重视单项 potted accuracy 还是三任务均衡表现。
- 本报告推荐将 `joint_d80_clsmean` 作为当前 BLFormer 主结果，报告 accuracy 为 `71.391755 / 70.189005 / 65.378004`。

## 7. 复现命令

固定环境:

```bash
qsub -I -q interact-g -l select=1 -l walltime=02:00:00 -W group_list=gw17
cd /work/gw17/w17001/Data/code/Billiards_Prediction
PY=/work/gw17/w17001/envs/billiards-prediction/bin/python
```

如果只跑 CPU 小实验，可以跳过 `qsub -I`；BLFormer 搜索和长轮数基线建议按上面的交互 GPU 流程进入 GPU 节点后再运行。

复现 Independent-Head Capacity Baseline:

```bash
./run_blformer_unified_capacity_interact.sh
```

复现 flat joint-head:

```bash
./run_blformer_joint_head_interact.sh
```

复现 BLCNN paper40 clean:

```bash
$PY -u run_blcnn_reproduction.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --protocol paper40_clean \
  --tasks all \
  --epochs 400 \
  --weight-decay 0.0001 \
  --output-dir Output/reproduction/formal/paper40_clean_wd0.0001
```

复现 MLP、Transformer、Attention paper40 clean:

```bash
$PY -u run_other_methods_reproduction.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --protocol paper40_clean \
  --models MLP Transformer Attention \
  --tasks all \
  --epochs 400 \
  --weight-decay 0.001 \
  --device cuda \
  --output-dir Output/reproduction/formal_other_methods/paper40_clean_wd0.001 \
  --log-every-epochs 25
```

代码健康检查:

```bash
$PY test_blformer.py
```

## 8. 产物索引

本报告:

```text
Doc/billiards_blformer_experiment_report.md
```

核心已有文档:

```text
Doc/billiards_reproduction_report.md
Doc/BLFormer_exploration_report.md
Doc/BLFormer_unified_exploration_report.md
```

核心结果:

```text
Output/reproduction/formal/BLCNN_formal_combined_summary.csv
Output/reproduction/formal_other_methods/OtherMethods_paper40_clean_combined_summary.csv
Output/blformer_paper40/blformer_unified_capacity_20260706/
Output/blformer_paper40/blformer_joint_head_20260706/
```
