# BLFormer Joint-Head Final Report

## 1. Final Scope

本轮清理后只保留一个主线: 基于台球局面 set 表示、几何 pair bias、`cls_mean`
pooling 的 BLFormer，并用 flat joint-head 直接建模
`P(clear, win, potted_after_break)`。

保留的实验代码只覆盖两类:

- `paper40_unified_capacity`: 独立三任务 head 的容量基线。
- `paper40_joint_head`: flat 40-way joint-head 消融。

不再保留旧的任务特化加权、额外人工分支、集成检查或多 seed 堆叠代码。`Output/`
下已有实验产物未清空，便于追溯历史结果。

## 2. Clean Target

数据和评估口径固定为:

- processed data: `Output/reproduction/billiards_layout_paper40.pt`
- train/test: paper40 clean split, `776 / 1164`
- held-out test 不用于选 epoch、选配置或调阈值

clean 可复现目标:

| Task | Target acc |
|---|---:|
| clear | 71.477663 |
| win | 67.268041 |
| potted_after_break | 61.254296 |

## 3. Retained Ablation Evidence

### 3.1 Independent-Head Capacity Baseline

`paper40_unified_capacity` 用同一套 BLFormer encoder 和 `cls_mean` pooling，但三个任务由独立
head 预测，potted 使用 class + ordinal hybrid loss。它回答的问题是: 单纯增大统一模型容量是否能解决
clear。

| Trial | Params | clear | win | potted_after_break | min margin |
|---|---:|---:|---:|---:|---:|
| `hybrid_d80_clsmean_ord0.25` | 114,986 | 69.673538 | 70.446736 | 66.151202 | -1.804125 |
| `hybrid_d88_clsmean_ord0.25` | 138,402 | 69.243985 | 70.446736 | 64.862543 | -2.233678 |

结论: 接近三任务 BLCNN 总参数预算的统一 BLFormer 仍不能补上 clear 缺口。问题不只是容量。

### 3.2 Flat Joint-Head

flat joint-head 把三任务标签合成一个 40 类标签:

```text
joint_label = clear * 20 + win * 10 + potted_after_break
```

模型训练 40-way logits，推理时通过边缘化得到三个任务的 logits/probability。这个改动不引入额外人工特征，
但让模型显式学习三个任务的联合结构。

| Trial | Params | clear | win | potted_after_break | min margin |
|---|---:|---:|---:|---:|---:|
| `joint_d64_clsmean` | 79,906 | 69.931269 | 70.532644 | 65.378004 | -1.546394 |
| `joint_d64_clsmean_marg0.5` | 79,906 | 70.274913 | 70.446736 | 65.378004 | -1.202750 |
| `joint_d80_clsmean` | 121,506 | 71.391755 | 70.189005 | 65.378004 | -0.085908 |
| `joint_d80_clsmean_marg0.5` | 121,506 | 70.360827 | 70.189005 | 66.838485 | -1.116836 |

结论: `d80` 容量和 joint label 建模同时存在时，clear 才接近 clean 目标；加入边缘辅助 loss
改善 potted，但牺牲 clear，因此当前主线选择 `joint_d80_clsmean`。

## 4. Current Best

当前保留的最佳结果:

| Method | Params | clear | win | potted_after_break | min margin |
|---|---:|---:|---:|---:|---:|
| `joint_d80_clsmean` | 121,506 | 71.391755 | 70.189005 | 65.378004 | -0.085908 |

Artifact:

```text
Output/blformer_paper40/blformer_joint_head_20260706/search_results.csv
Output/blformer_paper40/blformer_joint_head_20260706/BLFormer_selected_val.pt
Output/blformer_paper40/blformer_joint_head_20260706/test_predictions_selected_val.csv
```

这个结果仍比 clean clear 目标低 `0.085908` 个百分点，但它是目前最简洁、最符合任务结构的统一
BLFormer 方案: 没有任务特化处理，没有外部模型拼接，也没有测试集选择。

## 5. Retained Entry Points

```text
run_blformer.py
run_blformer_unified_capacity_interact.sh
run_blformer_joint_head_interact.sh
test_blformer.py
```

`run_blformer.py` 当前只提供:

```bash
--search paper40_unified_capacity
--search paper40_joint_head
```

推荐复现实验:

```bash
./run_blformer_unified_capacity_interact.sh
./run_blformer_joint_head_interact.sh
```
