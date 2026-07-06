# BLCNN 论文指标复现报告与对照方法补充

## 1. 复现结论

本次复现目标是论文 `On Predicting and Generating a Good Break Shot in Billiards Sports` 中 BLCNN 的 prediction accuracy:

| Task | Paper BLCNN Acc |
|---|---:|
| clear | 89.69 |
| win | 86.56 |
| potted balls | 80.94 |

结论:

- 按论文文字口径实现的 clean held-out evaluation (`paper40_clean`) 没有复现论文指标，三个任务低约 `18-20` 个百分点。
- 补充运行 `paper40_clean` 下的 MLP、Transformer、Attention 后，三种对照方法同样没有接近论文 BLCNN 指标；其中 MLP 是三者里最强的 clean 对照。
- 发布源码行为对照 (`release_code_parity`) 可以超过论文指标，但该口径存在 train/test 重叠和 test-set checkpoint selection，不能作为 clean test accuracy。
- 因此，当前可审计复现结论是: **公开 raw data + clean split 下不能复现论文 BLCNN 指标；发布源码风格结果可复现高准确率，但该高准确率不等价于论文文字描述的干净评估。**

## 2. 数据


可审计预处理产物:

| Item | Value |
|---|---:|
| Processed data | `Output/reproduction/billiards_layout_paper40.pt` |
| SHA256 | `ec68a269fe0597f25b4384446eb1240886bdc5763349c39c7a69ac3c3331d087` |
| accepted samples | 1940 |
| `x_paper` shape | `[1940, 10, 27]` |
| paper40 train / test | `776 / 1164` |
| audit input XML / indexed XML / variables | `2090 / 2052 / 178` |

正式结果文件:

```text
Output/reproduction/formal/BLCNN_formal_combined_summary.csv
Output/reproduction/formal/*/BLCNN_*_result.csv
Output/reproduction/formal/*/BLCNN_*_result.json
Output/reproduction/formal/*/run.log
Output/reproduction/formal_other_methods/OtherMethods_paper40_clean_combined_summary.csv
Output/reproduction/formal_other_methods/paper40_clean_wd*/{MLP,Transformer,Attention}_*_result.csv
Output/reproduction/formal_other_methods/paper40_clean_wd*/{MLP,Transformer,Attention}_*_result.json
Output/reproduction/formal_other_methods/paper40_clean_wd*/run.log
```

## 3. 可复现命令

预处理:

```bash
PY=/work/gw17/w17001/envs/billiards-prediction/bin/python

$PY ClassesData/PreprocessBilliards.py \
  --root Dataset \
  --output Output/reproduction/billiards_layout_paper40.pt \
  --split-method paper40 \
  --audit-json Output/reproduction/preprocess_paper40_audit.json
```

BLCNN 正式矩阵:

```bash
for cfg in \
  "paper40_clean Output/reproduction/billiards_layout_paper40.pt 0.001" \
  "paper40_clean Output/reproduction/billiards_layout_paper40.pt 0.0001" \
  "release_code_parity Output/reproduction/billiards_layout_paper40.pt 0.001" \
  "release_code_parity Output/reproduction/billiards_layout_paper40.pt 0.0001" \
  "current_control Dataset/processed/billiards_layout.pt 0.001"
do
  set -- $cfg
  PROTOCOL=$1
  DATA=$2
  WD=$3
  OUT="Output/reproduction/formal/${PROTOCOL}_wd${WD}"

  $PY -u run_blcnn_reproduction.py \
    --processed-path "$DATA" \
    --protocol "$PROTOCOL" \
    --tasks all \
    --epochs 400 \
    --weight-decay "$WD" \
    --output-dir "$OUT" \
    > "$OUT/run.log" 2>&1
done
```

对照方法补充只运行 `paper40_clean`:

```bash
for WD in 0.001 0.0001
do
  OUT="Output/reproduction/formal_other_methods/paper40_clean_wd${WD}"

  $PY -u run_other_methods_reproduction.py \
    --processed-path Output/reproduction/billiards_layout_paper40.pt \
    --protocol paper40_clean \
    --models MLP Transformer Attention \
    --tasks all \
    --epochs 400 \
    --weight-decay "$WD" \
    --device cuda \
    --output-dir "$OUT" \
    --log-every-epochs 25 \
    > "$OUT/run.log" 2>&1
done
```

## 4. 正式复现结果

`paper40_clean` 是 clean implementation: random 40% train / 60% evaluation，没有 validation selection。

| Protocol | Weight Decay | Task | Paper Acc | Reproduced Acc | Gap |
|---|---:|---|---:|---:|---:|
| paper40_clean | 0.001 | clear | 89.69 | 71.219931 | -18.470069 |
| paper40_clean | 0.001 | win | 86.56 | 66.580756 | -19.979244 |
| paper40_clean | 0.001 | potted_after_break | 80.94 | 61.254296 | -19.685704 |
| paper40_clean | 0.0001 | clear | 89.69 | 71.477663 | -18.212337 |
| paper40_clean | 0.0001 | win | 86.56 | 66.752577 | -19.807423 |
| paper40_clean | 0.0001 | potted_after_break | 80.94 | 61.254296 | -19.685704 |

对照方法补充结果只使用 `paper40_clean`。表中数值为 400 epoch 后的 held-out test accuracy；论文 BLCNN 指标仅作为背景参照，不表示这些方法的论文目标值。

| Weight Decay | Model | clear | win | potted_after_break |
|---:|---|---:|---:|---:|
| 0.001 | MLP | 69.158076 | 67.268041 | 55.068729 |
| 0.001 | Transformer | 65.807560 | 66.408935 | 40.120275 |
| 0.001 | Attention | 65.893471 | 66.408935 | 48.024055 |
| 0.0001 | MLP | 69.329897 | 67.182131 | 55.240550 |
| 0.0001 | Transformer | 65.807560 | 66.408935 | 39.432990 |
| 0.0001 | Attention | 65.893471 | 66.408935 | 48.024055 |

`release_code_parity` 保留发布源码行为: split index 未 remap、训练循环不在每个 batch 前 `optimizer.zero_grad()`、使用 test loader 做 checkpoint selection。该结果可复现，但不是 clean evaluation。

| Protocol | Weight Decay | Task | Paper Acc | Reproduced Acc | Gap | Best Epoch |
|---|---:|---|---:|---:|---:|---:|
| release_code_parity | 0.001 | clear | 89.69 | 94.329897 | +4.639897 | 386 |
| release_code_parity | 0.001 | win | 86.56 | 96.219931 | +9.659931 | 390 |
| release_code_parity | 0.001 | potted_after_break | 80.94 | 94.673540 | +13.733540 | 397 |
| release_code_parity | 0.0001 | clear | 89.69 | 94.501718 | +4.811718 | 388 |
| release_code_parity | 0.0001 | win | 86.56 | 96.048110 | +9.488110 | 381 |
| release_code_parity | 0.0001 | potted_after_break | 80.94 | 94.673540 | +13.733540 | 399 |

旧 processed 数据上的 400 epoch 对照只用于解释欠训练影响，不作为论文数据口径。

| Protocol | Weight Decay | Task | Paper Acc | Reproduced Acc | Gap | Best Epoch |
|---|---:|---|---:|---:|---:|---:|
| current_control | 0.001 | clear | 89.69 | 72.393822 | -17.296178 | 172 |
| current_control | 0.001 | win | 86.56 | 72.007722 | -14.552278 | 28 |
| current_control | 0.001 | potted_after_break | 80.94 | 68.339768 | -12.600232 | 280 |

## 5. 为什么没有复现论文指标

### 数据集不一致

论文/补充材料说明 layout 数据集约为 `3019`。当前公开 raw XML/XLSX 经可审计预处理后只有 `1940` 条 accepted samples。即使关闭 bad remarks 过滤和去重，也只能得到 `1963` 条，无法接近 `3019`。

这说明当前复现没有论文当时使用的精确 processed dataset，只能从公开 raw data 重建近似数据。

### clean 口径下模型过拟合

`paper40_clean` 训练集只有 `776` 条，测试集 `1164` 条。400 epoch 后训练准确率已经很高，但测试准确率仍明显低于论文:

| Task | Train Acc | Test Acc |
|---|---:|---:|
| clear | ~96 | ~71 |
| win | ~93 | ~67 |
| potted_after_break | ~72 | ~61 |

`weight_decay=0.001` 和 `0.0001` 的 clean 结果几乎相同，因此主要问题不是 weight decay，而是数据/split/泛化口径。

对照方法在相同 `paper40_clean` split 下也呈现相同趋势: 两个 weight decay 的 test accuracy 差异很小，MLP 的三个任务分别约为 `69 / 67 / 55`，Transformer 和 Attention 更低。这说明补充的三种方法没有提供能接近论文 BLCNN 指标的 clean 结果。

### 发布源码口径存在泄漏

`release_code_parity` 的高准确率来自发布源码行为对照，不能解释为 clean held-out performance。量化结果:

| Overlap | Count |
|---|---:|
| train/test overlap | `529 / 582` test samples |
| val/test overlap | `53 / 582` test samples |

此外，该口径用 test loader 作为 checkpoint selection 数据。因此它可以稳定超过论文指标，但这个结果不能和论文文字描述的 40% train / 60% evaluation 直接等价。

### 公开材料之间不完全一致

官方 README、论文文字和发布的 `new_entry.py` 不是同一个严格实验规范:

- README 写 prediction task 使用 `epoch = 400`、`weight_decay = 0.0001`。
- 发布源码默认 `weight_decay = 1e-3`。
- 发布源码入口 hard-code `participate_data(..., 1000)`，且 split/evaluation 行为与论文文字描述不一致。
- 发布源码训练循环没有每 batch 清梯度。

这些差异使得论文表格指标无法仅凭公开源码和公开 raw data 唯一还原。

## 6. 验收

BLCNN 正式矩阵共生成 `5` 个配置目录、`15` 条 task 结果。每个配置目录均包含:

| Artifact | Count |
|---|---:|
| `*_result.csv` | 3 |
| `*_result.json` | 3 |
| `*_summary_*.csv` | 1 |
| `run.log` | 1 |

本次对照方法补充只运行 `paper40_clean`，共生成 `2` 个配置目录、`18` 条 task 结果。每个配置目录均包含:

| Artifact | Count |
|---|---:|
| `*_result.csv` | 9 |
| `*_result.json` | 9 |
| `*_summary_*.csv` | 1 |
| `run.log` | 1 |

最终可追踪结论以本文件、`Output/reproduction/formal/BLCNN_formal_combined_summary.csv` 和 `Output/reproduction/formal_other_methods/OtherMethods_paper40_clean_combined_summary.csv` 为准。
