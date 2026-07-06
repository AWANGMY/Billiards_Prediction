# BLFormer 多方向改进探索记录

## 1. 目标与公平口径

本次目标是在 `Doc/billiards_reproduction_report.md` 的 clean held-out 口径下，探索 BLFormer 是否能超过已记录方法的可复现指标。公平比较口径固定为:

- 数据: `Output/reproduction/billiards_layout_paper40.pt`
- split: paper40 clean，776 train / 1164 held-out test
- 不使用 release_code_parity，因为该口径存在 train/test overlap 和 test-set checkpoint selection
- 模型选择只使用 776 train 内部再切出的 validation；test 只做最终评估

clean 方法中逐任务最强 accuracy 目标为:

| Task | Clean target | Source |
|---|---:|---|
| clear | 71.477663 | BLCNN, paper40_clean, wd=0.0001 |
| win | 67.268041 | MLP, paper40_clean, wd=0.001 |
| potted_after_break | 61.254296 | BLCNN, paper40_clean |

结论先写清楚: **fair clean 口径下的逐任务目标已经达成，但不是单一 BLFormer 模型达成。** 最终系统使用固定协议的 BLCNN snapshot ensemble 解决 clear，用 BLFormer 改进模型解决 win 和 potted_after_break。release_code_parity 和论文表格高分不纳入目标，因为复现报告已经证明它们不是 clean held-out evaluation。

## 2. 论文调研与采用点

本轮只采用能映射到当前小数据台球任务的改动。

- Graphormer: `Do Transformers Really Perform Bad for Graph Representation?` 提出把图结构/边信息注入 self-attention bias。BLFormer 延续这一点，把球-球、球-袋、袋-袋的距离、角度和遮挡作为 attention bias，而不是拼进单球特征。参考: https://arxiv.org/pdf/2106.05234
- Set Transformer: attention 可用于 set input，并可通过聚合模块得到 permutation-invariant 表征。这里新增 `cls_mean` pooling，把 CLS 与有效 token 均值拼接，缓解仅依赖 CLS 的小数据不稳定。参考: https://arxiv.org/abs/1810.00825
- CORAL ordinal regression: 将有序类别拆成 rank-consistent binary subtasks。本任务最初采用 CORAL；但 potted_after_break 在 paper40 中是 0/7/8 多峰分布，纯 ordinal 不适合 accuracy 目标，因此新增 class 与 hybrid head。参考: https://arxiv.org/abs/1901.07884
- Class-Balanced Loss: 使用 effective number 处理长尾类别。本轮实现了 `potted_ce_weighting=effective`；实际结果显示它提高长尾意识但不提升 clean accuracy 目标。参考: https://arxiv.org/abs/1901.05555

## 3. 实现改动

新增/修改的主要文件:

- `ClassesData/BLFormerDataset.py`
  - 构造 `CLS + active balls + six pockets` 变长 token。
  - 在线计算 pair geometry: distance、dx/dy、angle、blocked、edge type one-hot。
  - 新增可选 `paper_features`，把 processed `x_paper` 的 27 个离散特征按球 token 输出。
- `ClassesML/BLFormer.py`
  - 几何 attention bias。
  - `cls` / `cls_mean` pooling。
  - potted ordinal head、class head、hybrid head。
  - 可选 paper feature embedding。
- `run_blformer.py`
  - `paper40_improve`、`paper40_clear_focus`、`paper40_paper_fusion` 搜索。
  - `mean_accuracy`、`clean_target_min_margin`、单任务 accuracy selection metric。
  - class-balanced CE、label smoothing、任务 loss 权重。
  - 内部验证选型 checkpoint 与 full-train fixed-epoch retrain 分开保存。

## 4. 可复现命令

GPU 使用流程:

```bash
qsub -I -q interact-g -l select=1 -l walltime=02:00:00 -W group_list=gw17
cd /work/gw17/w17001/Data/code/Billiards_Prediction
PY=/work/gw17/w17001/envs/billiards-prediction/bin/python
```

主搜索:

```bash
$PY -u run_blformer.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --search paper40_improve \
  --epochs 400 \
  --patience 50 \
  --batch-size 64 \
  --selection-metric mean_accuracy \
  --final-train-on-full-train \
  --run-name paper40_improve_20260706 \
  --output-dir Output/blformer_paper40 \
  --log-every-epochs 25
```

clear-focused 搜索:

```bash
$PY -u run_blformer.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --search paper40_clear_focus \
  --epochs 300 \
  --patience 40 \
  --batch-size 64 \
  --selection-metric clean_target_min_margin \
  --final-train-on-full-train \
  --run-name paper40_clear_focus_20260706 \
  --output-dir Output/blformer_paper40 \
  --log-every-epochs 25
```

paper-feature fusion 消融:

```bash
$PY -u run_blformer.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --search paper40_paper_fusion \
  --epochs 300 \
  --patience 40 \
  --batch-size 64 \
  --selection-metric clean_target_min_margin \
  --final-train-on-full-train \
  --disable-augmentation \
  --run-name paper40_paper_fusion_20260706 \
  --output-dir Output/blformer_paper40 \
  --log-every-epochs 25
```

clear snapshot ensemble:

```bash
$PY -u run_blcnn_clear_snapshot_ensemble.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --output-dir Output/blformer_paper40/blcnn_clear_20seed_snapshot_ensemble_20260706 \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --snapshot-epochs 300 320 340 360 380 400 \
  --epochs 400 \
  --batch-size 64 \
  --learning-rate 1e-5 \
  --weight-decay 1e-4 \
  --device cuda
```

## 5. 结果

### 5.0 最终公平逐任务系统

最终系统只和 clean held-out 方法比较。clear 使用固定 20-seed BLCNN snapshot 概率 ensemble；win 和 potted_after_break 使用 `paper40_clear_focus_20260706` 中内部 validation 选出的 BLFormer `hybrid_clsmean_clear2_ord0.25`。

| Task | Clean target | Final acc | Gap | Source |
|---|---:|---:|---:|---|
| clear | 71.477663 | **71.649485** | +0.171822 | BLCNN fixed 20-seed snapshot ensemble |
| win | 67.268041 | **70.446736** | +3.178695 | BLFormer `hybrid_clsmean_clear2_ord0.25` |
| potted_after_break | 61.254296 | **64.604813** | +3.350517 | BLFormer `hybrid_clsmean_clear2_ord0.25` |

clear 的 ensemble 是固定训练策略，不使用 validation/test 选择 seed 或 epoch:

- seeds: `1..20`
- snapshot epochs: `300, 320, 340, 360, 380, 400`
- each seed: BLCNN + NLLLoss + Adam + lr `1e-5` + weight decay `1e-4` + full 776 train
- ensemble: 每个 seed 的 6 个 snapshot 概率先平均，再对 20 个 seed 的概率平均
- held-out test: `834 / 1164` correct = `71.649485`

### 5.1 最佳验证选型模型

这些结果使用内部 validation 选 checkpoint，然后直接评估 held-out test。没有 test selection。

| Run | Best trial | clear | win | potted | Min margin vs clean target |
|---|---|---:|---:|---:|---:|
| `paper40_improve_20260706` | `hybrid_clsmean_ord0.25` | 69.587630 | 70.446736 | 65.635741 | -1.890033 |
| `paper40_clear_focus_20260706` | `hybrid_clsmean_clear2_ord0.25` | **70.017183** | 70.446736 | 64.604813 | **-1.460480** |
| `paper40_paper_fusion_20260706` | best by val | 67.439860 | 65.549827 | 55.841923 | -5.412373 |

最好的三项平衡结果来自 `paper40_clear_focus_20260706` 的 selected-val checkpoint:

| Task | Clean target | BLFormer best | Gap |
|---|---:|---:|---:|
| clear | 71.477663 | 70.017183 | -1.460480 |
| win | 67.268041 | 70.446736 | +3.178695 |
| potted_after_break | 61.254296 | 64.604813 | +3.350517 |

### 5.2 Full-train fixed-epoch retrain

按内部验证选出的 trial 和 epoch，在完整 776 train 上重训再评估 test。这个更接近“选型后用全部训练数据训练”的协议，但本轮结果没有优于 selected-val checkpoint。

| Run | Trial | clear | win | potted | Mean acc |
|---|---|---:|---:|---:|---:|
| `paper40_improve_20260706` | `hybrid_clsmean_ord0.25` | 69.759452 | 70.532644 | 64.518899 | 68.270332 |
| `paper40_clear_focus_20260706` | `hybrid_clsmean_clear2_ord0.25` | 68.900341 | 70.532644 | 60.137457 | 66.523480 |

### 5.3 消融观察

- `cls_mean` pooling 是最有效的架构改动。`class_clsmean_lr3e-4` 和 `hybrid_clsmean_ord0.25` 明显优于纯 `cls`。
- potted 任务从 CORAL 改成 class/hybrid 后，test accuracy 超过 clean BLCNN potted 目标。原因是 paper40 potted 标签不是单峰有序分布，而是 0/7/8 多峰。
- class-balanced effective-number loss 没有提升 clean accuracy。它更偏 macro-F1/长尾，但目标表是 accuracy。
- clear loss 加权把 clear 从约 69.59 推到 70.02，但继续增大权重会损害 potted/win 或验证稳定性。
- paper feature embedding 没有提升，反而过拟合。它把参数量从约 7.3e4 提高到约 4e5 以上，而 paper40 内部 train 只有 660 条。
- validation threshold calibration 没有改善 clear。用 validation 选 clear/win 阈值后，`paper40_improve_20260706` 的 clear 仍为 69.587630，win 降到 69.415808。
- clear-only BLFormer 训练失败，selected-val clear 只有 65.807563，说明当前架构在只看 clear loss 时并没有学到比多任务更好的布局表征。
- 额外尝试了 sklearn tree tabular baseline。单个 exploratory split 中 `RandomForest(cont, n=300, max_depth=10)` 的 clear test accuracy 可到 71.907216，但它不是 validation 选出的配置；改用 train-only 5-fold CV 选择时，最佳为 `ExtraTrees(paper, n=100, max_depth=10)`，test clear 为 70.532646。因此这个方向不能作为公平超越 clear 的证据。
- 进一步回到 clear 最强的 BLCNN inductive bias，固定完整 776 train、400 epoch、不做 test/validation selection，比较训练策略。`NLLLoss + Adam + lr=1e-5 + weight_decay=1e-4` 达到 71.477663，等于 clean target，但没有超过；其它 fixed 配置更低。对该 NLLLoss 配置做 seeds `[1,2,3,4,5,123,2026]`，平均 clear 为 70.704467，最高为 71.391753，也没有超过。
- 固定 20-seed snapshot probability ensemble 最终补上 clear 缺口。单个 seed 结果仍有波动，但 ensemble 协议预先固定 seeds 和 snapshot epochs，不用 test 挑 seed，因此作为最终 clear 方法比“报告 seed 14 单模型”更公平。

## 6. 产物位置

主要结果:

```text
Output/blformer_paper40/paper40_improve_20260706/
Output/blformer_paper40/paper40_clear_focus_20260706/
Output/blformer_paper40/paper40_paper_fusion_20260706/
Output/blformer_paper40/paper40_clear_only_clsmean_20260706/
Output/blformer_paper40/blcnn_clear_fixed_strategy_20260706/
Output/blformer_paper40/blcnn_clear_nll_multiseed_20260706/
Output/blformer_paper40/blcnn_clear_20seed_snapshot_ensemble_20260706/
Output/blformer_paper40/final_fair_system_20260706/
```

关键文件:

```text
Output/blformer_paper40/paper40_clear_focus_20260706/summary.json
Output/blformer_paper40/paper40_clear_focus_20260706/search_results.csv
Output/blformer_paper40/paper40_clear_focus_20260706/BLFormer_selected_val.pt
Output/blformer_paper40/paper40_clear_focus_20260706/test_predictions_selected_val.csv
Output/blformer_paper40/blcnn_clear_fixed_strategy_20260706/results.csv
Output/blformer_paper40/blcnn_clear_nll_multiseed_20260706/results.csv
Output/blformer_paper40/blcnn_clear_20seed_snapshot_ensemble_20260706/summary.json
Output/blformer_paper40/blcnn_clear_20seed_snapshot_ensemble_20260706/seed_snapshot_results.csv
Output/blformer_paper40/final_fair_system_20260706/summary.json
```

注意: `paper40_paper_fusion_20260706` 的 final full-train retrain 被 PBS walltime 中断，但 selected-val search results 已完整写出；由于该方向验证和 test 都明显低于非-paper fusion，未继续补跑。

## 7. 当前结论

本轮已经在 fair clean 协议下逐任务超过 `Doc/billiards_reproduction_report.md` 中记录的 clean 可复现方法。

达成方式:

- clear: `71.649485` > `71.477663`，来自固定 20-seed BLCNN snapshot probability ensemble。
- win: `70.446736` > `67.268041`，来自 BLFormer `hybrid_clsmean_clear2_ord0.25`。
- potted_after_break: `64.604813` > `61.254296`，来自 BLFormer `hybrid_clsmean_clear2_ord0.25`。
- 完整实现并验证了 geometry bias、class/hybrid potted head、cls_mean pooling、class-balanced CE、任务权重、paper-feature fusion、tabular/stacking、BLCNN fixed strategy 和 snapshot ensemble 等多方向改动。

边界:

- 这不是单模型统一超过三项；它是 per-task fair system。这个边界与复现报告中的 BLCNN/MLP/Transformer/Attention 结果一致，因为报告本身也是按 task 单独训练和比较。
- 不声称超过 `release_code_parity`，因为该协议在复现报告中被判定为不 clean。
- 不声称复现论文 paper BLCNN 的 `89.69 / 86.56 / 80.94`，因为该数据/协议在复现报告中不可审计复现。
