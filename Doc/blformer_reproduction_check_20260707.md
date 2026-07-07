# BLFormer 当前代码复现检查记录

## 1. 检查目标

本次检查目标是确认整理后的当前代码库，是否还能复现
`Doc/billiards_blformer_experiment_report.md` 中记录的 BLFormer 结果。

原则:

- 不使用整理前已有的 `Output/blformer_paper40/...` artifact 作为复现依据。
- 不修改训练代码。
- 只通过交互式 GPU 运行当前代码，结果写入新的复现目录。
- 复现记录写入本文件。

结论:

**当前代码可以完整跑通 GPU 训练、验证选择、held-out test 评估和 full-train retrain 流程；但不能严格逐数值复现旧报告。** 主要差异体现在:

- `paper40_unified_capacity` 当前搜索空间和 trial 顺序已经不同，best trial 变为新增的 `hybrid_d64_clsmean_ord0.25`。
- `paper40_joint_head` 当前结果整体不差，best selected-val trial 变为 `joint_d80_clsmean_marg0.5`，但旧报告推荐的 `joint_d80_clsmean` 三项 accuracy 未逐项一致。
- joint-head 当前参数量与旧报告不一致，例如当前 `joint_d80_clsmean` 为 `121426` 参数，旧报告为 `121506`，说明整理后模型配置已经发生过小幅变化。

## 2. 环境与命令

交互式 GPU 申请:

```bash
qsub -I -q interact-g -l select=1 -l walltime=02:00:00 -W group_list=gw17
cd /work/gw17/w17001/Data/code/Billiards_Prediction
PY=/work/gw17/w17001/envs/billiards-prediction/bin/python
```

运行环境:

| Item | Value |
|---|---|
| node | `mg0016` |
| GPU | `NVIDIA GH200 120GB` |
| driver | `595.58.03` |
| CUDA reported by nvidia-smi | `13.2` |
| torch | `2.11.0+cu130` |
| `torch.cuda.is_available()` | `True` |
| git commit | `c2d8159ab3f14e1b902529fe0d62987ff3d156cb` |

当时工作区已有未跟踪文件:

```text
?? run_blformer_joint_head_interact.sh
?? run_blformer_unified_capacity_interact.sh
?? test_blformer.py
```

代码健康检查:

```bash
$PY test_blformer.py
```

结果:

```text
Ran 8 tests in 0.811s
OK
```

## 3. 数据与 split

两组实验均使用:

```text
Output/reproduction/billiards_layout_paper40.pt
```

运行时打印的 split:

| Split | Count |
|---|---:|
| internal train | 660 |
| internal validation | 116 |
| final train | 776 |
| held-out test | 1164 |

split notes:

```text
re-splits stored train+val with validation ratio 0.15
uses stored test indices as held-out test set
```

## 4. 复现产物

新的复现输出目录:

```text
Output/reproduction_check_20260707/blformer_paper40/
```

两组核心结果:

```text
Output/reproduction_check_20260707/blformer_paper40/blformer_unified_capacity_current_20260707/
Output/reproduction_check_20260707/blformer_paper40/blformer_joint_head_current_20260707/
```

关键文件:

```text
.../search_results.csv
.../summary.json
.../history.csv
.../interact.log
.../test_predictions_selected_val.csv
.../test_predictions_final_full_train.csv
```

## 5. Independent-Head Capacity 重跑

命令:

```bash
$PY -u run_blformer.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --search paper40_unified_capacity \
  --epochs 400 \
  --patience 60 \
  --batch-size 64 \
  --selection-metric clean_target_min_margin \
  --final-train-on-full-train \
  --run-name blformer_unified_capacity_current_20260707 \
  --output-dir Output/reproduction_check_20260707/blformer_paper40 \
  --log-every-epochs 25
```

当前代码实际搜索空间为 3 个 trial:

| Trial | Params | Best epoch | clear | win | potted_after_break | Mean acc |
|---|---:|---:|---:|---:|---:|---:|
| `hybrid_d64_clsmean_ord0.25` | 74682 | 153 | 69.415808 | 70.446736 | 65.807563 | 68.556702 |
| `hybrid_d80_clsmean_ord0.25` | 114986 | 115 | 69.415808 | 70.103091 | 62.371135 | 67.296678 |
| `hybrid_d88_clsmean_ord0.25` | 138402 | 80 | 69.845361 | 70.532644 | 65.034366 | 68.470790 |

当前 selected-val best trial:

```text
hybrid_d64_clsmean_ord0.25
selected_val_test_mean_accuracy = 68.556702
```

当前 final-full-train 结果:

```text
trial = hybrid_d64_clsmean_ord0.25
epochs = 153
final_full_train_test_mean_accuracy = 68.127147
clear / win / potted = 69.673538 / 70.532644 / 64.175260
```

与旧报告中表 5.1 的同名 trial 对比:

| Trial | Metric | Old report | Current rerun | Delta |
|---|---|---:|---:|---:|
| `hybrid_d80_clsmean_ord0.25` | clear | 69.673538 | 69.415808 | -0.257730 |
| `hybrid_d80_clsmean_ord0.25` | win | 70.446736 | 70.103091 | -0.343645 |
| `hybrid_d80_clsmean_ord0.25` | potted | 66.151202 | 62.371135 | -3.780067 |
| `hybrid_d80_clsmean_ord0.25` | mean | 68.757159 | 67.296678 | -1.460481 |
| `hybrid_d88_clsmean_ord0.25` | clear | 69.243985 | 69.845361 | +0.601376 |
| `hybrid_d88_clsmean_ord0.25` | win | 70.446736 | 70.532644 | +0.085908 |
| `hybrid_d88_clsmean_ord0.25` | potted | 64.862543 | 65.034366 | +0.171823 |
| `hybrid_d88_clsmean_ord0.25` | mean | 68.184421 | 68.470790 | +0.286369 |

判断:

- `hybrid_d88_clsmean_ord0.25` 与旧报告同量级，且当前均值略高。
- `hybrid_d80_clsmean_ord0.25` 未复现旧报告结果，主要差异在 `potted_after_break`，低了约 `3.78` 个百分点。
- 当前脚本新增/保留了 `d64` trial，并且它被选为 best trial；因此整体 selected result 与旧报告不一致。

## 6. Flat Joint-Head 重跑

命令:

```bash
$PY -u run_blformer.py \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --search paper40_joint_head \
  --epochs 400 \
  --patience 60 \
  --batch-size 64 \
  --selection-metric clean_target_min_margin \
  --final-train-on-full-train \
  --run-name blformer_joint_head_current_20260707 \
  --output-dir Output/reproduction_check_20260707/blformer_paper40 \
  --log-every-epochs 25
```

当前代码实际搜索空间为 4 个 trial:

| Trial | Params | Best epoch | clear | win | potted_after_break | Mean acc |
|---|---:|---:|---:|---:|---:|---:|
| `joint_d64_clsmean` | 79842 | 319 | 70.618558 | 70.360827 | 66.065294 | 69.014893 |
| `joint_d64_clsmean_marg0.5` | 79842 | 264 | 70.790380 | 70.360827 | 66.065294 | 69.072167 |
| `joint_d80_clsmean` | 121426 | 240 | 70.962197 | 70.189005 | 66.924399 | 69.358534 |
| `joint_d80_clsmean_marg0.5` | 121426 | 259 | 71.821308 | 69.673538 | 67.096221 | 69.530356 |

当前 selected-val best trial:

```text
joint_d80_clsmean_marg0.5
selected_val_test_mean_accuracy = 69.530356
clear / win / potted = 71.821308 / 69.673538 / 67.096221
```

当前 final-full-train 结果:

```text
trial = joint_d80_clsmean_marg0.5
epochs = 259
final_full_train_test_mean_accuracy = 68.785795
clear / win / potted = 70.103091 / 69.673538 / 66.580755
```

与旧报告中表 5.2 的同名 trial 对比:

| Trial | Metric | Old report | Current rerun | Delta |
|---|---|---:|---:|---:|
| `joint_d64_clsmean` | clear | 69.931269 | 70.618558 | +0.687289 |
| `joint_d64_clsmean` | win | 70.532644 | 70.360827 | -0.171817 |
| `joint_d64_clsmean` | potted | 65.378004 | 66.065294 | +0.687290 |
| `joint_d64_clsmean` | mean | 68.613972 | 69.014893 | +0.400921 |
| `joint_d64_clsmean_marg0.5` | clear | 70.274913 | 70.790380 | +0.515467 |
| `joint_d64_clsmean_marg0.5` | win | 70.446736 | 70.360827 | -0.085909 |
| `joint_d64_clsmean_marg0.5` | potted | 65.378004 | 66.065294 | +0.687290 |
| `joint_d64_clsmean_marg0.5` | mean | 68.699884 | 69.072167 | +0.372283 |
| `joint_d80_clsmean` | clear | 71.391755 | 70.962197 | -0.429558 |
| `joint_d80_clsmean` | win | 70.189005 | 70.189005 | +0.000000 |
| `joint_d80_clsmean` | potted | 65.378004 | 66.924399 | +1.546395 |
| `joint_d80_clsmean` | mean | 68.986255 | 69.358534 | +0.372279 |
| `joint_d80_clsmean_marg0.5` | clear | 70.360827 | 71.821308 | +1.460481 |
| `joint_d80_clsmean_marg0.5` | win | 70.189005 | 69.673538 | -0.515467 |
| `joint_d80_clsmean_marg0.5` | potted | 66.838485 | 67.096221 | +0.257736 |
| `joint_d80_clsmean_marg0.5` | mean | 69.129439 | 69.530356 | +0.400917 |

判断:

- 旧报告推荐的 `joint_d80_clsmean = 71.391755 / 70.189005 / 65.378004` 没有严格复现；当前为 `70.962197 / 70.189005 / 66.924399`。
- 当前 `joint_d80_clsmean_marg0.5` 是本轮 selected-val best，mean accuracy 为 `69.530356`，高于旧报告中所有 joint-head 行的 mean accuracy。
- 当前 selected-val best 在三项 clean target 上均为正 margin:
  - clear: `71.821308 > 71.477663`
  - win: `69.673538 > 67.268041`
  - potted_after_break: `67.096221 > 61.254296`
- 但它牺牲了一部分 `win`，相对旧报告 `joint_d80_clsmean` 低 `0.515467` 个百分点。

## 7. 差异原因判断

本次没有修改训练代码。根据当前运行日志、当前 `run_blformer.py` 行为和旧报告数值，严格复现失败更像是代码整理后实验定义发生了变化，而不是 GPU 环境或数据文件不可用。

已观察到的差异:

- `paper40_unified_capacity` 当前实际跑 3 个 trial，并且包含 `hybrid_d64_clsmean_ord0.25`；旧报告表 5.1 只记录 d80 和 d88。
- `run_blformer.py` 中 dataloader shuffle seed 使用 `args.seed + trial_index`。当 trial 顺序变化时，同名 trial 的训练数据 shuffle 也会变化，这是同名配置数值漂移的一个直接来源。
- joint-head 当前参数量与旧报告不一致:
  - 当前 `joint_d64_clsmean`: `79842`; 旧报告: `79906`
  - 当前 `joint_d80_clsmean`: `121426`; 旧报告: `121506`
- 当前 summary 中 BLFormer hyperparameters 使用 `num_token_types: 4`；整理前旧 artifact 的 joint-head summary 曾包含更多可选 feature 开关和不同参数量。这说明模型实现或默认配置已经不是同一个可执行版本。

## 8. 最终判断

按“逐数值复现旧报告”的标准:

```text
不能严格复现。
```

按“当前代码是否还能完整运行同一协议并得到同量级结论”的标准:

```text
可以复现训练流程，并且 joint-head 当前结果整体不弱于旧报告。
```

当前代码下更合适的 BLFormer selected-val 主结果应记录为:

```text
joint_d80_clsmean_marg0.5
clear / win / potted_after_break = 71.821308 / 69.673538 / 67.096221
mean accuracy = 69.530356
```

如果需要严格复现 `Doc/billiards_blformer_experiment_report.md` 中的旧数值，需要找回整理前对应版本的 `run_blformer.py`、`ClassesML/BLFormer.py` 和相关 dataset/model 配置，或在当前代码中显式恢复旧搜索空间、trial 顺序、模型参数量和 seed 行为后再跑。
