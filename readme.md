# Billiards Break Shot Classification

## Project Objective

This project studies break-shot outcome prediction in billiards. The repository keeps five model entry points:

1. `0 - BLFormer`
2. `1 - BLCNN`
3. `2 - MLP`
4. `3 - Transformer`
5. `4 - Attention`

The three prediction tasks are:

```text
Clear               clear / not clear after the break shot
Win                 win / not win after the break shot
Potted Balls        number of balls potted after the break shot
```

## Dataset Download

Official Google Drive folder:

```text
https://drive.google.com/drive/folders/1NBqonYLr_cParMMn4xSeE0KTJNhjeYuG
```

After downloading, place the folders as follows:

```text
Baseline/code/               official released code
Dataset/data_layouts/        layout dataset
Dataset/data _trajectories/  trajectory dataset
```

## Preprocessing

This repository uses the clean `paper40` processed file from the docs:

```text
Output/reproduction/billiards_layout_paper40.pt
```

Recommended preprocessing command:

```bash
python ClassesData/PreprocessBilliards.py \
  --root Dataset \
  --output Output/reproduction/billiards_layout_paper40.pt \
  --split-method paper40 \
  --audit-json Output/reproduction/preprocess_paper40_audit.json
```

The numbered training and evaluation scripts read the stored `split_indices` from this processed file directly.

## Environment Setup

Example setup:

```bash
conda create -n billiards-prediction python=3.13.14
conda activate billiards-prediction
pip install -r requirements.txt
```

## Training Scripts

The repository now follows the `DL Class base` style more closely:

- each model has one training script and one evaluation script
- training only trains and saves checkpoints / logs
- evaluation only loads a saved checkpoint and reports metrics / predictions

### 0 - BLFormer Training

Recommended main result from the docs:

```bash
python "0 - BLFormer Training.py" \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --experiment joint_d80_clsmean
```

Independent-head baseline from the docs:

```bash
python "0 - BLFormer Training.py" \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --experiment hybrid_d80_clsmean_ord0.25
```

Optional marginal-loss variant:

```bash
python "0 - BLFormer Training.py" \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --experiment joint_d80_clsmean_marg0.5
```

### 1 - BLCNN Training

```bash
python "1 - BLCNN Training.py" \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --weight-decay 0.0001
```

### 2 - MLP Training

```bash
python "2 - MLP Training.py" \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --weight-decay 0.001
```

### 3 - Transformer Training

```bash
python "3 - Transformer Training.py" \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --weight-decay 0.001
```

### 4 - Attention Training

```bash
python "4 - Attention Training.py" \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --weight-decay 0.001
```

## Evaluation Scripts

### 0 - BLFormer Evaluation

```bash
python "0 - BLFormer Evaluation.py" \
  --checkpoint-path Output/blformer_paper40/joint_d80_clsmean/BLFormer_joint_d80_clsmean.pt \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --split test
```

### 1 - BLCNN Evaluation

```bash
python "1 - BLCNN Evaluation.py" \
  --checkpoint-path Output/reproduction/formal/paper40_clean_wd0.0001/BLCNN_clear.pt \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --split test
```

### 2 - MLP Evaluation

```bash
python "2 - MLP Evaluation.py" \
  --checkpoint-path Output/reproduction/formal_other_methods/paper40_clean_wd0.001/MLP_clear.pt \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --split test
```

### 3 - Transformer Evaluation

```bash
python "3 - Transformer Evaluation.py" \
  --checkpoint-path Output/reproduction/formal_other_methods/paper40_clean_wd0.001/Transformer_clear.pt \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --split test
```

### 4 - Attention Evaluation

```bash
python "4 - Attention Evaluation.py" \
  --checkpoint-path Output/reproduction/formal_other_methods/paper40_clean_wd0.001/Attention_clear.pt \
  --processed-path Output/reproduction/billiards_layout_paper40.pt \
  --split test
```

## Saved Outputs

Training scripts save files such as:

```text
*.pt
*_history.csv
*_train.json
*_training_summary.json
```

Evaluation scripts save files such as:

```text
*_evaluation/<split>_metrics.json
*_evaluation/<split>_predictions.csv
```

## Reference Results From The Docs

### Clean `paper40` baseline reproduction

| Model | Setting | clear | win | potted_after_break |
|---|---|---:|---:|---:|
| BLCNN | `weight_decay=0.0001` | 71.477663 | 66.752577 | 61.254296 |
| MLP | `weight_decay=0.001` | 69.158076 | 67.268041 | 55.068729 |
| Transformer | `weight_decay=0.001` | 65.807560 | 66.408935 | 40.120275 |
| Attention | `weight_decay=0.001` | 65.893471 | 66.408935 | 48.024055 |

### Current BLFormer main result

| Experiment | clear | win | potted_after_break |
|---|---:|---:|---:|
| `joint_d80_clsmean` | 71.391755 | 70.189005 | 65.378004 |

## Project Structure

```text
├── readme.md
├── requirements.txt
├── student_information.txt
├── download_official_billiards.py
├── 0 - BLFormer Training.py
├── 0 - BLFormer Evaluation.py
├── 1 - BLCNN Training.py
├── 1 - BLCNN Evaluation.py
├── 2 - MLP Training.py
├── 2 - MLP Evaluation.py
├── 3 - Transformer Training.py
├── 3 - Transformer Evaluation.py
├── 4 - Attention Training.py
├── 4 - Attention Evaluation.py
├── ClassesData/
├── ClassesML/
├── Utilities/
├── Doc/
├── Dataset/
├── Baseline/
└── Output/
```
