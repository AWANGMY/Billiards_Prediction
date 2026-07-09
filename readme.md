# Billiards Break Shot Classification

## Project Objective

This project reproduces break-shot outcome prediction in billiards and compares four model families:

- BLFormer
- BLCNN
- MLP
- Transformer
The prediction tasks are:

```text
Clear          clear / not clear after the break shot
Win            win / not win after the break shot
Potted Balls   number of balls potted after the break shot
```

## Dataset Download Instructions

Official Google Drive folder:

```text
https://drive.google.com/drive/folders/1NBqonYLr_cParMMn4xSeE0KTJNhjeYuG
```

Place the downloaded folders as follows:

```text
Baseline/code/               official released code
Dataset/data_layouts/        layout dataset
Dataset/data _trajectories/  trajectory dataset
```

Report-aligned processed dataset used by the BLFormer reproduction entrypoint:

```text
Dataset/processed/billiards_layout_paper40.pt
```

This file contains 1,940 accepted post-break layouts: 776 training samples and
1,164 held-out test samples. The Final Report uses an internal 660/116
train/validation split inside the 776 training samples for BLFormer model
selection.

If the processed file is not available, create it with:

```bash
python ClassesData/PreprocessBilliards.py --root Dataset
```

## Environment Setup

Example setup:

```bash
conda create -n billiards-prediction python=3.13.14
conda activate billiards-prediction
pip install -r requirements.txt
```

## How to Train the Model

Edit the configuration block at the top of the corresponding training script, then run the script.
The default BLFormer training entrypoint is configured for the paper40 fixed-epoch
reproduction setting: `billiards_layout_paper40.pt`, 250 epochs, and
`joint_marginal_weight=0.5`. This approximates the Final Report configuration
without running the internal validation search.

```bash
python "0 - BLFormer Training.py"
python "1 - BLCNN Training.py"
python "2 - MLP Training.py"
python "3 - Transformer Training.py"
```

## How to Run Inference / Evaluation

Edit the configuration block at the top of the corresponding evaluation script, then run the script.

```bash
python "0 - BLFormer Evaluation.py"
python "1 - BLCNN Evaluation.py"
python "2 - MLP Evaluation.py"
python "3 - Transformer Evaluation.py"
```

## Expected Output / Results

Training scripts save outputs such as:

```text
*.pt
*_history.csv
*_train.json
*_training_summary.json
```

Evaluation scripts save outputs such as:

```text
*_evaluation/<split>_metrics.json
*_evaluation/<split>_predictions.csv
```

Reference held-out test results from `Doc/Final_report`:

| Model family | Clear | Win | Potted | Mean |
|---|---:|---:|---:|---:|
| BLCNN baseline | 71.48 | 66.75 | 61.25 | 66.49 |
| MLP baseline | 69.33 | 67.18 | 55.24 | 63.92 |
| Transformer baseline | 65.81 | 66.41 | 40.12 | 57.45 |
| Independent-head BLFormer | 69.42 | 70.45 | 65.81 | 68.56 |
| Joint-outcome BLFormer | 70.96 | 70.19 | 66.92 | 69.36 |
| Joint-outcome BLFormer with auxiliary loss | 71.82 | 69.67 | 67.10 | 69.53 |

The strongest BLFormer row is selected by validation inside the training
partition, not by held-out test performance. The simplified default training
script does not run this search, so its fixed-epoch retrain result may be lower
than the validation-selected checkpoint.
