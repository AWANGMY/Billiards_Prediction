# Billiards Break Shot Classification

## Project Objective

This project reproduces break-shot outcome prediction in billiards and compares five models:

- BLFormer
- BLCNN
- MLP
- Transformer
- Attention

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

Recommended processed dataset used by the current scripts:

```text
Dataset/processed/billiards_layout.pt
```

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

```bash
python "0 - BLFormer Training.py"
python "1 - BLCNN Training.py"
python "2 - MLP Training.py"
python "3 - Transformer Training.py"
python "4 - Attention Training.py"
```

## How to Run Inference / Evaluation

Edit the configuration block at the top of the corresponding evaluation script, then run the script.

```bash
python "0 - BLFormer Evaluation.py"
python "1 - BLCNN Evaluation.py"
python "2 - MLP Evaluation.py"
python "3 - Transformer Evaluation.py"
python "4 - Attention Evaluation.py"
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

Reference results from the current docs:

### Clean baseline reproduction

| Model | Setting | clear | win | potted_after_break |
|---|---|---:|---:|---:|
| BLCNN | `weight_decay=0.0001` | 71.477663 | 66.752577 | 61.254296 |
| MLP | `weight_decay=0.001` | 69.158076 | 67.268041 | 55.068729 |
| Transformer | `weight_decay=0.001` | 65.807560 | 66.408935 | 40.120275 |
| Attention | `weight_decay=0.001` | 65.893471 | 66.408935 | 48.024055 |

### Current BLFormer main result

| Model | Setting | clear | win | potted_after_break |
|---|---|---:|---:|---:|
| BLFormer | `joint_d80_clsmean` | 71.391755 | 70.189005 | 65.378004 |
