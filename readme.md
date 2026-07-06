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

## Script Style

The numbered scripts now follow the `DL Class base` style directly:

- no `argparse`
- edit the configuration values at the top of each file
- run the script itself
- training and evaluation stay separated

## Training Scripts

### 0 - BLFormer Training

Open [0 - BLFormer Training.py](/home/gvlab/Desktop/WANG/MachineLearning/Billiards_Prediction/0%20-%20BLFormer%20Training.py) and edit values such as:

```python
run_name = "joint_d80_clsmean"
potted_head = "class"
use_joint_head = True
joint_marginal_weight = 0.0
output_dir = os.path.join("Output", "blformer_paper40", run_name)
```

Then run:

```bash
python "0 - BLFormer Training.py"
```

If you want another doc setting, edit the BLFormer hyperparameter values at the top of the file directly.

### 1 - BLCNN Training

Edit the top of [1 - BLCNN Training.py](/home/gvlab/Desktop/WANG/MachineLearning/Billiards_Prediction/1%20-%20BLCNN%20Training.py), then run:

```bash
python "1 - BLCNN Training.py"
```

### 2 - MLP Training

Edit the top of [2 - MLP Training.py](/home/gvlab/Desktop/WANG/MachineLearning/Billiards_Prediction/2%20-%20MLP%20Training.py), then run:

```bash
python "2 - MLP Training.py"
```

### 3 - Transformer Training

Edit the top of [3 - Transformer Training.py](/home/gvlab/Desktop/WANG/MachineLearning/Billiards_Prediction/3%20-%20Transformer%20Training.py), then run:

```bash
python "3 - Transformer Training.py"
```

### 4 - Attention Training

Edit the top of [4 - Attention Training.py](/home/gvlab/Desktop/WANG/MachineLearning/Billiards_Prediction/4%20-%20Attention%20Training.py), then run:

```bash
python "4 - Attention Training.py"
```

## Evaluation Scripts

Each evaluation file also uses top-of-file configuration. Set `checkpoint_path`, `output_dir`, `split`, and related values in the script, then run it.

```bash
python "0 - BLFormer Evaluation.py"
python "1 - BLCNN Evaluation.py"
python "2 - MLP Evaluation.py"
python "3 - Transformer Evaluation.py"
python "4 - Attention Evaluation.py"
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
