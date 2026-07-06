# Billiards Break Shot Classification

## Project Objective

This project studies break-shot outcome prediction in billiards. The main goals are:

1. Reproduce the prediction tasks from the paper `On Predicting and Generating a Good Break Shot in Billiards Sports`.
2. Compare several models, including BLCNN, MLP, Transformer, Spatial Attention, and BLFormer.
3. Analyze the gap between reproduced results and the results reported in the papers.

The three prediction tasks are:

```text
Clear               clear / not clear after the break shot
Win                 win / not win after the break shot
Potted Balls        number of balls potted after the break shot
```

## Dataset Download Instructions

Official Google Drive folder:

```text
https://drive.google.com/drive/folders/1NBqonYLr_cParMMn4xSeE0KTJNhjeYuG
```

You can download the data in either of the following ways.

### Option 1: Manual Download

Place the downloaded folders into the project as follows:

```text
Baseline/code/               official released code
Dataset/data_layouts/        layout dataset
Dataset/data _trajectories/  trajectory dataset
```

If needed, also create the output folders:

```text
Output/images/
Output/results/
```


### Preprocessing

After the raw dataset is placed in the correct folders, preprocess it with:

```bash
python ClassesData/PreprocessBilliards.py --root Dataset
```

The processed file is expected at:

```text
Dataset/processed/billiards_layout.pt
```

## Environment Setup

Example setup:

```bash
conda create -n billiards-prediction python=3.13.14
conda activate billiards-prediction
pip install torch==2.11.0 torchvision==0.26.0 numpy==2.4.4 matplotlib==3.10.8 scikit-learn==1.8.0 seaborn==0.13.2
```

The same information is also recorded in `requirements.txt`.

## How to Train the Model

To be completed.

## How to Run Inference / Evaluation

To be completed.

## Expected Output / Results

To be completed.

## Project Structure

```text
├── readme.md
├── requirements.txt
├── setup.md
├── download_official_billiards.py
├── run_blcnn_reproduction.py
├── run_other_methods_reproduction.py
├── run_blformer.py
├── ClassesData/
├── ClassesML/
├── Utilities/
├── Doc/
├── Dataset/
├── Baseline/
└── Output/
```
