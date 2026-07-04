# Billiards Break Shot Classification

This project reproduces the prediction tasks from the paper `On Predicting and Generating a Good Break Shot in Billiards Sports` and compares BLCNN with MLP, Transformer, and Spatial Attention baselines.

The prediction tasks are:

```text
Clear          clear/not clear after the break shot
Win            win/not win after the break shot
Potted Balls   number of balls potted after the break shot
```

## Project Structure

```text
.
├── .gitignore
├── readme.md
├── setup.md
├── 1 - MLP.py
├── 2 - Transformer.py
├── 3 - BLCNN.py
├── 4 - Attention.py
├── ClassesData/
│   ├── BilliardsDataset.py       # PyTorch Dataset
│   ├── DatasetLoader.py          # load processed tensor data
│   └── PreprocessBilliards.py    # convert raw XML/XLSX files to processed .pt data
├── ClassesML/
│   ├── Block.py                  # shared model blocks
│   ├── CNN.py                    # paper-style BLCNN model
│   ├── MLP.py                    # MLP model
│   ├── Scope.py                  # classification loss and optimizer
│   └── Transformer.py            # Transformer and Spatial Attention models
├── Utilities/
│   └── Utilities.py              # metrics, plots, saved results
├── Dataset/
│   ├── data_layouts/             # raw layout data (git ignored)
│   │   ├── All cordinates/
│   │   └── Variables/
│   ├── data _trajectories/       # raw trajectory data (git ignored)
│   │   └── ...
│   └── processed/
│       └── billiards_layout.pt   # processed tensor dataset
├── Baseline/                     # original baseline code (git ignored)
│   └── code/
│       ├── BLCNN/
│       ├── BLGAN/
│       ├── Preprocessing/
│       └── README.md
└── Output/                       # training outputs (git ignored)
    ├── images/
    └── results/
```

## Preprocessing

Run preprocessing once before training:

```bash
/home/gvlab/Environments/DLClass/bin/python ClassesData/PreprocessBilliards.py --root Dataset
```

The processed file is saved to:

```text
Dataset/processed/billiards_layout.pt
```

The preprocessing now stores two inputs:

```text
x        [N, 10, 4]    normalized layout check features
x_paper  [N, 10, 27]   paper-style token features used by the models
```

`x_paper` follows the baseline BLCNN representation: position token, six pocket angle/distance/path/pocket tokens, best angle, and best pocket for each of the 10 balls.

The default split follows the original baseline code:

```text
test: first 30% of samples
validation: random 10% of the remaining training samples
train: remaining samples
```

Current processed data shape:

```text
samples: 1729
train:   1090
val:     121
test:    518
```

## Model Inputs

All four models use the same paper-style information:

```text
MLP:         [batch, 270]      flattened x_paper as float
BLCNN:       [batch, 10, 27]   x_paper as long token ids
Transformer: [batch, 10, 27]   x_paper as float
Attention:   [batch, 10, 27]   x_paper as float
```

## Models

BLCNN follows the original baseline structure: seven embedding tables, embedding dimension 10, kernel sizes 1 to 10, 3 convolution filters per kernel size, and Adam with `lr=1e-5`, `weight_decay=1e-3`. The baseline README mentions `weight_decay=1e-4`, but the released `new_entry.py` code uses `1e-3`, so this project follows the code.

The comparison models are kept at a similar parameter scale for the binary tasks:

```text
MLP          about 51k parameters
BLCNN        about 47k parameters
Transformer  about 53k parameters
Attention    about 46k parameters
```

## Run

Use the DLClass environment:

```bash
/home/gvlab/Environments/DLClass/bin/python "1 - MLP.py"
/home/gvlab/Environments/DLClass/bin/python "2 - Transformer.py"
/home/gvlab/Environments/DLClass/bin/python "3 - BLCNN.py"
/home/gvlab/Environments/DLClass/bin/python "4 - Attention.py"
```

Each script runs Clear, Win, and Potted Balls. Matplotlib uses `WebAgg`, which is suitable for remote connection.

## Outputs

Figures are saved to:

```text
Output/images/
```

```text
<Model>_<Task>_loss_curve.png
<Model>_<Task>_accuracy_curve.png
<Model>_<Task>_confusion_matrix.png
<Model>_<Task>_prediction_distribution.png
```

CSV results are saved to:

```text
Output/results/
```

```text
<Model>_<Task>_result.csv                 # final metrics and hyperparameters
<Model>_<Task>_predictions.csv            # test set true labels, predicted labels, confidence
<Model>_<Task>_classification_report.csv  # per-class precision, recall, F1, support
```

## Notes

The following folders are currently ignored by git:

```text
Baseline/
Dataset/data_layouts/
Dataset/data _trajectories/
Output/
```
