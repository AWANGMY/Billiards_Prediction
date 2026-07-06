import os

import torch

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.BLFormer import BLFormer
from ClassesML.Scope import ScopeBLFormer
from ClassesML.TrainerClassifier import TrainerBLFormer
from Utilities.Utilities import Utilities


device = Utilities.resolve_device(allow_cpu=True)
path_parent_project = os.getcwd()
dataset_root = os.path.join(path_parent_project, "Dataset")
processed_path = os.path.join("Output", "reproduction", "billiards_layout_paper40.pt")
run_name = "joint_d80_clsmean"
output_dir = os.path.join("Output", "blformer_paper40", run_name)
seed = 123
batch_size = 64
num_workers = 0
epochs = 200
learning_rate = 3e-4
weight_decay = 1e-2
augment_train = True
potted_head = "class"
use_joint_head = True
joint_marginal_weight = 0.0
potted_ordinal_weight = 0.25

Utilities.set_seed(seed)

dataset = DatasetLoader(root=dataset_root)
data = dataset.load_processed_data(processed_path=processed_path)
splits = dataset.load_paper40_split(data)
os.makedirs(output_dir, exist_ok=True)

train_loader = dataset.load_blformer_loader(
    data=data,
    indices=splits["train"],
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    augment=augment_train,
)
train_eval_loader = dataset.load_blformer_loader(
    data=data,
    indices=splits["train"],
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    augment=False,
)

hyperparameters = {
    "d_model": 80,
    "num_heads": 4,
    "num_layers": 2,
    "ffn_dim": 160,
    "dropout_rate": 0.1,
    "pair_feature_dim": 10,
    "bias_hidden_dim": 32,
    "num_token_types": 4,
    "num_ball_ids": 11,
    "num_potted_thresholds": 9,
    "num_potted_classes": 10,
    "pooling": "cls_mean",
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "max_epochs": epochs,
    "batch_size": batch_size,
    "clear_weight": 1.0,
    "win_weight": 1.0,
    "potted_weight": 1.0,
    "potted_head": potted_head,
    "use_joint_head": use_joint_head,
    "joint_marginal_weight": joint_marginal_weight,
    "potted_ordinal_weight": potted_ordinal_weight,
}

model = BLFormer(hyperparameters=hyperparameters).to(device)
scope = ScopeBLFormer(model, hyperparameters)

trainer = TrainerBLFormer(hyperparameters=hyperparameters)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data(train_loader=train_loader)
trainer.run()

train_metrics, _ = trainer.evaluate(train_eval_loader)

checkpoint_path = os.path.join(output_dir, "BLFormer_" + run_name + ".pt")
history_path = os.path.join(output_dir, "BLFormer_" + run_name + "_history.csv")
summary_path = os.path.join(output_dir, "BLFormer_" + run_name + "_train.json")

torch.save(
    {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "hyperparameters": hyperparameters,
        "metadata": {
            "model": "BLFormer",
            "experiment": run_name,
            "protocol": "paper40_clean",
            "seed": seed,
            "splits": splits,
            "config": {
                "processed_path": processed_path,
                "output_dir": output_dir,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "epochs": epochs,
                "augment_train": augment_train,
            },
        },
    },
    checkpoint_path,
)

Utilities.write_csv(history_path, trainer.history)
Utilities.write_json(
    summary_path,
    {
        "model": "BLFormer",
        "experiment": run_name,
        "protocol": "paper40_clean",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_size": len(splits["train"]),
        "test_size": len(splits["test"]),
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "hyperparameters": hyperparameters,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["mean_accuracy"],
        "train_macro_f1": train_metrics["mean_macro_f1"],
        "clear_accuracy": train_metrics["task_metrics"]["clear"]["accuracy"],
        "win_accuracy": train_metrics["task_metrics"]["win"]["accuracy"],
        "potted_after_break_accuracy": train_metrics["task_metrics"]["potted_after_break"]["accuracy"],
        "split_notes": splits["notes"],
    },
)

print("device:", device)
print("processed_path:", processed_path)
print("split_sizes:", {"train": len(splits["train"]), "test": len(splits["test"])})
print("checkpoint_path:", checkpoint_path)
print("summary_path:", summary_path)
