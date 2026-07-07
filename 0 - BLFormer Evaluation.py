import os

import torch

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.BLFormer import BLFormer
from ClassesML.Scope import ScopeBLFormer
from ClassesML.TrainerClassifier import TrainerBLFormer
from Utilities.Utilities import Utilities


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_root = "Dataset"
processed_path = os.path.join(dataset_root, "processed", "billiards_layout_paper40.pt")
checkpoint_path = os.path.join(
    "Output",
    "blformer_paper40",
    "joint_d80_clsmean_marg0.5_fixed250",
    "BLFormer_joint_d80_clsmean_marg0.5_fixed250.pt",
)
output_dir = os.path.join(
    "Output",
    "blformer_paper40",
    "joint_d80_clsmean_marg0.5_fixed250",
    "BLFormer_joint_d80_clsmean_marg0.5_fixed250_evaluation",
)
split = "test"
batch_size = 64
num_workers = 0

dataset = DatasetLoader(root=dataset_root)
data = dataset.load_processed_data(processed_path=processed_path)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

loader = dataset.load_blformer_loader(
    data=data,
    indices=checkpoint["metadata"]["splits"][split],
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    augment=False,
)

model = BLFormer(hyperparameters=checkpoint["hyperparameters"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
scope = ScopeBLFormer(model, checkpoint["hyperparameters"])

trainer = TrainerBLFormer(hyperparameters=checkpoint["hyperparameters"])
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
metrics, rows = trainer.evaluate(loader)

os.makedirs(output_dir, exist_ok=True)
metrics_path = os.path.join(output_dir, split + "_metrics.json")
predictions_path = os.path.join(output_dir, split + "_predictions.csv")

Utilities.write_json(
    metrics_path,
    {
        "checkpoint_path": checkpoint_path,
        "processed_path": processed_path,
        "split": split,
        "metrics": metrics,
        "metadata": checkpoint["metadata"],
    },
)
Utilities.write_csv(predictions_path, rows)

print("checkpoint_path:", checkpoint_path)
print("processed_path:", processed_path)
print("split:", split)
print("mean_accuracy:", Utilities.format_float(metrics["mean_accuracy"]))
print("mean_macro_f1:", Utilities.format_float(metrics["mean_macro_f1"]))
print("clear_accuracy:", Utilities.format_float(metrics["task_metrics"]["clear"]["accuracy"]))
print("win_accuracy:", Utilities.format_float(metrics["task_metrics"]["win"]["accuracy"]))
print(
    "potted_after_break_accuracy:",
    Utilities.format_float(metrics["task_metrics"]["potted_after_break"]["accuracy"]),
)
print("metrics_path:", metrics_path)
print("predictions_path:", predictions_path)
