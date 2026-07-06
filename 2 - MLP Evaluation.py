import os

import torch

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.MLP import MLP
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier
from Utilities.Utilities import Utilities


device = Utilities.resolve_device(allow_cpu=True)
path_parent_project = os.getcwd()
dataset_root = os.path.join(path_parent_project, "Dataset")
processed_path = os.path.join("Output", "reproduction", "billiards_layout_paper40.pt")
checkpoint_path = os.path.join(
    "Output",
    "reproduction",
    "formal_other_methods",
    "paper40_clean_wd0.001",
    "MLP_clear.pt",
)
output_dir = os.path.join(
    "Output",
    "reproduction",
    "formal_other_methods",
    "paper40_clean_wd0.001",
    "MLP_clear_evaluation",
)
split = "test"
batch_size = 64
num_workers = 0

dataset = DatasetLoader(root=dataset_root)
data = dataset.load_processed_data(processed_path=processed_path)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

loader, _, _ = dataset.load_classifier_loader(
    data=data,
    task=checkpoint["metadata"]["task"],
    model_name="MLP",
    indices=checkpoint["metadata"]["splits"][split],
    batch_size=batch_size,
    shuffle=False,
    seed=checkpoint["metadata"]["seed"],
    num_workers=num_workers,
)

model = MLP(hyperparameters=checkpoint["hyperparameters"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
scope = ScopeClassifier(model, checkpoint["hyperparameters"])

trainer = TrainerClassifier(hyperparameters=checkpoint["hyperparameters"])
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
print("accuracy:", Utilities.format_float(metrics["accuracy"]))
print("loss:", Utilities.format_float(metrics["loss"]))
print("metrics_path:", metrics_path)
print("predictions_path:", predictions_path)
