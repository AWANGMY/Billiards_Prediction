import os

import torch

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.CNN import BLCNN
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier
from Utilities.Utilities import Utilities


TASKS = ["clear", "win", "potted_after_break"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_parent_project = os.getcwd()
dataset_root = os.path.join(path_parent_project, "Dataset")
processed_path = os.path.join(dataset_root, "processed", "billiards_layout.pt")
output_dir = os.path.join("Output", "reproduction", "formal", "paper40_clean_wd0.0001")
seed = 123
batch_size = 64
num_workers = 0
epochs = 400
learning_rate = 1e-5
weight_decay = 1e-4

Utilities.set_seed(seed)

dataset = DatasetLoader(root=dataset_root)
data = dataset.load_processed_data(processed_path=processed_path)
splits = dataset.load_paper40_split(data)
os.makedirs(output_dir, exist_ok=True)

print("device:", device)
print("processed_path:", processed_path)
print("split_sizes:", {"train": len(splits["train"]), "test": len(splits["test"])} )

results = []

for task in TASKS:
    train_loader, input_dim, n_classes = dataset.load_classifier_loader(
        data=data,
        task=task,
        model_name="BLCNN",
        indices=splits["train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )

    hyperparameters = {
        "input_dim": input_dim,
        "output_dim": n_classes,
        "embed_num_pos": 200,
        "embed_num_deg_45": 4,
        "embed_num_deg_90": 7,
        "embed_num_deg_180": 37,
        "embed_num_distance": 25,
        "embed_num_pocket": 6,
        "embed_pocket": 2,
        "embed_dim": 10,
        "kernel_num": 3,
        "kernel_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "dropout_rate": 0.3,
        "criterion": "nll",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": epochs,
        "batch_size": batch_size,
    }

    model = BLCNN(hyperparameters=hyperparameters).to(device)
    scope = ScopeClassifier(model, hyperparameters)

    trainer = TrainerClassifier(hyperparameters=hyperparameters)
    trainer.set_model(model=model, device=device)
    trainer.set_scope(scope=scope)
    trainer.set_data(train_loader=train_loader)
    trainer.run()

    train_metrics, _ = trainer.evaluate(train_loader)

    checkpoint_path = os.path.join(output_dir, "BLCNN_" + task + ".pt")
    history_path = os.path.join(output_dir, "BLCNN_" + task + "_history.csv")
    summary_path = os.path.join(output_dir, "BLCNN_" + task + "_train.json")

    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "hyperparameters": hyperparameters,
            "metadata": {
                "model": "BLCNN",
                "task": task,
                "protocol": "paper40_clean",
                "seed": seed,
                "splits": splits,
                "config": {
                    "processed_path": processed_path,
                    "output_dir": output_dir,
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "epochs": epochs,
                },
            },
        },
        checkpoint_path,
    )

    Utilities.write_csv(history_path, trainer.history)
    Utilities.write_json(
        summary_path,
        {
            "model": "BLCNN",
            "task": task,
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
            "train_accuracy": train_metrics["accuracy"],
            "split_notes": splits["notes"],
        },
    )

    print("saved:", checkpoint_path)
    results.append(
        {
            "model": "BLCNN",
            "task": task,
            "checkpoint_path": checkpoint_path,
            "train_accuracy": train_metrics["accuracy"],
        }
    )

summary_path = os.path.join(output_dir, "BLCNN_training_summary.json")
Utilities.write_json(summary_path, {"results": results})
print("summary:", summary_path)
