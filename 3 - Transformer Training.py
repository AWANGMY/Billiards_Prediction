import argparse
import os

import torch

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerClassifier import TrainerClassifier
from ClassesML.Transformer import Transformer
from Utilities.Utilities import Utilities


TASKS = ["clear", "win", "potted_after_break"]


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-path",
        default=os.path.join("Output", "reproduction", "billiards_layout_paper40.pt"),
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def main():

    args = parse_args()
    Utilities.set_seed(args.seed)

    device = Utilities.resolve_device(args.device, allow_cpu=True)
    path_parent_project = os.getcwd()
    dataset = DatasetLoader(root=os.path.join(path_parent_project, "Dataset"))
    data = dataset.load_processed_data(processed_path=args.processed_path)
    splits = dataset.load_paper40_split(data)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "Output",
            "reproduction",
            "formal_other_methods",
            "paper40_clean_wd" + str(args.weight_decay),
        )
    os.makedirs(output_dir, exist_ok=True)

    print("device:", device)
    print("processed_path:", args.processed_path)
    print("split_sizes:", {"train": len(splits["train"]), "test": len(splits["test"])})

    results = []

    for task in TASKS:
        train_loader, input_dim, n_classes = dataset.load_classifier_loader(
            data=data,
            task=task,
            model_name="Transformer",
            indices=splits["train"],
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            num_workers=args.num_workers,
        )

        hyperparameters = {
            "input_dim": input_dim,
            "output_dim": n_classes,
            "embedding_dim": 56,
            "num_heads": 4,
            "num_layers": 2,
            "expansion_factor": 2,
            "activation": "relu",
            "dropout_rate": 0.3,
            "criterion": "cross_entropy",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
        }

        model = Transformer(hyperparameters=hyperparameters).to(device)
        scope = ScopeClassifier(model, hyperparameters)

        trainer = TrainerClassifier(hyperparameters=hyperparameters)
        trainer.set_model(model=model, device=device)
        trainer.set_scope(scope=scope)
        trainer.set_data(train_loader=train_loader)
        trainer.run()

        train_metrics, _ = trainer.evaluate(train_loader)

        checkpoint_path = os.path.join(output_dir, "Transformer_" + task + ".pt")
        history_path = os.path.join(output_dir, "Transformer_" + task + "_history.csv")
        summary_path = os.path.join(output_dir, "Transformer_" + task + "_train.json")

        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "hyperparameters": hyperparameters,
                "metadata": {
                    "model": "Transformer",
                    "task": task,
                    "protocol": "paper40_clean",
                    "seed": args.seed,
                    "splits": splits,
                    "args": vars(args),
                },
            },
            checkpoint_path,
        )

        Utilities.write_csv(history_path, trainer.history)
        Utilities.write_json(
            summary_path,
            {
                "model": "Transformer",
                "task": task,
                "protocol": "paper40_clean",
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
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
                "model": "Transformer",
                "task": task,
                "checkpoint_path": checkpoint_path,
                "train_accuracy": train_metrics["accuracy"],
            }
        )

    summary_path = os.path.join(output_dir, "Transformer_training_summary.json")
    Utilities.write_json(summary_path, {"results": results})
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
