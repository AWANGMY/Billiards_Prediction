import argparse
import os

import torch

from ClassesData.DatasetLoader import DatasetLoader
from ClassesML.BLFormer import BLFormer
from ClassesML.Scope import ScopeBLFormer
from ClassesML.TrainerClassifier import TrainerBLFormer
from Utilities.Utilities import Utilities


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-path",
        default=os.path.join("Output", "reproduction", "billiards_layout_paper40.pt"),
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "joint_d80_clsmean",
            "joint_d80_clsmean_marg0.5",
            "hybrid_d80_clsmean_ord0.25",
        ],
        default="joint_d80_clsmean",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-augmentation", action="store_true")

    return parser.parse_args()


def experiment_hyperparameters(args):

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
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "clear_weight": 1.0,
        "win_weight": 1.0,
        "potted_weight": 1.0,
    }

    if args.experiment == "joint_d80_clsmean":
        hyperparameters["potted_head"] = "class"
        hyperparameters["use_joint_head"] = True
        hyperparameters["joint_marginal_weight"] = 0.0
    elif args.experiment == "joint_d80_clsmean_marg0.5":
        hyperparameters["potted_head"] = "class"
        hyperparameters["use_joint_head"] = True
        hyperparameters["joint_marginal_weight"] = 0.5
    elif args.experiment == "hybrid_d80_clsmean_ord0.25":
        hyperparameters["potted_head"] = "hybrid"
        hyperparameters["use_joint_head"] = False
        hyperparameters["potted_ordinal_weight"] = 0.25
    else:
        raise ValueError("Unknown experiment: " + str(args.experiment))

    return hyperparameters


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
        output_dir = os.path.join("Output", "blformer_paper40", args.experiment)
    os.makedirs(output_dir, exist_ok=True)

    train_loader = dataset.load_blformer_loader(
        data=data,
        indices=splits["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        augment=not args.disable_augmentation,
    )
    train_eval_loader = dataset.load_blformer_loader(
        data=data,
        indices=splits["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augment=False,
    )

    hyperparameters = experiment_hyperparameters(args)
    model = BLFormer(hyperparameters=hyperparameters).to(device)
    scope = ScopeBLFormer(model, hyperparameters)

    trainer = TrainerBLFormer(hyperparameters=hyperparameters)
    trainer.set_model(model=model, device=device)
    trainer.set_scope(scope=scope)
    trainer.set_data(train_loader=train_loader)
    trainer.run()

    train_metrics, _ = trainer.evaluate(train_eval_loader)

    checkpoint_path = os.path.join(output_dir, "BLFormer_" + args.experiment + ".pt")
    history_path = os.path.join(output_dir, "BLFormer_" + args.experiment + "_history.csv")
    summary_path = os.path.join(output_dir, "BLFormer_" + args.experiment + "_train.json")

    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "hyperparameters": hyperparameters,
            "metadata": {
                "model": "BLFormer",
                "experiment": args.experiment,
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
            "model": "BLFormer",
            "experiment": args.experiment,
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
            "train_accuracy": train_metrics["mean_accuracy"],
            "train_macro_f1": train_metrics["mean_macro_f1"],
            "clear_accuracy": train_metrics["task_metrics"]["clear"]["accuracy"],
            "win_accuracy": train_metrics["task_metrics"]["win"]["accuracy"],
            "potted_after_break_accuracy": train_metrics["task_metrics"]["potted_after_break"]["accuracy"],
            "split_notes": splits["notes"],
        },
    )

    print("device:", device)
    print("processed_path:", args.processed_path)
    print("split_sizes:", {"train": len(splits["train"]), "test": len(splits["test"])})
    print("checkpoint_path:", checkpoint_path)
    print("summary_path:", summary_path)


if __name__ == "__main__":
    main()
