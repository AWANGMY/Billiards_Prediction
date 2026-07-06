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
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--processed-path",
        default=os.path.join("Output", "reproduction", "billiards_layout_paper40.pt"),
    )
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)

    return parser.parse_args()


def main():

    args = parse_args()
    device = Utilities.resolve_device(args.device, allow_cpu=True)

    path_parent_project = os.getcwd()
    dataset = DatasetLoader(root=os.path.join(path_parent_project, "Dataset"))
    data = dataset.load_processed_data(processed_path=args.processed_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    split_indices = checkpoint["metadata"]["splits"][args.split]
    loader = dataset.load_blformer_loader(
        data=data,
        indices=split_indices,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        augment=False,
    )

    model = BLFormer(hyperparameters=checkpoint["hyperparameters"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    scope = ScopeBLFormer(model, checkpoint["hyperparameters"])

    trainer = TrainerBLFormer(hyperparameters=checkpoint["hyperparameters"])
    trainer.set_model(model=model, device=device)
    trainer.set_scope(scope=scope)
    metrics, rows = trainer.evaluate(loader)

    output_dir = args.output_dir
    if output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        output_dir = os.path.join(checkpoint_dir, checkpoint_name + "_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, args.split + "_metrics.json")
    predictions_path = os.path.join(output_dir, args.split + "_predictions.csv")

    Utilities.write_json(
        metrics_path,
        {
            "checkpoint_path": args.checkpoint_path,
            "processed_path": args.processed_path,
            "split": args.split,
            "metrics": metrics,
            "metadata": checkpoint["metadata"],
        },
    )
    Utilities.write_csv(predictions_path, rows)

    print("checkpoint_path:", args.checkpoint_path)
    print("processed_path:", args.processed_path)
    print("split:", args.split)
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


if __name__ == "__main__":
    main()
