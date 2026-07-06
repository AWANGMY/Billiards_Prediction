import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ClassesML.MLP import MLP
from ClassesML.Transformer import SpatialAttention, Transformer


TASKS = ["clear", "win", "potted_after_break"]
MODELS = ["MLP", "Transformer", "Attention"]
PAPER_BLCNN_TARGET_ACCURACY = {
    "clear": 89.69,
    "win": 86.56,
    "potted_after_break": 80.94,
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-path",
        default=os.path.join("Output", "reproduction", "billiards_layout_paper40.pt"),
    )
    parser.add_argument(
        "--protocol", choices=["paper40_clean"], default="paper40_clean"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help='Model names or "all". Choices: MLP, Transformer, Attention.',
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["all"], help='Task names or "all".'
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("Output", "reproduction", "formal_other_methods"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every-epochs", type=int, default=25)

    return parser.parse_args()


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_processed_data(processed_path):

    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed data not found: " + processed_path)

    try:
        return torch.load(processed_path, weights_only=False)
    except TypeError:
        return torch.load(processed_path)


def selected_tasks(task_args):

    if task_args == ["all"]:
        return TASKS

    unknown = [task for task in task_args if task not in TASKS]
    if len(unknown) > 0:
        raise ValueError("Unknown tasks: " + ", ".join(unknown))

    return task_args


def selected_models(model_args):

    if model_args == ["all"]:
        return MODELS

    unknown = [model for model in model_args if model not in MODELS]
    if len(unknown) > 0:
        raise ValueError("Unknown models: " + ", ".join(unknown))

    return model_args


def build_paper40_clean_splits(data, seed):

    num_data = int(data["x_paper"].shape[0])
    rng = np.random.default_rng(seed)
    indices = np.arange(num_data)
    rng.shuffle(indices)
    split = int(np.floor(0.4 * num_data))

    return {
        "train": indices[:split].tolist(),
        "val": [],
        "test": indices[split:].tolist(),
        "notes": ["random 40% train and 60% held-out evaluation, matching paper text"],
    }


def model_input(data, model_name):

    if model_name == "MLP":
        return data["x_paper"].float().reshape(data["x_paper"].shape[0], -1)

    if model_name in ["Transformer", "Attention"]:
        return data["x_paper"].float()

    raise ValueError("Unknown model: " + model_name)


def make_loader(x, y, indices, batch_size, shuffle, seed, num_workers):

    if len(indices) == 0:
        return None

    dataset = TensorDataset(x[indices], y[indices].long())
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def hyperparameters(model_name, input_dim, output_dim, args):

    base = {
        "input_dim": tuple(input_dim),
        "output_dim": int(output_dim),
        "dropout_rate": 0.3,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
    }

    if model_name == "MLP":
        base.update(
            {
                "hidden_layers_sizes": [128, 128],
                "activation": "relu",
                "batch_normalization": False,
            }
        )
        return base

    if model_name == "Transformer":
        base.update(
            {
                "embedding_dim": 56,
                "num_heads": 4,
                "num_layers": 2,
                "expansion_factor": 2,
                "activation": "relu",
            }
        )
        return base

    if model_name == "Attention":
        base.update(
            {
                "embedding_dim": 144,
                "hidden_layers_sizes": [144],
                "activation": "relu",
            }
        )
        return base

    raise ValueError("Unknown model: " + model_name)


def build_model(model_name, hparams):

    if model_name == "MLP":
        return MLP(hyperparameters=hparams)

    if model_name == "Transformer":
        return Transformer(hyperparameters=hparams)

    if model_name == "Attention":
        return SpatialAttention(hyperparameters=hparams)

    raise ValueError("Unknown model: " + model_name)


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    batch_accuracy_sum = 0.0
    batch_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        batch_size = int(y.size(0))
        predicted = torch.argmax(y_hat, dim=1)
        correct = int((predicted == y).sum().item())

        total_loss += float(loss.item()) * batch_size
        total_correct += correct
        total_count += batch_size
        batch_accuracy_sum += correct / batch_size * 100.0
        batch_count += 1

    return metric_dict(
        total_loss,
        total_correct,
        total_count,
        batch_accuracy_sum,
        batch_count,
        loader.batch_size,
    )


def evaluate(model, loader, criterion, device):

    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    batch_accuracy_sum = 0.0
    batch_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            batch_size = int(y.size(0))
            predicted = torch.argmax(y_hat, dim=1)
            correct = int((predicted == y).sum().item())

            total_loss += float(loss.item()) * batch_size
            total_correct += correct
            total_count += batch_size
            batch_accuracy_sum += correct / batch_size * 100.0
            batch_count += 1

    return metric_dict(
        total_loss,
        total_correct,
        total_count,
        batch_accuracy_sum,
        batch_count,
        loader.batch_size,
    )


def metric_dict(
    total_loss, total_correct, total_count, batch_accuracy_sum, batch_count, batch_size
):

    release_denominator = batch_count * batch_size

    return {
        "loss": total_loss / total_count if total_count > 0 else None,
        "accuracy": total_correct / total_count * 100.0 if total_count > 0 else None,
        "batch_mean_accuracy": (
            batch_accuracy_sum / batch_count if batch_count > 0 else None
        ),
        "release_denominator_accuracy": (
            total_correct / release_denominator * 100.0
            if release_denominator > 0
            else None
        ),
        "correct": total_correct,
        "count": total_count,
        "batches": batch_count,
    }


def run_task(data, model_name, task, splits, args, device):

    x = model_input(data, model_name)
    y = data[task]
    n_classes = int(data.get("n_classes", {}).get(task, int(torch.max(y).item()) + 1))
    hparams = hyperparameters(model_name, tuple(x.shape[1:]), n_classes, args)

    train_loader = make_loader(
        x, y, splits["train"], args.batch_size, True, args.seed, args.num_workers
    )
    test_loader = make_loader(
        x, y, splits["test"], args.batch_size, False, args.seed, args.num_workers
    )

    model = build_model(model_name, hparams).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    final_train_metrics = None
    for epoch in range(1, args.epochs + 1):
        final_train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        should_log = args.log_every_epochs > 0 and (
            epoch % args.log_every_epochs == 0 or epoch == args.epochs
        )
        if should_log:
            print(
                format_progress(
                    model_name, task, epoch, args.epochs, final_train_metrics
                )
            )

    final_test_metrics = evaluate(model, test_loader, criterion, device)
    paper_target = PAPER_BLCNN_TARGET_ACCURACY[task]

    return {
        "model": model_name,
        "task": task,
        "protocol": args.protocol,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "zero_grad_each_batch": True,
        "selection_metric": "none",
        "best_epoch": None,
        "best_selection_accuracy": None,
        "final_train_loss": final_train_metrics["loss"],
        "final_train_accuracy": final_train_metrics["accuracy"],
        "final_test_loss": final_test_metrics["loss"],
        "final_test_accuracy": final_test_metrics["accuracy"],
        "final_test_batch_mean_accuracy": final_test_metrics["batch_mean_accuracy"],
        "final_test_release_denominator_accuracy": final_test_metrics[
            "release_denominator_accuracy"
        ],
        "best_test_loss": final_test_metrics["loss"],
        "best_test_accuracy": final_test_metrics["accuracy"],
        "best_test_batch_mean_accuracy": final_test_metrics["batch_mean_accuracy"],
        "best_test_release_denominator_accuracy": final_test_metrics[
            "release_denominator_accuracy"
        ],
        "reported_test_accuracy": final_test_metrics["accuracy"],
        "paper_blcnn_target_accuracy": paper_target,
        "accuracy_gap_to_paper_blcnn": final_test_metrics["accuracy"] - paper_target,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "sample_count": int(data["x_paper"].shape[0]),
        "checkpoint_path": None,
        "hyperparameters": hparams,
        "protocol_notes": splits["notes"],
    }


def format_progress(model_name, task, epoch, epochs, train_metrics):

    return (
        model_name
        + " "
        + task
        + " epoch "
        + str(epoch)
        + "/"
        + str(epochs)
        + " train_acc="
        + format_float(train_metrics["accuracy"])
    )


def format_float(value):

    if value is None:
        return "NA"

    return "{:.4f}".format(value)


def write_task_result(result, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    prefix = result["model"] + "_" + result["protocol"] + "_" + result["task"]
    csv_path = os.path.join(output_dir, prefix + "_result.csv")
    json_path = os.path.join(output_dir, prefix + "_result.json")

    flat_result = {}
    for key, value in result.items():
        if isinstance(value, (dict, list, tuple)):
            flat_result[key] = json.dumps(value, sort_keys=True)
        else:
            flat_result[key] = value

    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(flat_result.keys()))
        writer.writeheader()
        writer.writerow(flat_result)

    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)

    return csv_path, json_path


def write_summary(results, output_dir, protocol):

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        output_dir, "OtherMethods_" + protocol + "_summary_" + timestamp + ".csv"
    )

    fieldnames = [
        "model",
        "task",
        "protocol",
        "seed",
        "epochs",
        "weight_decay",
        "reported_test_accuracy",
        "paper_blcnn_target_accuracy",
        "accuracy_gap_to_paper_blcnn",
        "best_epoch",
        "best_selection_accuracy",
        "train_size",
        "val_size",
        "test_size",
        "sample_count",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fieldnames})

    return summary_path


def main():

    args = parse_args()
    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = load_processed_data(args.processed_path)
    splits = build_paper40_clean_splits(data, args.seed)
    tasks = selected_tasks(args.tasks)
    models = selected_models(args.models)

    os.makedirs(args.output_dir, exist_ok=True)

    print("device:", device)
    print("processed_path:", args.processed_path)
    print("protocol:", args.protocol)
    print("models:", models)
    print(
        "split_sizes:", {name: len(splits[name]) for name in ["train", "val", "test"]}
    )
    print("protocol_notes:", "; ".join(splits["notes"]))

    results = []
    for model_name in models:
        for task in tasks:
            result = run_task(data, model_name, task, splits, args, device)
            csv_path, json_path = write_task_result(result, args.output_dir)
            print("saved:", csv_path)
            print("saved:", json_path)
            results.append(result)

    summary_path = write_summary(results, args.output_dir, args.protocol)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
