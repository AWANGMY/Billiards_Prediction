import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ClassesML.CNN import BLCNN


TASKS = ["clear", "win", "potted_after_break"]
PAPER_TARGET_ACCURACY = {
    "clear": 89.69,
    "win": 86.56,
    "potted_after_break": 80.94,
}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-path",
        default=os.path.join("Dataset", "processed", "billiards_layout.pt"),
    )
    parser.add_argument(
        "--protocol",
        choices=["current_control", "paper40_clean", "release_code_parity"],
        default="current_control",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["all"], help='Task names or "all".'
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=os.path.join("Output", "reproduction"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-best-checkpoint", action="store_true")

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
        raise FileNotFoundError(
            "Processed data not found: "
            + processed_path
            + "\nRun ClassesData/PreprocessBilliards.py first, preferably with official data_layouts."
        )

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


def protocol_weight_decay(protocol, requested_weight_decay):

    if requested_weight_decay is not None:
        return requested_weight_decay

    if protocol == "release_code_parity":
        return 1e-3

    return 1e-3


def build_splits(data, protocol, seed):

    num_data = int(data["x_paper"].shape[0])

    if protocol == "current_control":
        split_indices = data["split_indices"]
        return {
            "train": to_index_list(split_indices["train"]),
            "val": to_index_list(split_indices.get("val", [])),
            "test": to_index_list(split_indices["test"]),
            "notes": ["uses split_indices stored in processed data"],
        }

    if protocol == "paper40_clean":
        rng = np.random.default_rng(seed)
        indices = np.arange(num_data)
        rng.shuffle(indices)
        split = int(np.floor(0.4 * num_data))
        return {
            "train": indices[:split].tolist(),
            "val": [],
            "test": indices[split:].tolist(),
            "notes": [
                "random 40% train and 60% held-out evaluation, matching paper text"
            ],
        }

    if protocol == "release_code_parity":
        test_size = 0.3
        valid_size = 0.1
        split_tt = int(np.floor(test_size * num_data))
        test_indices = list(range(split_tt))

        num_train = num_data - split_tt
        train_positions = list(range(num_train))
        rng = np.random.default_rng(seed)
        rng.shuffle(train_positions)
        split_tv = int(np.floor(valid_size * num_train))

        return {
            "train": train_positions[split_tv:],
            "val": train_positions[:split_tv],
            "test": test_indices,
            "notes": [
                "matches released new_entry.py index behavior",
                "train/val positions are intentionally not remapped through train_idx",
                "test loader is used for checkpoint selection, as in released script call",
            ],
        }

    raise ValueError("Unknown protocol: " + protocol)


def to_index_list(indices):

    if isinstance(indices, torch.Tensor):
        return indices.cpu().long().tolist()

    return list(indices)


def make_loader(x, y, indices, batch_size, shuffle, seed, num_workers):

    if len(indices) == 0:
        return None

    dataset = TensorDataset(x[indices].long(), y[indices].long())
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def hyperparameters(input_dim, output_dim, args, weight_decay):

    return {
        "input_dim": tuple(input_dim),
        "output_dim": int(output_dim),
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
        "learning_rate": args.learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, zero_grad):

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    batch_accuracy_sum = 0.0
    batch_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if zero_grad:
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


def clone_state_dict(model):

    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def run_task(data, task, splits, args, device, weight_decay):

    x = data["x_paper"]
    y = data[task]
    n_classes = int(data.get("n_classes", {}).get(task, int(torch.max(y).item()) + 1))
    hparams = hyperparameters(tuple(x.shape[1:]), n_classes, args, weight_decay)

    train_loader = make_loader(
        x, y, splits["train"], args.batch_size, True, args.seed, args.num_workers
    )
    val_loader = make_loader(
        x, y, splits["val"], args.batch_size, False, args.seed, args.num_workers
    )
    test_loader = make_loader(
        x, y, splits["test"], args.batch_size, False, args.seed, args.num_workers
    )

    model = BLCNN(hyperparameters=hparams).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=weight_decay
    )

    zero_grad = args.protocol != "release_code_parity"
    selection_loader = val_loader
    selection_name = "val"
    if args.protocol == "release_code_parity":
        selection_loader = test_loader
        selection_name = "test_release_parity"

    best_state = None
    best_epoch = None
    best_selection_accuracy = None
    final_train_metrics = None
    final_selection_metrics = None

    for epoch in range(1, args.epochs + 1):
        final_train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, zero_grad
        )

        should_eval = (
            selection_loader is not None
            and args.eval_every_epochs > 0
            and (epoch % args.eval_every_epochs == 0 or epoch == args.epochs)
        )

        if should_eval:
            final_selection_metrics = evaluate(
                model, selection_loader, criterion, device
            )
            selection_accuracy = final_selection_metrics["accuracy"]
            if (
                best_selection_accuracy is None
                or selection_accuracy > best_selection_accuracy
            ):
                best_selection_accuracy = selection_accuracy
                best_epoch = epoch
                best_state = clone_state_dict(model)

        print(
            format_progress(
                task,
                epoch,
                args.epochs,
                final_train_metrics,
                final_selection_metrics,
                selection_name,
            )
        )

    final_test_metrics = evaluate(model, test_loader, criterion, device)
    best_test_metrics = final_test_metrics

    if best_state is not None:
        model.load_state_dict(best_state)
        best_test_metrics = evaluate(model, test_loader, criterion, device)

    if args.save_best_checkpoint and best_state is not None:
        checkpoint_path = os.path.join(
            args.output_dir, "BLCNN_" + args.protocol + "_" + task + "_best.pt"
        )
        torch.save(
            {
                "model_state_dict": best_state,
                "hyperparameters": hparams,
                "task": task,
                "protocol": args.protocol,
                "best_epoch": best_epoch,
            },
            checkpoint_path,
        )
    else:
        checkpoint_path = None

    primary_test_metrics = (
        best_test_metrics if best_state is not None else final_test_metrics
    )
    paper_target = PAPER_TARGET_ACCURACY[task]

    return {
        "model": "BLCNN",
        "task": task,
        "protocol": args.protocol,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": weight_decay,
        "zero_grad_each_batch": zero_grad,
        "selection_metric": selection_name if selection_loader is not None else "none",
        "best_epoch": best_epoch,
        "best_selection_accuracy": best_selection_accuracy,
        "final_train_loss": final_train_metrics["loss"],
        "final_train_accuracy": final_train_metrics["accuracy"],
        "final_test_loss": final_test_metrics["loss"],
        "final_test_accuracy": final_test_metrics["accuracy"],
        "final_test_batch_mean_accuracy": final_test_metrics["batch_mean_accuracy"],
        "final_test_release_denominator_accuracy": final_test_metrics[
            "release_denominator_accuracy"
        ],
        "best_test_loss": best_test_metrics["loss"],
        "best_test_accuracy": best_test_metrics["accuracy"],
        "best_test_batch_mean_accuracy": best_test_metrics["batch_mean_accuracy"],
        "best_test_release_denominator_accuracy": best_test_metrics[
            "release_denominator_accuracy"
        ],
        "reported_test_accuracy": primary_test_metrics["accuracy"],
        "paper_target_accuracy": paper_target,
        "accuracy_gap_to_paper": primary_test_metrics["accuracy"] - paper_target,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "sample_count": int(x.shape[0]),
        "checkpoint_path": checkpoint_path,
        "hyperparameters": hparams,
        "protocol_notes": splits["notes"],
    }


def format_progress(
    task, epoch, epochs, train_metrics, selection_metrics, selection_name
):

    message = (
        task
        + " epoch "
        + str(epoch)
        + "/"
        + str(epochs)
        + " train_acc="
        + format_float(train_metrics["accuracy"])
    )
    if selection_metrics is not None:
        message += (
            " " + selection_name + "_acc=" + format_float(selection_metrics["accuracy"])
        )
    return message


def format_float(value):

    if value is None:
        return "NA"

    return "{:.4f}".format(value)


def write_task_result(result, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir, "BLCNN_" + result["protocol"] + "_" + result["task"] + "_result.csv"
    )
    json_path = os.path.join(
        output_dir,
        "BLCNN_" + result["protocol"] + "_" + result["task"] + "_result.json",
    )

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
        output_dir, "BLCNN_" + protocol + "_summary_" + timestamp + ".csv"
    )

    fieldnames = [
        "model",
        "task",
        "protocol",
        "seed",
        "epochs",
        "weight_decay",
        "reported_test_accuracy",
        "paper_target_accuracy",
        "accuracy_gap_to_paper",
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

    weight_decay = protocol_weight_decay(args.protocol, args.weight_decay)
    data = load_processed_data(args.processed_path)
    splits = build_splits(data, args.protocol, args.seed)
    tasks = selected_tasks(args.tasks)

    os.makedirs(args.output_dir, exist_ok=True)

    print("device:", device)
    print("processed_path:", args.processed_path)
    print("protocol:", args.protocol)
    print(
        "split_sizes:", {name: len(splits[name]) for name in ["train", "val", "test"]}
    )
    print("protocol_notes:", "; ".join(splits["notes"]))

    results = []
    for task in tasks:
        result = run_task(data, task, splits, args, device, weight_decay)
        csv_path, json_path = write_task_result(result, args.output_dir)
        print("saved:", csv_path)
        print("saved:", json_path)
        results.append(result)

    summary_path = write_summary(results, args.output_dir, args.protocol)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
