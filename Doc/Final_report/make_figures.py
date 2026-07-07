import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"

DATA_PATH = ROOT / "Output" / "reproduction" / "billiards_layout_paper40.pt"
RUN_ROOT = ROOT / "Doc" / "reproduction_check_20260707" / "blformer_paper40"
UNIFIED_DIR = RUN_ROOT / "blformer_unified_capacity_current_20260707"
JOINT_DIR = RUN_ROOT / "blformer_joint_head_current_20260707"
BASELINE_BLCNN_PATH = ROOT / "Output" / "reproduction" / "formal" / "BLCNN_formal_combined_summary.csv"
BASELINE_OTHER_PATH = ROOT / "Output" / "reproduction" / "formal_other_methods" / "OtherMethods_paper40_clean_combined_summary.csv"

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def split_internal_indices(data, seed=123, val_ratio=0.15):
    stored = data["split_indices"]
    pool = torch.cat([stored["train"], stored.get("val", torch.tensor([], dtype=torch.long))])
    pool_list = pool.long().cpu().numpy().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(pool_list)
    val_size = int(np.floor(val_ratio * len(pool_list)))
    return {
        "train": np.array(pool_list[val_size:], dtype=np.int64),
        "val": np.array(pool_list[:val_size], dtype=np.int64),
        "final_train": np.array(pool_list, dtype=np.int64),
        "test": stored["test"].long().cpu().numpy(),
    }


def counts(values, minlength):
    return np.bincount(np.asarray(values, dtype=np.int64), minlength=minlength).astype(int)


def shorten_trial(name):
    formal_names = {
        "hybrid_d64_clsmean_ord0.25": "Ind-64",
        "hybrid_d80_clsmean_ord0.25": "Ind-80",
        "hybrid_d88_clsmean_ord0.25": "Ind-88",
        "joint_d64_clsmean": "Joint-64",
        "joint_d64_clsmean_marg0.5": "Joint-64+A",
        "joint_d80_clsmean": "Joint-80",
        "joint_d80_clsmean_marg0.5": "Joint-80+A",
    }
    return formal_names.get(name, name)


def trial_rows():
    rows = []
    for family, run_dir in [("Hybrid", UNIFIED_DIR), ("Joint", JOINT_DIR)]:
        for row in read_csv(run_dir / "search_results.csv"):
            rows.append({
                "family": family,
                "trial": row["trial_name"],
                "short": shorten_trial(row["trial_name"]),
                "params": int(row["parameter_count"]),
                "epoch": int(row["best_epoch"]),
                "mean": float(row["test_mean_accuracy"]),
                "clear": float(row["test_clear_accuracy"]),
                "win": float(row["test_win_accuracy"]),
                "potted": float(row["test_potted_after_break_accuracy"]),
                "f1": float(row["test_mean_macro_f1"]),
            })
    return rows


def baseline_rows():
    rows = []
    all_rows = []
    if BASELINE_BLCNN_PATH.exists():
        all_rows.extend(read_csv(BASELINE_BLCNN_PATH))
    if BASELINE_OTHER_PATH.exists():
        all_rows.extend(read_csv(BASELINE_OTHER_PATH))

    grouped = {}
    for row in all_rows:
        if row.get("protocol") != "paper40_clean":
            continue
        model = row["model"]
        if model not in ["BLCNN", "MLP", "Transformer"]:
            continue
        key = (model, row.get("config", model))
        grouped.setdefault(key, {})[row["task"]] = float(row["reported_test_accuracy"])

    by_model = {}
    for (model, config), metrics in grouped.items():
        if not all(task in metrics for task in ["clear", "win", "potted_after_break"]):
            continue
        item = {
            "model": model,
            "config": config,
            "clear": metrics["clear"],
            "win": metrics["win"],
            "potted": metrics["potted_after_break"],
        }
        item["mean"] = (item["clear"] + item["win"] + item["potted"]) / 3.0
        if model not in by_model or item["mean"] > by_model[model]["mean"]:
            by_model[model] = item

    for model in ["BLCNN", "MLP", "Transformer"]:
        if model in by_model:
            rows.append(by_model[model])
    return rows


def comparison_rows(rows):
    baselines = baseline_rows()
    independent = max([row for row in rows if row["family"] == "Hybrid"],
                      key=lambda row: row["mean"])
    joint = max([row for row in rows if row["family"] == "Joint"],
                key=lambda row: row["mean"])
    return baselines + [
        {"model": "Ind. BLFormer",
         "clear": independent["clear"],
         "win": independent["win"],
         "potted": independent["potted"],
         "mean": independent["mean"]},
        {"model": "Joint BLFormer",
         "clear": joint["clear"],
         "win": joint["win"],
         "potted": joint["potted"],
         "mean": joint["mean"]},
    ]


def confusion(true_values, pred_values, num_classes):
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, pred in zip(true_values, pred_values):
        mat[int(y), int(pred)] += 1
    return mat


def plot_dashboard(data, splits, rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25))
    fig.subplots_adjust(hspace=0.45, wspace=0.32)

    ax = axes[0, 0]
    binary_labels = ["clear=0", "clear=1", "win=0", "win=1"]
    binary_counts = [
        int((data["clear"] == 0).sum().item()),
        int((data["clear"] == 1).sum().item()),
        int((data["win"] == 0).sum().item()),
        int((data["win"] == 1).sum().item()),
    ]
    ax.bar(binary_labels, binary_counts,
           color=["#62748e", "#2f9e8f", "#b15b5b", "#d6a13a"])
    ax.set_title("(a) Binary outcome balance")
    ax.set_ylabel("samples")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    potted_counts = counts(data["potted_after_break"].cpu().numpy(), 10)
    ax.bar(np.arange(10), potted_counts, color="#4c78a8")
    ax.set_title("(b) Potted balls distribution")
    ax.set_xlabel("balls potted after break")
    ax.set_ylabel("samples")
    ax.set_xticks(np.arange(10))
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    compare = comparison_rows(rows)
    x = np.arange(len(compare))
    colors = ["#9aa5b1", "#9aa5b1", "#9aa5b1", "#4c78a8", "#f58518"]
    ax.bar(x, [row["mean"] for row in compare], color=colors)
    ax.set_title("(c) Baselines vs. BLFormer")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(55, 71)
    ax.set_xticks(x)
    ax.set_xticklabels([row["model"] for row in compare], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    best_specs = [
        (UNIFIED_DIR / "history.csv", "hybrid_d64_clsmean_ord0.25", "Independent", "#4c78a8"),
        (JOINT_DIR / "history.csv", "joint_d80_clsmean_marg0.5", "Joint + auxiliary", "#f58518"),
    ]
    for path, trial, label, color in best_specs:
        hist = [row for row in read_csv(path) if row["trial_name"] == trial]
        epochs = [int(row["epoch"]) for row in hist]
        values = [float(row["val_mean_accuracy"]) for row in hist]
        ax.plot(epochs, values, label=label, color=color, linewidth=1.4)
    ax.set_title("(d) Internal validation trajectory")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(FIG_DIR / "dataset_results_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(pred_rows):
    y_clear = np.array([int(row["clear_true"]) for row in pred_rows])
    p_clear = np.array([int(row["clear_pred"]) for row in pred_rows])
    y_win = np.array([int(row["win_true"]) for row in pred_rows])
    p_win = np.array([int(row["win_pred"]) for row in pred_rows])
    y_potted = np.array([int(row["potted_after_break_true"]) for row in pred_rows])
    p_potted = np.array([int(row["potted_after_break_pred"]) for row in pred_rows])

    mats = [
        ("clear", confusion(y_clear, p_clear, 2)),
        ("win", confusion(y_win, p_win, 2)),
        ("potted_after_break", confusion(y_potted, p_potted, 10)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.75]})
    fig.subplots_adjust(wspace=0.45)

    for ax, (title, mat) in zip(axes, mats):
        normalized = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_yticks(np.arange(mat.shape[0]))
        if mat.shape[0] <= 2:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            color="white" if normalized[i, j] > 0.5 else "#202020",
                            fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="row-normalized")
    fig.savefig(FIG_DIR / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "clear_confusion": mats[0][1].tolist(),
        "win_confusion": mats[1][1].tolist(),
        "potted_confusion": mats[2][1].tolist(),
        "potted_mae": float(np.mean(np.abs(y_potted - p_potted))),
        "potted_within_one": float(np.mean(np.abs(y_potted - p_potted) <= 1) * 100.0),
        "potted_pred_counts": counts(p_potted, 10).tolist(),
    }


def main():
    data = torch.load(DATA_PATH, weights_only=False)
    splits = split_internal_indices(data)
    rows = trial_rows()
    selected_predictions = read_csv(JOINT_DIR / "test_predictions_selected_val.csv")

    plot_dashboard(data, splits, rows)
    confusion_stats = plot_confusion(selected_predictions)

    split_counts = {name: int(len(indices)) for name, indices in splits.items()}
    label_stats = {
        "clear": counts(data["clear"].cpu().numpy(), 2).tolist(),
        "win": counts(data["win"].cpu().numpy(), 2).tolist(),
        "potted_after_break": counts(data["potted_after_break"].cpu().numpy(), 10).tolist(),
        "potted_when_break": counts(data["potted_when_break"].cpu().numpy(), 10).tolist(),
        "active_balls_mean": float(data["x"][:, :, 2].sum(dim=1).float().mean().item()),
        "active_balls_min": int(data["x"][:, :, 2].sum(dim=1).min().item()),
        "active_balls_max": int(data["x"][:, :, 2].sum(dim=1).max().item()),
    }

    stats = {
        "data_path": str(DATA_PATH.relative_to(ROOT)),
        "num_samples": int(data["x"].shape[0]),
        "split_method": data["split_method"],
        "stored_split_counts": {k: int(len(v)) for k, v in data["split_indices"].items()},
        "internal_split_counts": split_counts,
        "label_stats": label_stats,
        "trials": rows,
        "baselines": baseline_rows(),
        "unified_summary": load_json(UNIFIED_DIR / "summary.json"),
        "joint_summary": load_json(JOINT_DIR / "summary.json"),
        "selected_joint_error_stats": confusion_stats,
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(json.dumps({
        "figures": [
            str((FIG_DIR / "dataset_results_dashboard.pdf").relative_to(ROOT)),
            str((FIG_DIR / "confusion_matrices.pdf").relative_to(ROOT)),
        ],
        "stats": str((OUT_DIR / "stats.json").relative_to(ROOT)),
        "best_joint_mean_accuracy": stats["joint_summary"]["best_trial"]["test_metrics"]["mean_accuracy"],
        "potted_mae": confusion_stats["potted_mae"],
        "potted_within_one": confusion_stats["potted_within_one"],
    }, indent=2))


if __name__ == "__main__":
    main()
