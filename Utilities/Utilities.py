import csv
import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class Utilities:

    @staticmethod
    def get_activation(activation_str: Optional[str]):

        if activation_str == "relu":
            return nn.ReLU()
        elif activation_str == "sigmoid":
            return nn.Sigmoid()
        elif activation_str == "tanh":
            return nn.Tanh()
        elif activation_str == "linear":
            return None
        else:
            raise ValueError("Unknown activation function: " + str(activation_str))

    @staticmethod
    def compute_accuracy(y, y_hat):

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = torch.tensor(y_hat)

        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0) * 100.0

        return accuracy

    @staticmethod
    def set_seed(seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def resolve_device(device_name=None, allow_cpu=False):

        if device_name is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if allow_cpu:
                return torch.device("cpu")
            raise RuntimeError("CUDA is not available.")

        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            if allow_cpu:
                return torch.device("cpu")
            raise RuntimeError("CUDA is not available.")

        return device

    @staticmethod
    def format_float(value):

        if value is None:
            return "NA"

        return "{:.4f}".format(value)

    @staticmethod
    def json_safe(value):

        if isinstance(value, dict):
            return {str(key): Utilities.json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [Utilities.json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [Utilities.json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)

        return value

    @staticmethod
    def write_json(path, data):

        directory = os.path.dirname(path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(Utilities.json_safe(data), output_file, indent=2)

    @staticmethod
    def write_csv(path, rows):

        if len(rows) == 0:
            return

        directory = os.path.dirname(path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
