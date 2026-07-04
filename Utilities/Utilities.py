import os
import csv
import json
import numpy as np
from typing import Optional

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import torch
import torch.nn as nn
from torchvision.utils import make_grid


class Utilities:

    @staticmethod
    def get_activation(activation_str: Optional[str]):

        if activation_str == 'relu':
            return nn.ReLU()
        elif activation_str == 'sigmoid':
            return nn.Sigmoid()
        elif activation_str == 'tanh':
            return nn.Tanh()
        elif activation_str == "linear":
            return None
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")

    @staticmethod
    def make_output_dirs():

        image_path = os.path.join("Output", "images")
        result_path = os.path.join("Output", "results")

        os.makedirs(image_path, exist_ok=True)
        os.makedirs(result_path, exist_ok=True)

        return image_path, result_path

    @staticmethod
    def images_as_canvas(images, title: str = ""):

        canvas = make_grid(images.cpu(), padding=10, nrow=10, normalize=True)
        canvas = canvas.permute(1, 2, 0).numpy() * 255
        canvas = canvas.astype("uint8")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(title)
        plt.show()

    @staticmethod
    def compute_accuracy(y, y_hat):

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        if not isinstance(y_hat, torch.Tensor):
            y_hat = torch.tensor(y_hat)

        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0) * 100

        return accuracy

    @staticmethod
    def plot_training_curves(train_loss_list, valid_loss_list,
                             train_accuracy_list, valid_accuracy_list,
                             model_name, show=False):

        image_path, _ = Utilities.make_output_dirs()

        epochs = range(1, len(train_loss_list) + 1)

        plt.figure()
        plt.plot(epochs, train_loss_list, "b", label="Train Loss")
        plt.plot(epochs, valid_loss_list, "r", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(model_name + " Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, model_name + "_loss_curve.png"), dpi=300)

        plt.figure()
        plt.plot(epochs, train_accuracy_list, "b", label="Train Accuracy")
        plt.plot(epochs, valid_accuracy_list, "r", label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(model_name + " Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, model_name + "_accuracy_curve.png"), dpi=300)

        if show:
            plt.show()

    @staticmethod
    def plot_confusion_matrix_billiards(y, y_hat, model_name, n_classes=10, show=False):

        image_path, _ = Utilities.make_output_dirs()

        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().detach().numpy()

        y_pred = np.argmax(y_hat, axis=1)
        accuracy = np.mean(y_pred == y) * 100

        labels = list(range(n_classes))
        cm = confusion_matrix(y, y_pred, labels=labels)
        cm_normalized = np.divide(cm.astype("float"),
                                  cm.sum(axis=1, keepdims=True),
                                  out=np.zeros_like(cm, dtype=float),
                                  where=cm.sum(axis=1, keepdims=True) != 0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(model_name + " Confusion Matrix - Accuracy: " + str(round(accuracy, 2)))
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, model_name + "_confusion_matrix.png"), dpi=300)

        if show:
            plt.show()

    @staticmethod
    def save_training_result(model_name, hyperparameters,
                             train_loss_list, valid_loss_list,
                             train_accuracy_list, valid_accuracy_list,
                             test_loss, test_accuracy):

        _, result_path = Utilities.make_output_dirs()

        result_file = os.path.join(result_path, model_name + "_result.csv")

        result = {
            "model": model_name,
            "final_train_loss": train_loss_list[-1],
            "final_valid_loss": valid_loss_list[-1],
            "final_train_accuracy": train_accuracy_list[-1],
            "final_valid_accuracy": valid_accuracy_list[-1],
            "best_valid_accuracy": max(valid_accuracy_list),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "hyperparameters": json.dumps(Utilities.convert_hyperparameters(hyperparameters))
        }

        with open(result_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            writer.writeheader()
            writer.writerow(result)

    @staticmethod
    def save_predictions(model_name, y, y_hat):

        _, result_path = Utilities.make_output_dirs()

        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().detach().numpy()

        y_pred = np.argmax(y_hat, axis=1)
        y_hat = y_hat - np.max(y_hat, axis=1, keepdims=True)
        y_prob = np.exp(y_hat) / np.sum(np.exp(y_hat), axis=1, keepdims=True)
        confidence = np.max(y_prob, axis=1)

        prediction_file = os.path.join(result_path, model_name + "_predictions.csv")

        with open(prediction_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "y_true", "y_pred", "confidence"])
            for i in range(len(y)):
                writer.writerow([i, int(y[i]), int(y_pred[i]), float(confidence[i])])

    @staticmethod
    def save_classification_report_billiards(model_name, y, y_hat, n_classes=10):

        _, result_path = Utilities.make_output_dirs()

        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().detach().numpy()

        y_pred = np.argmax(y_hat, axis=1)
        labels = list(range(n_classes))
        report = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)

        report_file = os.path.join(result_path, model_name + "_classification_report.csv")

        with open(report_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "precision", "recall", "f1_score", "support"])
            for label in labels:
                values = report[str(label)]
                writer.writerow([label, values["precision"], values["recall"], values["f1-score"], values["support"]])
            writer.writerow(["macro avg",
                             report["macro avg"]["precision"],
                             report["macro avg"]["recall"],
                             report["macro avg"]["f1-score"],
                             report["macro avg"]["support"]])
            writer.writerow(["weighted avg",
                             report["weighted avg"]["precision"],
                             report["weighted avg"]["recall"],
                             report["weighted avg"]["f1-score"],
                             report["weighted avg"]["support"]])

    @staticmethod
    def plot_prediction_distribution_billiards(y, y_hat, model_name, n_classes=10, show=False):

        image_path, _ = Utilities.make_output_dirs()

        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.cpu().detach().numpy()

        y_pred = np.argmax(y_hat, axis=1)
        labels = np.arange(n_classes)
        true_counts = np.array([(y == label).sum() for label in labels])
        pred_counts = np.array([(y_pred == label).sum() for label in labels])

        x = np.arange(n_classes)
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, true_counts, width, label="True")
        plt.bar(x + width / 2, pred_counts, width, label="Predicted")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(model_name + " Prediction Distribution")
        plt.xticks(labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, model_name + "_prediction_distribution.png"), dpi=300)

        if show:
            plt.show()

    @staticmethod
    def convert_hyperparameters(hyperparameters):

        hyperparameters_save = {}

        for key in hyperparameters.keys():
            value = hyperparameters[key]

            if isinstance(value, torch.Size):
                hyperparameters_save[key] = list(value)
            elif isinstance(value, np.ndarray):
                hyperparameters_save[key] = value.tolist()
            else:
                hyperparameters_save[key] = value

        return hyperparameters_save

    @staticmethod
    def show_figures():

        plt.show()

    @staticmethod
    def plot_confusion_matrix_fashion(y, y_hat):

        accuracy = Utilities.compute_accuracy(y, y_hat)

        y_hat = np.argmax(y_hat, 1)

        cm = confusion_matrix(y, y_hat)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        label_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
                     3: 'Dress', 4: 'Coat', 5: 'Sandal',
                     6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        plt.figure()
        plt.subplot(1, 1, 1)
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[label_map[i] for i in range(10)],
                    yticklabels=[label_map[i] for i in range(10)])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix - Accuracy: " + str(accuracy))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_latent_space(z_fit, y_fit):

        label_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
                     3: 'Dress', 4: 'Coat', 5: 'Sandal',
                     6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(1, 1, 1)

        cmap = plt.get_cmap('gist_rainbow')
        colors = cmap(np.linspace(0, 1, 10))
        colors = dict(zip(label_map.keys(), colors))

        for y in label_map.keys():
            index = np.where(y_fit == y)
            ax.scatter(z_fit[index, 0], z_fit[index, 1], color=colors[y],
                       marker='o', s=30, alpha=0.5,
                       label=label_map[y])

        ax.legend()
        plt.show()

    @staticmethod
    def images_2_as_canvas(image, image2, title: str = ""):

        canvas = make_grid(image.cpu(), padding=10, nrow=10, normalize=True)
        canvas = canvas.permute(1, 2, 0).numpy() * 255
        canvas = canvas.astype("uint8")

        canvas2 = make_grid(image2.cpu(), padding=10, nrow=10, normalize=True)
        canvas2 = canvas2.permute(1, 2, 0).numpy() * 255
        canvas2 = canvas2.astype("uint8")

        canvas = np.concatenate((canvas, canvas2), axis=0)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(title)
        plt.show()
