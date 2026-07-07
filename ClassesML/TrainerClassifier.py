import torch

from Utilities.Utilities import Utilities


class TrainerClassifier:

    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.history = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_accuracy_list = []
        self.valid_accuracy_list = []

    def set_model(self, model, device):

        self.model = model
        self.device = device

    def set_scope(self, scope):

        self.scope = scope

    def set_data(self, train_loader, valid_loader=None):

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def _unpack_batch(self, batch):

        if len(batch) == 2:
            x, y = batch
            sample_indices = None
        else:
            x, y, sample_indices = batch

        return x, y, sample_indices

    def _run_single_loader(self, loader, training=False, keep_rows=False):

        if loader is None:
            return None, []

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_correct = 0
        total_count = 0
        rows = []

        for batch in loader:
            x, y, sample_indices = self._unpack_batch(batch)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.set_grad_enabled(training):
                y_hat = self.model(x)
                loss = self.scope.compute_loss(y_hat, y)

                if training:
                    self.scope.optimizer.zero_grad()
                    loss.backward()
                    self.scope.optimizer.step()

            predicted = torch.argmax(y_hat, dim=1)
            batch_size = int(y.size(0))
            batch_accuracy = Utilities.compute_accuracy(y, y_hat)
            batch_correct = int((predicted == y).sum().item())

            total_loss += float(loss.item()) * batch_size
            total_accuracy += batch_accuracy
            total_correct += batch_correct
            total_count += batch_size

            if keep_rows and sample_indices is not None:
                for row_index in range(batch_size):
                    rows.append({
                        "sample_index": int(sample_indices[row_index].item()),
                        "y_true": int(y[row_index].item()),
                        "y_pred": int(predicted[row_index].item()),
                    })

        metrics = {
            "loss": total_loss / total_count if total_count > 0 else None,
            "accuracy": total_correct / total_count * 100.0 if total_count > 0 else None,
            "batch_mean_accuracy": total_accuracy / len(loader) if len(loader) > 0 else None,
            "correct": total_correct,
            "count": total_count,
        }

        return metrics, rows

    def run(self):

        self.history = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_accuracy_list = []
        self.valid_accuracy_list = []

        max_epochs = self.hyperparameters["max_epochs"]

        for epoch in range(max_epochs):
            train_metrics, _ = self._run_single_loader(self.train_loader, training=True)
            self.train_loss_list.append(train_metrics["loss"])
            self.train_accuracy_list.append(train_metrics["accuracy"])

            print("Epoch:" + str(epoch + 1) + "/" + str(max_epochs))
            print(
                "Training Loss:" + str(train_metrics["loss"])
                + " - Train Accuracy:" + str(train_metrics["accuracy"])
            )

            row = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_batch_mean_accuracy": train_metrics["batch_mean_accuracy"],
            }

            if self.valid_loader is not None:
                valid_metrics, _ = self._run_single_loader(self.valid_loader, training=False)
                self.valid_loss_list.append(valid_metrics["loss"])
                self.valid_accuracy_list.append(valid_metrics["accuracy"])

                print("Epoch:" + str(epoch + 1) + "/" + str(max_epochs))
                print(
                    "Validation Loss:" + str(valid_metrics["loss"])
                    + " - Validation Accuracy:" + str(valid_metrics["accuracy"])
                )

                row["valid_loss"] = valid_metrics["loss"]
                row["valid_accuracy"] = valid_metrics["accuracy"]
                row["valid_batch_mean_accuracy"] = valid_metrics["batch_mean_accuracy"]

            self.history.append(row)

        return self.train_accuracy_list, self.valid_accuracy_list

    def evaluate(self, loader):

        return self._run_single_loader(loader, training=False, keep_rows=True)


class TrainerBLFormer:

    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters
        self.history = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_accuracy_list = []
        self.valid_accuracy_list = []
        self.tasks = ["clear", "win", "potted_after_break"]

    def set_model(self, model, device):

        self.model = model
        self.device = device

    def set_scope(self, scope):

        self.scope = scope

    def set_data(self, train_loader, valid_loader=None):

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def move_batch(self, batch):

        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value

        return moved

    def safe_divide(self, numerator, denominator):

        if denominator == 0:
            return 0.0

        return float(numerator) / float(denominator)

    def classification_metrics(self, y_true, y_pred, labels):

        accuracy = float((y_true == y_pred).float().mean().item()) * 100.0

        precision_values = []
        recall_values = []
        f1_values = []

        for label in labels:
            label_tensor = torch.tensor(label, device=y_true.device)
            true_positive = int(((y_true == label_tensor) & (y_pred == label_tensor)).sum().item())
            false_positive = int(((y_true != label_tensor) & (y_pred == label_tensor)).sum().item())
            false_negative = int(((y_true == label_tensor) & (y_pred != label_tensor)).sum().item())

            precision = self.safe_divide(true_positive, true_positive + false_positive)
            recall = self.safe_divide(true_positive, true_positive + false_negative)
            f1 = self.safe_divide(2.0 * precision * recall, precision + recall)

            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)

        return {
            "accuracy": accuracy,
            "macro_precision": sum(precision_values) / len(precision_values),
            "macro_recall": sum(recall_values) / len(recall_values),
            "macro_f1": sum(f1_values) / len(f1_values),
        }

    def run_loader(self, loader, training=False, keep_rows=False):

        if loader is None:
            return None, []

        if training:
            self.model.train()
        else:
            self.model.eval()

        loss_sums = {
            "loss": 0.0,
            "clear_loss": 0.0,
            "win_loss": 0.0,
            "potted_loss": 0.0,
            "potted_ordinal_loss": 0.0,
            "potted_class_loss": 0.0,
            "joint_loss": 0.0,
        }
        count = 0
        y_true = {task: [] for task in self.tasks}
        y_pred = {task: [] for task in self.tasks}
        rows = []

        for batch in loader:
            batch = self.move_batch(batch)

            with torch.set_grad_enabled(training):
                outputs = self.model(batch)
                losses = self.scope.compute_loss(outputs, batch)

                if training:
                    self.scope.optimizer.zero_grad()
                    losses["loss"].backward()
                    self.scope.optimizer.step()

            predictions = self.scope.predict(outputs)
            batch_size = int(batch["clear"].shape[0])
            count += batch_size

            for key in loss_sums:
                loss_sums[key] += float(losses[key].item()) * batch_size

            for task in self.tasks:
                y_true[task].append(batch[task].detach().cpu())
                y_pred[task].append(predictions[task].detach().cpu())

            if keep_rows:
                sample_indices = batch["sample_indices"].detach().cpu()
                for row_index in range(batch_size):
                    rows.append(
                        {
                            "sample_index": int(sample_indices[row_index].item()),
                            "clear_true": int(batch["clear"][row_index].item()),
                            "clear_pred": int(predictions["clear"][row_index].item()),
                            "win_true": int(batch["win"][row_index].item()),
                            "win_pred": int(predictions["win"][row_index].item()),
                            "potted_after_break_true": int(batch["potted_after_break"][row_index].item()),
                            "potted_after_break_pred": int(predictions["potted_after_break"][row_index].item()),
                        }
                    )

        metrics = {}
        for key in loss_sums:
            metrics[key] = loss_sums[key] / count if count > 0 else None

        task_metrics = {}
        for task in self.tasks:
            task_true = torch.cat(y_true[task], dim=0)
            task_pred = torch.cat(y_pred[task], dim=0)

            if task in ["clear", "win"]:
                labels = [0, 1]
            else:
                labels = list(range(10))

            task_metrics[task] = self.classification_metrics(task_true, task_pred, labels)

        metrics["task_metrics"] = task_metrics
        metrics["mean_accuracy"] = sum(
            [task_metrics[task]["accuracy"] for task in self.tasks]
        ) / len(self.tasks)
        metrics["mean_macro_f1"] = sum(
            [task_metrics[task]["macro_f1"] for task in self.tasks]
        ) / len(self.tasks)

        return metrics, rows

    def run(self):

        self.history = []
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_accuracy_list = []
        self.valid_accuracy_list = []

        for epoch in range(self.hyperparameters["max_epochs"]):
            train_metrics, _ = self.run_loader(self.train_loader, training=True)
            self.train_loss_list.append(train_metrics["loss"])
            self.train_accuracy_list.append(train_metrics["mean_accuracy"])

            print("Epoch:" + str(epoch + 1) + "/" + str(self.hyperparameters["max_epochs"]))
            print(
                "Training Loss:" + str(train_metrics["loss"])
                + " - Train Accuracy:" + str(train_metrics["mean_accuracy"])
            )

            row = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["mean_accuracy"],
                "train_macro_f1": train_metrics["mean_macro_f1"],
                "train_clear_accuracy": train_metrics["task_metrics"]["clear"]["accuracy"],
                "train_win_accuracy": train_metrics["task_metrics"]["win"]["accuracy"],
                "train_potted_accuracy": train_metrics["task_metrics"]["potted_after_break"]["accuracy"],
            }

            if self.valid_loader is not None:
                valid_metrics, _ = self.run_loader(self.valid_loader, training=False)
                self.valid_loss_list.append(valid_metrics["loss"])
                self.valid_accuracy_list.append(valid_metrics["mean_accuracy"])

                print("Epoch:" + str(epoch + 1) + "/" + str(self.hyperparameters["max_epochs"]))
                print(
                    "Validation Loss:" + str(valid_metrics["loss"])
                    + " - Validation Accuracy:" + str(valid_metrics["mean_accuracy"])
                )

                row["valid_loss"] = valid_metrics["loss"]
                row["valid_accuracy"] = valid_metrics["mean_accuracy"]
                row["valid_macro_f1"] = valid_metrics["mean_macro_f1"]
                row["valid_clear_accuracy"] = valid_metrics["task_metrics"]["clear"]["accuracy"]
                row["valid_win_accuracy"] = valid_metrics["task_metrics"]["win"]["accuracy"]
                row["valid_potted_accuracy"] = valid_metrics["task_metrics"]["potted_after_break"]["accuracy"]

            self.history.append(row)

        return self.train_accuracy_list, self.valid_accuracy_list

    def evaluate(self, loader):

        return self.run_loader(loader, training=False, keep_rows=True)
