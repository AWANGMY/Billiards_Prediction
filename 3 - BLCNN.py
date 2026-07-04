import os
import numpy as np

import torch
from torchinfo import summary

from ClassesData.DatasetLoader import DatasetLoader
from Utilities.Utilities import Utilities
from ClassesML.CNN import BLCNN
from ClassesML.Scope import ScopeClassifier


model_type = "BLCNN"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_parent_project = os.getcwd()
dataset_path = os.path.join(path_parent_project, "Dataset", "")
batch_size = 64
prediction_tasks = ["clear", "win", "potted_after_break"]

dataset = DatasetLoader(root=dataset_path)

for target in prediction_tasks:

    model_name = model_type + "_" + target

    train_loader, val_loader, test_loader, input_dim, n_classes = dataset.load_billiards_data(
        mode="blcnn",
        target=target,
        batch_size=batch_size
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
        "learning_rate": 0.00001,
        "weight_decay": 0.001,
        "max_epochs": 20,
        "batch_size": batch_size,
    }

    model = BLCNN(hyperparameters=hyperparameters).to(device)
    scope = ScopeClassifier(model, hyperparameters)

    input_size = (128, hyperparameters["input_dim"][0], hyperparameters["input_dim"][1])
    input_data = torch.zeros(size=input_size, dtype=torch.long, device=device)
    print("Task:" + target)
    print(summary(model=model, input_data=input_data, depth=5))

    train_loss_list = []
    valid_loss_list = []
    train_accurancy_list = []
    valid_accurancy_list = []

    for epoch in range(hyperparameters["max_epochs"]):

        model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batch = len(train_loader)

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = scope.compute_loss(y_hat, y)
            total_loss += loss.item()

            scope.optimizer.zero_grad()
            loss.backward()
            scope.optimizer.step()

            batch_accuracy = Utilities.compute_accuracy(y, y_hat)
            total_accuracy += batch_accuracy

        train_loss = total_loss / n_batch
        train_accurancy = total_accuracy / n_batch
        print("Task:" + target)
        print("Epoch:" + str(epoch + 1) + "/" + str(hyperparameters["max_epochs"]))
        print("Training Loss:" + str(train_loss) + " - Train Accuracy:" + str(train_accurancy))

        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batch = len(val_loader)

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                loss = scope.compute_loss(y_hat, y)
                total_loss += loss.item()

                batch_accuracy = Utilities.compute_accuracy(y, y_hat)
                total_accuracy += batch_accuracy

        valid_loss = total_loss / n_batch
        valid_accurancy = total_accuracy / n_batch
        print("Task:" + target)
        print("Epoch:" + str(epoch + 1) + "/" + str(hyperparameters["max_epochs"]))
        print("Validation Loss:" + str(valid_loss) + " - Validation Accuracy:" + str(valid_accurancy))

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_accurancy_list.append(train_accurancy)
        valid_accurancy_list.append(valid_accurancy)

    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    n_batch = len(test_loader)
    y_test_list = []
    y_hat_test_list = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = scope.compute_loss(y_hat, y)
            total_loss += loss.item()

            batch_accuracy = Utilities.compute_accuracy(y, y_hat)
            total_accuracy += batch_accuracy

            y_test_list.append(y.cpu().detach().numpy())
            y_hat_test_list.append(y_hat.cpu().detach().numpy())

    test_loss = total_loss / n_batch
    test_accuracy = total_accuracy / n_batch
    print("Task:" + target)
    print("Test Loss:" + str(test_loss) + " - Test Accuracy:" + str(test_accuracy))

    y_test = np.concatenate(y_test_list)
    y_hat_test = np.concatenate(y_hat_test_list)

    Utilities.plot_training_curves(train_loss_list=train_loss_list,
                                   valid_loss_list=valid_loss_list,
                                   train_accuracy_list=train_accurancy_list,
                                   valid_accuracy_list=valid_accurancy_list,
                                   model_name=model_name)

    Utilities.plot_confusion_matrix_billiards(y=y_test,
                                              y_hat=y_hat_test,
                                              model_name=model_name,
                                              n_classes=n_classes)

    Utilities.save_training_result(model_name=model_name,
                                   hyperparameters=hyperparameters,
                                   train_loss_list=train_loss_list,
                                   valid_loss_list=valid_loss_list,
                                   train_accuracy_list=train_accurancy_list,
                                   valid_accuracy_list=valid_accurancy_list,
                                   test_loss=test_loss,
                                   test_accuracy=test_accuracy)

    Utilities.save_predictions(model_name=model_name,
                               y=y_test,
                               y_hat=y_hat_test)

    Utilities.save_classification_report_billiards(model_name=model_name,
                                                   y=y_test,
                                                   y_hat=y_hat_test,
                                                   n_classes=n_classes)

    Utilities.plot_prediction_distribution_billiards(y=y_test,
                                                     y_hat=y_hat_test,
                                                     model_name=model_name,
                                                     n_classes=n_classes)

Utilities.show_figures()
