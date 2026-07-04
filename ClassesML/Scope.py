import torch.nn as nn
import torch.optim as optim


class ScopeClassifier:

    def __init__(self, model, hyperparameters):

        self.criterion = nn.CrossEntropyLoss()
        weight_decay = hyperparameters["weight_decay"] if "weight_decay" in hyperparameters else 0.0
        self.optimizer = optim.Adam(model.parameters(),
                                    lr=hyperparameters["learning_rate"],
                                    weight_decay=weight_decay)

    def compute_loss(self, y_hat, y):

        loss = self.criterion(y_hat, y)

        return loss
