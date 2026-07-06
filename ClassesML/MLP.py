import torch.nn as nn

from ClassesML.Blocks import DenseBlock
from Utilities.Utilities import Utilities


class MLP(nn.Module):

    def __init__(self, hyperparameters):

        nn.Module.__init__(self)

        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]
        self.hidden_layers_sizes = hyperparameters["hidden_layers_sizes"]
        self.activation = hyperparameters["activation"]
        self.batch_normalization = hyperparameters["batch_normalization"]
        self.dropout_rate = hyperparameters["dropout_rate"]

        self.n_dense_layers = len(self.hidden_layers_sizes)
        self.layers = nn.ModuleList()

        layer = DenseBlock(
            in_size=self.input_dim[0],
            out_size=self.hidden_layers_sizes[0],
            activation=Utilities.get_activation(self.activation),
            batch_normalization=self.batch_normalization,
            dropout_rate=self.dropout_rate
        )
        self.layers.append(layer)

        for i in range(0, self.n_dense_layers - 1):
            layer = DenseBlock(
                in_size=self.hidden_layers_sizes[i],
                out_size=self.hidden_layers_sizes[i + 1],
                activation=Utilities.get_activation(self.activation),
                batch_normalization=self.batch_normalization,
                dropout_rate=self.dropout_rate
            )
            self.layers.append(layer)

        layer = nn.Linear(
            in_features=self.hidden_layers_sizes[-1],
            out_features=self.output_dim
        )
        self.layers.append(layer)

        self.classifier = nn.Sequential(*self.layers)

    def forward(self, x):

        y_hat = self.classifier(x)

        return y_hat
