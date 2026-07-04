import torch
import torch.nn as nn

from ClassesML.Block import BallProjectionBlock, DenseBlock, PositionalEncoding, TransformerEncoderBlock, SpatialAttentionBlock
from Utilities.Utilities import Utilities


class Transformer(nn.Module):

    def __init__(self, hyperparameters):

        nn.Module.__init__(self)

        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]
        self.embedding_dim = hyperparameters["embedding_dim"]
        self.num_heads = hyperparameters["num_heads"]
        self.num_layers = hyperparameters["num_layers"]
        self.expansion_factor = hyperparameters["expansion_factor"]
        self.activation = hyperparameters["activation"]
        self.dropout_rate = hyperparameters["dropout_rate"]

        self.embedding_layer = BallProjectionBlock(
            in_size=self.input_dim[-1],
            out_size=self.embedding_dim,
            activation=Utilities.get_activation(self.activation),
            dropout_rate=self.dropout_rate
        )

        self.position_layer = PositionalEncoding(
            num_embeddings=self.input_dim[0],
            d_model=self.embedding_dim,
            dropout_rate=self.dropout_rate
        )

        self.layers = nn.ModuleList()
        for i in range(0, self.num_layers):
            layer = TransformerEncoderBlock(
                input_dim=self.embedding_dim,
                num_heads=self.num_heads,
                expansion_factor=self.expansion_factor,
                activation=Utilities.get_activation(self.activation),
                dropout_rate=self.dropout_rate
            )
            self.layers.append(layer)

        self.classifier = nn.Linear(
            in_features=self.embedding_dim,
            out_features=self.output_dim
        )

    def forward(self, x):

        x = self.embedding_layer(x)
        x = self.position_layer(x)

        for layer in self.layers:
            x = layer(x)

        x = torch.mean(x, dim=1)
        y_hat = self.classifier(x)

        return y_hat


class SpatialAttention(nn.Module):

    def __init__(self, hyperparameters):

        nn.Module.__init__(self)

        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]
        self.embedding_dim = hyperparameters["embedding_dim"]
        self.hidden_layers_sizes = hyperparameters["hidden_layers_sizes"]
        self.activation = hyperparameters["activation"]
        self.dropout_rate = hyperparameters["dropout_rate"]

        self.embedding_layer = BallProjectionBlock(
            in_size=self.input_dim[-1],
            out_size=self.embedding_dim,
            activation=Utilities.get_activation(self.activation),
            dropout_rate=self.dropout_rate
        )

        self.attention_layer = SpatialAttentionBlock(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_layers_sizes[0],
            activation=Utilities.get_activation(self.activation)
        )

        self.classifier = nn.Sequential(
            DenseBlock(
                in_size=self.embedding_dim,
                out_size=self.hidden_layers_sizes[0],
                activation=Utilities.get_activation(self.activation),
                batch_normalization=False,
                dropout_rate=self.dropout_rate
            ),
            nn.Linear(
                in_features=self.hidden_layers_sizes[0],
                out_features=self.output_dim
            )
        )

    def forward(self, x):

        x = self.embedding_layer(x)
        x = self.attention_layer(x)
        y_hat = self.classifier(x)

        return y_hat
