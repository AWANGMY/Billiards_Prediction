import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DenseBlock(nn.Module):

    def __init__(self, in_size, out_size, activation=nn.ReLU(),
                 batch_normalization=False, dropout_rate=0.1):

        super(DenseBlock, self).__init__()

        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation

        if batch_normalization:
            self.batch_norm_layer = nn.BatchNorm1d(out_size)
        else:
            self.batch_norm_layer = None

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.linear(x)
        if self.batch_norm_layer is not None:
            x = self.batch_norm_layer(x)
        if self.activation:
            x = self.activation(x)
        x = self.dropout_layer(x)

        return x


class BallProjectionBlock(nn.Module):

    def __init__(self, in_size, out_size, activation=nn.ReLU(), dropout_rate=0.1):

        super(BallProjectionBlock, self).__init__()

        self.linear = nn.Linear(in_size, out_size)
        self.activation = activation
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        x = self.dropout_layer(x)

        return x


class BallConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes, activation=nn.ReLU()):

        super(BallConvolutionBlock, self).__init__()

        self.activation = activation
        self.convs = nn.ModuleList()

        for kernel_size in kernel_sizes:
            layer = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)
            self.convs.append(layer)

    def forward(self, x):

        x = x.transpose(1, 2)

        output = []
        for conv in self.convs:
            conv_x = conv(x)
            if self.activation:
                conv_x = self.activation(conv_x)
            conv_x = F.max_pool1d(conv_x, conv_x.size(2)).squeeze(2)
            output.append(conv_x)

        x = torch.cat(output, dim=1)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, num_embeddings, d_model, dropout_rate=0.1):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(num_embeddings, d_model)
        position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe
        x = self.dropout(x)

        return x


class TransformerEncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, expansion_factor=2,
                 activation=nn.ReLU(), dropout_rate=0.1):

        super(TransformerEncoderBlock, self).__init__()

        hidden_dim = input_dim * expansion_factor

        self.mha_layer = nn.MultiheadAttention(embed_dim=input_dim,
                                               num_heads=num_heads,
                                               batch_first=True)
        self.attention_weights = None
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 activation,
                                 nn.Linear(hidden_dim, input_dim))
        self.norm_layer1 = nn.LayerNorm(input_dim)
        self.norm_layer2 = nn.LayerNorm(input_dim)
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        self.dropout_layer2 = nn.Dropout(dropout_rate)

    def forward(self, x):

        attention_out, self.attention_weights = self.mha_layer(x, x, x)
        x = x + attention_out
        x = self.norm_layer1(x)
        x = self.dropout_layer1(x)

        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm_layer2(x)
        x = self.dropout_layer2(x)

        return x


class SpatialAttentionBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation=nn.ReLU()):

        super(SpatialAttentionBlock, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )
        self.attention_weights = None

    def forward(self, x):

        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        self.attention_weights = weights
        x = x * weights
        x = x.sum(dim=1)

        return x
