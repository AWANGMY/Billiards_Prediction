import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GeometricMultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model, num_heads, dropout_rate=0.1):

        super(GeometricMultiHeadAttentionBlock, self).__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_layer = nn.Linear(d_model, d_model * 3)
        self.output_layer = nn.Linear(d_model, d_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.attention_weights = None

    def forward(self, x, attention_bias, attention_mask):

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention_logits = attention_logits + attention_bias

        if attention_mask is not None:
            key_mask = ~attention_mask[:, None, None, :]
            attention_logits = attention_logits.masked_fill(key_mask, -1.0e9)

        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.dropout_layer(attention)
        self.attention_weights = attention.detach()

        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, self.d_model)

        return self.output_layer(context)


class BLFormerEncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate=0.1):

        super(BLFormerEncoderBlock, self).__init__()

        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)
        self.attention_layer = GeometricMultiHeadAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        self.dropout_layer2 = nn.Dropout(dropout_rate)

    def forward(self, x, attention_bias, attention_mask):

        attention_out = self.attention_layer(
            self.norm_layer1(x),
            attention_bias,
            attention_mask,
        )
        x = x + self.dropout_layer1(attention_out)

        feed_forward_out = self.feed_forward(self.norm_layer2(x))
        x = x + self.dropout_layer2(feed_forward_out)

        return x
