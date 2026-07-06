import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout_rate=0.1):

        super(GeometricMultiHeadAttention, self).__init__()

        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')

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


class BLFormerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate=0.1):

        super(BLFormerEncoderLayer, self).__init__()

        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)
        self.attention_layer = GeometricMultiHeadAttention(d_model=d_model,
                                                           num_heads=num_heads,
                                                           dropout_rate=dropout_rate)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, ffn_dim),
                                          nn.GELU(),
                                          nn.Dropout(dropout_rate),
                                          nn.Linear(ffn_dim, d_model))
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        self.dropout_layer2 = nn.Dropout(dropout_rate)

    def forward(self, x, attention_bias, attention_mask):

        attention_out = self.attention_layer(self.norm_layer1(x),
                                             attention_bias,
                                             attention_mask)
        x = x + self.dropout_layer1(attention_out)

        feed_forward_out = self.feed_forward(self.norm_layer2(x))
        x = x + self.dropout_layer2(feed_forward_out)

        return x


class BLFormer(nn.Module):

    def __init__(self, hyperparameters):

        super(BLFormer, self).__init__()

        self.d_model = hyperparameters.get('d_model', 64)
        self.num_heads = hyperparameters.get('num_heads', 4)
        self.num_layers = hyperparameters.get('num_layers', 2)
        self.ffn_dim = hyperparameters.get('ffn_dim', 128)
        self.dropout_rate = hyperparameters.get('dropout_rate', 0.1)
        self.pair_feature_dim = hyperparameters.get('pair_feature_dim', 10)
        self.bias_hidden_dim = hyperparameters.get('bias_hidden_dim', 32)
        self.num_token_types = hyperparameters.get('num_token_types', 4)
        self.num_ball_ids = hyperparameters.get('num_ball_ids', 11)
        self.num_potted_thresholds = hyperparameters.get('num_potted_thresholds', 9)
        self.num_potted_classes = hyperparameters.get('num_potted_classes', 10)
        self.pooling = hyperparameters.get('pooling', 'cls')
        self.use_paper_features = hyperparameters.get('use_paper_features', False)
        self.num_paper_features = hyperparameters.get('num_paper_features', 27)
        self.paper_feature_vocab_size = hyperparameters.get('paper_feature_vocab_size', 200)

        self.type_embedding = nn.Embedding(self.num_token_types, self.d_model)
        self.ball_id_embedding = nn.Embedding(self.num_ball_ids, self.d_model,
                                              padding_idx=0)
        self.coord_encoder = nn.Sequential(nn.Linear(2, self.d_model),
                                           nn.GELU(),
                                           nn.Linear(self.d_model, self.d_model))
        self.paper_feature_embeddings = None
        self.paper_feature_norm = None
        if self.use_paper_features:
            self.paper_feature_embeddings = nn.ModuleList()
            for _ in range(self.num_paper_features):
                self.paper_feature_embeddings.append(
                    nn.Embedding(self.paper_feature_vocab_size, self.d_model))
            self.paper_feature_norm = nn.LayerNorm(self.d_model)

        self.input_dropout = nn.Dropout(self.dropout_rate)

        self.bias_network = nn.Sequential(nn.Linear(self.pair_feature_dim,
                                                    self.bias_hidden_dim),
                                          nn.GELU(),
                                          nn.Linear(self.bias_hidden_dim,
                                                    self.num_heads))

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(BLFormerEncoderLayer(d_model=self.d_model,
                                                    num_heads=self.num_heads,
                                                    ffn_dim=self.ffn_dim,
                                                    dropout_rate=self.dropout_rate))

        head_input_dim = self.d_model
        if self.pooling == 'cls_mean':
            head_input_dim = self.d_model * 2
        elif self.pooling != 'cls':
            raise ValueError('Unknown pooling: ' + str(self.pooling))

        self.output_norm = nn.LayerNorm(head_input_dim)
        self.clear_head = nn.Linear(head_input_dim, 1)
        self.win_head = nn.Linear(head_input_dim, 1)
        self.potted_score_head = nn.Linear(head_input_dim, 1)
        self.potted_class_head = nn.Linear(head_input_dim, self.num_potted_classes)
        self.potted_first_bias = nn.Parameter(torch.tensor(2.0))
        self.potted_bias_steps = nn.Parameter(torch.zeros(self.num_potted_thresholds - 1))

    def forward(self, batch):

        token_type_ids = batch['token_type_ids']
        coords = batch['coords']
        ball_ids = batch['ball_ids']
        attention_mask = batch['attention_mask']
        pair_features = batch['pair_features']

        type_embedding = self.type_embedding(token_type_ids)
        ball_id_embedding = self.ball_id_embedding(ball_ids)
        coord_embedding = self.coord_encoder(coords)

        coordinate_mask = ((token_type_ids == 1) | (token_type_ids == 2)).unsqueeze(-1)
        coord_embedding = coord_embedding * coordinate_mask.to(coord_embedding.dtype)

        x = type_embedding + ball_id_embedding + coord_embedding

        if self.use_paper_features:
            paper_features = batch['paper_features'].clamp(min=0,
                                                           max=self.paper_feature_vocab_size - 1)
            paper_embedding = 0.0
            for feature_index, embedding_layer in enumerate(self.paper_feature_embeddings):
                paper_embedding = paper_embedding + embedding_layer(
                    paper_features[:, :, feature_index])
            paper_embedding = paper_embedding / math.sqrt(self.num_paper_features)
            paper_embedding = self.paper_feature_norm(paper_embedding)
            paper_embedding = paper_embedding * (token_type_ids == 1).unsqueeze(-1).to(
                paper_embedding.dtype)
            x = x + paper_embedding

        x = self.input_dropout(x)

        attention_bias = self.bias_network(pair_features)
        attention_bias = attention_bias.permute(0, 3, 1, 2).contiguous()

        for layer in self.layers:
            x = layer(x, attention_bias, attention_mask)

        cls_output = x[:, 0]
        if self.pooling == 'cls_mean':
            valid_mask = attention_mask.to(x.dtype).unsqueeze(-1)
            masked_sum = (x * valid_mask).sum(dim=1)
            valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
            mean_output = masked_sum / valid_count
            cls_output = torch.cat([cls_output, mean_output], dim=1)

        cls_output = self.output_norm(cls_output)
        potted_score = self.potted_score_head(cls_output)
        potted_logits = potted_score + self.ordered_potted_biases()[None, :]

        return {'clear_logit': self.clear_head(cls_output).squeeze(-1),
                'win_logit': self.win_head(cls_output).squeeze(-1),
                'potted_logits': potted_logits,
                'potted_class_logits': self.potted_class_head(cls_output),
                'embedding': cls_output}

    def ordered_potted_biases(self):

        if self.num_potted_thresholds == 1:
            return self.potted_first_bias.reshape(1)

        positive_steps = F.softplus(self.potted_bias_steps)
        remaining_biases = self.potted_first_bias - torch.cumsum(positive_steps,
                                                                 dim=0)

        return torch.cat([self.potted_first_bias.reshape(1), remaining_biases], dim=0)
