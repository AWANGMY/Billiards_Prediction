import torch
import torch.nn as nn
import torch.nn.functional as F


class BLCNN(nn.Module):

    def __init__(self, hyperparameters):

        nn.Module.__init__(self)

        self.input_dim = hyperparameters["input_dim"]
        self.output_dim = hyperparameters["output_dim"]
        self.embed_dim = hyperparameters["embed_dim"]
        self.kernel_num = hyperparameters["kernel_num"]
        self.kernel_sizes = hyperparameters["kernel_sizes"]
        self.dropout_rate = hyperparameters["dropout_rate"]

        self.embed_1 = nn.Embedding(hyperparameters["embed_num_pos"], self.embed_dim)
        self.embed_2 = nn.Embedding(hyperparameters["embed_num_deg_45"], self.embed_dim)
        self.embed_3 = nn.Embedding(hyperparameters["embed_num_deg_90"], self.embed_dim)
        self.embed_4 = nn.Embedding(hyperparameters["embed_num_deg_180"], self.embed_dim)
        self.embed_5 = nn.Embedding(hyperparameters["embed_num_distance"], self.embed_dim)
        self.embed_6 = nn.Embedding(hyperparameters["embed_pocket"], self.embed_dim)
        self.embed_7 = nn.Embedding(hyperparameters["embed_num_pocket"], self.embed_dim)

        self.embedding_size = self.embed_dim * 27
        self.convs = nn.ModuleList()

        for kernel_size in self.kernel_sizes:
            layer = nn.Conv2d(in_channels=1,
                              out_channels=self.kernel_num,
                              kernel_size=(kernel_size, self.embedding_size))
            self.convs.append(layer)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.kernel_num,
                            self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        position_embed = self.embed_1(x[:, :, 0])

        deg_45_embed_1 = self.embed_2(x[:, :, 1:9:4])
        deg_45_embed_1 = torch.reshape(deg_45_embed_1, (x.shape[0], x.shape[1], -1))

        deg_90_embed_1 = self.embed_3(x[:, :, 9:13:4])
        deg_90_embed_1 = torch.reshape(deg_90_embed_1, (x.shape[0], x.shape[1], -1))

        deg_45_embed_2 = self.embed_2(x[:, :, 13:17:4])
        deg_45_embed_2 = torch.reshape(deg_45_embed_2, (x.shape[0], x.shape[1], -1))

        deg_90_embed_2 = self.embed_3(x[:, :, 17:21:4])
        deg_90_embed_2 = torch.reshape(deg_90_embed_2, (x.shape[0], x.shape[1], -1))

        deg_45_embed_3 = self.embed_2(x[:, :, 21:25:4])
        deg_45_embed_3 = torch.reshape(deg_45_embed_3, (x.shape[0], x.shape[1], -1))

        distance_embed = self.embed_5(x[:, :, 2:25:4])
        distance_embed = torch.reshape(distance_embed, (x.shape[0], x.shape[1], -1))

        is_pocket_embed = self.embed_6(x[:, :, 3:25:4])
        is_pocket_embed = torch.reshape(is_pocket_embed, (x.shape[0], x.shape[1], -1))

        pocket_num_embed_1 = self.embed_7(x[:, :, 4:25:4])
        pocket_num_embed_1 = torch.reshape(pocket_num_embed_1, (x.shape[0], x.shape[1], -1))

        deg_180_embed = self.embed_4(x[:, :, 25])
        deg_180_embed = torch.reshape(deg_180_embed, (x.shape[0], x.shape[1], -1))

        pocket_num_embed_2 = self.embed_4(x[:, :, 26])
        pocket_num_embed_2 = torch.reshape(pocket_num_embed_2, (x.shape[0], x.shape[1], -1))

        x = torch.cat((position_embed,
                       deg_45_embed_1,
                       deg_90_embed_1,
                       deg_45_embed_2,
                       deg_90_embed_2,
                       deg_45_embed_3,
                       distance_embed,
                       is_pocket_embed,
                       pocket_num_embed_1,
                       deg_180_embed,
                       pocket_num_embed_2), 2)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        y_hat = self.fc(x)
        y_hat = self.softmax(y_hat)

        return y_hat
