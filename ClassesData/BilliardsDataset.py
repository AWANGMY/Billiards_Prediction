import torch
from torch.utils.data import Dataset


class BilliardsDataset(Dataset):

    def __init__(self, data, indices, mode='tokens', target='potted_after_break'):

        super(BilliardsDataset, self).__init__()

        self.data = data
        self.indices = indices.long() if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long)
        self.mode = mode
        self.target = target

        if self.target not in self.data:
            raise ValueError('Unknown target: ' + str(self.target))

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        data_idx = self.indices[idx]
        y = self.data[self.target][data_idx].clone().long()

        if self.mode == 'layout_mlp':
            x = self.data['x'][data_idx].clone().float().reshape(-1)
        elif self.mode == 'layout':
            x = self.data['x'][data_idx].clone().float()
        elif self.mode == 'mlp':
            x = self.data['x_paper'][data_idx].clone().float().reshape(-1)
        elif self.mode in ['transformer', 'attention']:
            x = self.data['x_paper'][data_idx].clone().float()
        elif self.mode in ['tokens', 'paper', 'blcnn']:
            x = self.data['x_paper'][data_idx].clone().long()
        elif self.mode == 'cnn':
            x = self.data['x'][data_idx].clone().float().transpose(0, 1)
        else:
            raise ValueError('Unknown mode: ' + str(self.mode))

        return x, y
