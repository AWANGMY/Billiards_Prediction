import os

import torch
from torch.utils.data import DataLoader

from ClassesData.BilliardsDataset import BilliardsDataset


class DatasetLoader:

    def __init__(self, root):

        self.root = root

    def load_billiards_data(self,
                            processed_path=None,
                            mode='tokens',
                            target='potted_after_break',
                            batch_size=64,
                            num_workers=0):

        if processed_path is None:
            processed_path = os.path.join(self.root, 'processed', 'billiards_layout.pt')

        if not os.path.exists(processed_path):
            raise FileNotFoundError('Processed data not found: ' + processed_path +
                                    '\nRun: python ClassesData/PreprocessBilliards.py --root ' + self.root)

        data = self._load_processed_data(processed_path)

        train_dataset = BilliardsDataset(data,
                                         data['split_indices']['train'],
                                         mode=mode,
                                         target=target)
        val_dataset = BilliardsDataset(data,
                                       data['split_indices']['val'],
                                       mode=mode,
                                       target=target)
        test_dataset = BilliardsDataset(data,
                                        data['split_indices']['test'],
                                        mode=mode,
                                        target=target)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

        input_dim = train_dataset[0][0].shape
        n_classes = data['n_classes'][target]

        return train_loader, val_loader, test_loader, input_dim, n_classes

    def _load_processed_data(self, processed_path):

        try:
            return torch.load(processed_path, weights_only=False)
        except TypeError:
            return torch.load(processed_path)
