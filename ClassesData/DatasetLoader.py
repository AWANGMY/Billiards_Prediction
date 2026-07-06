import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from ClassesData.BLFormerDataset import BLFormerDataset, blformer_collate


class DatasetLoader:

    def __init__(self, root):

        self.root = root

    def load_processed_data(self, processed_path=None):

        if processed_path is None:
            processed_path = os.path.join(self.root, "processed", "billiards_layout.pt")

        if not os.path.exists(processed_path):
            raise FileNotFoundError("Processed data not found: " + processed_path)

        try:
            return torch.load(processed_path, weights_only=False)
        except TypeError:
            return torch.load(processed_path)

    def load_paper40_split(self, data):

        if "split_indices" not in data:
            raise ValueError("Processed data does not contain split_indices.")

        split_indices = data["split_indices"]
        train_indices = self.to_index_list(split_indices["train"])
        val_indices = self.to_index_list(split_indices.get("val", []))
        test_indices = self.to_index_list(split_indices["test"])

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            "notes": ["uses split_indices stored in processed data"],
        }

    def classifier_input(self, data, model_name):

        if model_name == "BLCNN":
            return data["x_paper"].long()
        elif model_name == "MLP":
            return data["x_paper"].float().reshape(data["x_paper"].shape[0], -1)
        elif model_name in ["Transformer", "Attention"]:
            return data["x_paper"].float()
        else:
            raise ValueError("Unknown model name: " + str(model_name))

    def load_classifier_loader(
        self,
        data,
        task,
        model_name,
        indices,
        batch_size,
        shuffle,
        seed,
        num_workers,
    ):

        x = self.classifier_input(data, model_name)
        y = data[task].long()
        loader = self.make_tensor_loader(
            x=x,
            y=y,
            indices=indices,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
        )

        input_dim = tuple(x.shape[1:])
        n_classes = int(data.get("n_classes", {}).get(task, int(torch.max(y).item()) + 1))

        return loader, input_dim, n_classes

    def load_classifier_data(
        self,
        data,
        task,
        model_name,
        splits,
        batch_size,
        seed,
        num_workers,
    ):

        train_loader, input_dim, n_classes = self.load_classifier_loader(
            data=data,
            task=task,
            model_name=model_name,
            indices=splits["train"],
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
        )

        valid_loader = None
        if len(splits["val"]) > 0:
            valid_loader, _, _ = self.load_classifier_loader(
                data=data,
                task=task,
                model_name=model_name,
                indices=splits["val"],
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                num_workers=num_workers,
            )

        test_loader, _, _ = self.load_classifier_loader(
            data=data,
            task=task,
            model_name=model_name,
            indices=splits["test"],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
        )

        return train_loader, valid_loader, test_loader, input_dim, n_classes

    def load_blformer_loader(
        self,
        data,
        indices,
        batch_size,
        shuffle,
        num_workers,
        augment=False,
    ):

        dataset = BLFormerDataset(data, indices, augment=augment)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=blformer_collate,
        )

    def load_blformer_data(
        self,
        data,
        splits,
        batch_size,
        num_workers,
        augment_train=False,
    ):

        train_loader = self.load_blformer_loader(
            data=data,
            indices=splits["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            augment=augment_train,
        )

        valid_loader = None
        if len(splits["val"]) > 0:
            valid_loader = self.load_blformer_loader(
                data=data,
                indices=splits["val"],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                augment=False,
            )

        test_loader = self.load_blformer_loader(
            data=data,
            indices=splits["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            augment=False,
        )

        return train_loader, valid_loader, test_loader

    def make_tensor_loader(self, x, y, indices, batch_size, shuffle, seed, num_workers):

        index_tensor = torch.tensor(self.to_index_list(indices), dtype=torch.long)
        dataset = TensorDataset(x[index_tensor], y[index_tensor], index_tensor)
        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=generator,
        )

    def to_index_list(self, indices):

        if isinstance(indices, torch.Tensor):
            return indices.cpu().long().tolist()

        return list(indices)
