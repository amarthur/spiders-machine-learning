from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder

from sp_transform import SpDataTransforms


class SpDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        sampler: str = 'weighted',
        fold: int = 0,
        num_folds: int = 5,
        val_split: float = .1,
        test_split: float = .1,
        val_seed: int = 192873,
        test_seed: int = 4710349,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = 16

        # Cross Validation
        self.fold = fold
        self.num_folds = num_folds

        # Split
        self.val_split = val_split
        self.test_split = test_split
        self.val_seed = val_seed # Should not change during cross-validation
        self.test_seed = test_seed # Should not change during the whole analysis

        self.dataset = ImageFolder(self.data_dir)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.transforms = SpDataTransforms()
        self.sampler = sampler

    def setup(self, stage: str) -> None:
        dataset_indices = list(range(len(self.dataset)))
        train_val_indices, test_indices = train_test_split(
            dataset_indices, test_size=self.test_split, random_state=self.test_seed)

        if self.num_folds > 1:
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.val_seed)
            splits = list(kf.split(train_val_indices))
            train_indices, val_indices = splits[self.current_fold]
        else:
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=self.val_split/(1-self.test_split), random_state=self.val_seed)

        self.train_dataset = SpDataset(self.dataset, indices=train_indices, transform=self.transforms.train_transforms)
        self.val_dataset = SpDataset(self.dataset, indices=val_indices, transform=self.transforms.val_transforms)
        self.test_dataset = SpDataset(self.dataset, indices=test_indices, transform=self.transforms.test_transforms)

        self.sampler = self.get_weighted_rand_sampler()

    def get_weighted_rand_sampler(self) -> WeightedRandomSampler:
        targets = self.train_dataset.targets
        class_count = np.bincount(targets)
        class_weights = 1.0 / class_count
        sample_weights = class_weights[targets]

        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return weighted_sampler

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

class SpDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__()
        self.dataset = Subset(dataset, indices)
        self.transform = transform
        self.targets = [dataset.targets[i] for i in indices]

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
