from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder

from sp_transform import SpDataTransforms


class SpDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int = 64, num_workers: int = 4
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"

        self.train_dataset = None
        self.val_dataset = None

        self.transforms = SpDataTransforms()
        self.train_transforms = self.transforms.train_transforms
        self.val_transforms = self.transforms.val_transforms
        self.weighted_sampler = None

    def setup(self, stage: str) -> None:
        self.train_dataset = ImageFolder(
            root=self.train_dir,
            transform=self.train_transforms,
        )
        self.val_dataset = ImageFolder(
            root=self.val_dir,
            transform=self.val_transforms,
        )
        self.weighted_sampler = self.get_weighted_rand_sampler()

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
            sampler=self.weighted_sampler,
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
