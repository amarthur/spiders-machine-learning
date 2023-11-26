from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder

from sp_dataset import SpDataset
from sp_transform import SpDataTransforms


class SpDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 128,
        sampler: str = "weighted",
        fold: int = 0,
        num_folds: int = 1,
        val_split: float = 0.1,
        test_split: float = 0.1,
        val_seed: int = 3459782,
        test_seed: int = 4710349,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = 16

        # Cross Validation
        if not (0 <= fold < num_folds):
            raise ValueError(
                f"Invalid fold value: {fold}. Should be in range [0, {num_folds-1}]"
            )
        self.fold = fold
        self.num_folds = num_folds

        # Split
        self.val_split = val_split
        self.test_split = test_split
        self.val_seed = val_seed  # Should not change during cross-validation
        self.test_seed = test_seed  # Should not change during the whole analysis

        self.dataset = ImageFolder(self.data_dir)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.transforms = SpDataTransforms()
        self.sampler = sampler

    def setup(self, stage: str) -> None:
        dataset_indices = list(range(len(self.dataset)))

        train_val_indices, test_indices = train_test_split(
            dataset_indices,
            test_size=self.test_split,
            random_state=self.test_seed,
            stratify=self.dataset.targets,
        )
        train_val_labels = [self.dataset.targets[idx] for idx in train_val_indices]

        if self.num_folds > 1:
            print(f"Current fold: {self.fold}")
            skf = StratifiedKFold(
                n_splits=self.num_folds, shuffle=True, random_state=self.val_seed
            )
            splits = list(skf.split(train_val_indices, train_val_labels))
            train_indices, val_indices = splits[self.fold]

            # Map indices back to the original dataset range
            train_indices = [train_val_indices[idx] for idx in train_indices]
            val_indices = [train_val_indices[idx] for idx in val_indices]

        else:
            if self.val_split > 0:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=self.val_split / (1 - self.test_split),
                    random_state=self.val_seed,
                    stratify=train_val_labels,
                )
            else:
                train_indices, val_indices = train_val_indices, []

        self.train_dataset = SpDataset(
            self.dataset,
            indices=train_indices,
            transform=self.transforms.train_transforms,
        )
        self.val_dataset = SpDataset(
            self.dataset,
            indices=val_indices,
            transform=self.transforms.val_transforms,
        )
        self.test_dataset = SpDataset(
            self.dataset,
            indices=test_indices,
            transform=self.transforms.test_transforms,
        )

        self.sampler = self.get_weighted_rand_sampler()

    def get_weighted_rand_sampler(self) -> WeightedRandomSampler:
        targets = self.train_dataset.labels
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
