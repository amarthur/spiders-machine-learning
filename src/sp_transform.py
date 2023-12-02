from torchvision import transforms


class SpDataTransforms:
    def __init__(self) -> None:
        self.resize = 256
        self.center_crop = 224

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self._train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=self.resize, scale=(0.8, 1.0)
                ),  # zoom
                transforms.RandomRotation(degrees=15),  # rotation
                transforms.RandomHorizontalFlip(p=0.5),  # flip
                transforms.RandomVerticalFlip(p=0.5),  # flip
                transforms.ColorJitter(brightness=0.2),  # brightness
                transforms.CenterCrop(size=self.center_crop),  # crop
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

        self._val_transforms = transforms.Compose(
            [
                transforms.Resize(size=self.resize),
                transforms.CenterCrop(size=self.center_crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

        self._test_transforms = transforms.Compose(
            [
                transforms.Resize(size=self.resize),
                transforms.CenterCrop(size=self.center_crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

    @property
    def train_transforms(self) -> transforms.Compose:
        return self._train_transforms

    @property
    def val_transforms(self) -> transforms.Compose:
        return self._val_transforms

    @property
    def test_transforms(self) -> transforms.Compose:
        return self._test_transforms

    @property
    def tensor_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(size=self.resize),
                transforms.CenterCrop(size=self.center_crop),
                transforms.ToTensor(),
            ]
        )

    @property
    def normalize_transform(self) -> transforms.Normalize:
        return transforms.Normalize(mean=self.mean, std=self.std)
