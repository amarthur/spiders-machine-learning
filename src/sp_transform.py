from torchvision import transforms


class SpDataTransforms:
    def __init__(self) -> None:
        self._train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # zoom
                transforms.RandomRotation(degrees=15), # rotation
                transforms.RandomHorizontalFlip(p=0.5), # flip
                transforms.RandomVerticalFlip(p=0.5), # flip
                transforms.ColorJitter(brightness=0.2), # brightness
                transforms.CenterCrop(size=224), # crop
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._val_transforms = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        
        self._test_transforms = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @property
    def train_transforms(self):
        return self._train_transforms

    @property
    def val_transforms(self):
        return self._val_transforms

    @property
    def test_transforms(self):
        return self._test_transforms
