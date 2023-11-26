from torch.utils.data import Dataset, Subset


class SpDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__()
        self.dataset = Subset(dataset, indices)
        self.labels = [dataset.targets[idx] for idx in indices]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
