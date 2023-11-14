from torch.utils.data import Subset, Dataset

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

