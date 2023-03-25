import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from directory_structure import DirectoryStructure
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


class SpeciesModel:

    def __init__(self, dataset_name: str, device: torch.device) -> None:
        # Parameters
        self.dataset_name = dataset_name
        self.device = device

        # Directories
        self.dirs = DirectoryStructure(dataset_dir_name=dataset_name)
        self.phases = self.dirs.phases
        self.train_phase = self.dirs.train_phase
        self.valid_phase = self.dirs.valid_phase
        self.test_phase = self.dirs.test_phase

        # Data
        self.image_datasets = None
        self.dataset_sizes = None
        self.dataloaders = None
        self.class_names = None
        self.num_classes = None

        # Transforms
        resize_size = 256
        center_crop_size = 224
        rotation_degrees = 15
        transform_mean = [0.485, 0.456, 0.406]
        transform_std = [0.229, 0.224, 0.225]

        self.data_transforms = {
            self.train_phase:
                transforms.Compose([
                    transforms.RandomResizedCrop(size=resize_size, scale=(0.8, 1.0)),
                    transforms.RandomRotation(degrees=rotation_degrees),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=center_crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=transform_mean, std=transform_std)
                ]),
            self.valid_phase:
                transforms.Compose([
                    transforms.Resize(size=resize_size),
                    transforms.CenterCrop(size=center_crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=transform_mean, std=transform_std)
                ]),
            self.test_phase:
                transforms.Compose([
                    transforms.Resize(size=resize_size),
                    transforms.CenterCrop(size=center_crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=transform_mean, std=transform_std)
                ])
        }

    def load_data(self, batch_size, weighted_sampler=False):
        # Define image datasets
        self.image_datasets = {
            phase: datasets.ImageFolder(root=path, transform=self.data_transforms[phase])
            for phase, path in self.dirs.phases_dirs.items()
        }

        # Define dataloaders
        weighted_sampler = self.get_weighted_sampler() if weighted_sampler else None
        sampler = {phase: (weighted_sampler if phase == self.train_phase else None) for phase in self.phases}
        shuffle = {phase: (True if sampler[phase] is None else False) for phase in self.phases}
        self.dataloaders = {
            phase: DataLoader(dataset=self.image_datasets[phase],
                              batch_size=batch_size,
                              shuffle=shuffle[phase],
                              sampler=sampler[phase],
                              num_workers=4) for phase in self.phases
        }

        # Define related data
        self.dataset_sizes = {phase: len(self.image_datasets[phase]) for phase in self.phases}
        self.class_names = self.image_datasets[self.train_phase].classes
        self.num_classes = len(self.class_names)

    def get_weighted_sampler(self):
        training_set = pd.read_csv(self.dirs.phases_csv[self.train_phase])
        training_set_classes = training_set["class"]

        num_images_classes = training_set_classes.value_counts()
        sample_weights = [1 / num_images_classes[i] for i in training_set_classes.values]

        weighted_sampler = WeightedRandomSampler(weights=sample_weights,
                                                 num_samples=len(sample_weights),
                                                 replacement=True)
        return weighted_sampler

    def load_pretrained_model(self, model_name, weights_path=None):
        model_func = getattr(models, model_name)
        pretrained_model = model_func(weights='DEFAULT')

        # Freeze pre-trained parameters
        for parameter in pretrained_model.parameters():
            parameter.requires_grad = False

        # Replace the final layer (requires_grad=True by default)
        in_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.4),
                                            nn.Linear(256, self.num_classes))

        model = pretrained_model.to(self.device)

        if weights_path:
            if Path(weights_path).is_file():
                model.load_state_dict(torch.load(weights_path))
            else:
                print("Weights file not found, initializing with random weights.")

        return model

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            for phase in self.phases:
                if phase == self.train_phase:
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero out the gradients. Otherwise, the gradient would be a combination of the old gradient
                    # (which we have already used to update the model) and the newly-computed gradient.
                    optimizer.zero_grad()

                    # Enable gradient while training
                    with torch.set_grad_enabled(phase == self.train_phase):
                        outputs = model(inputs)  # Forward pass
                        _, preds = torch.max(outputs, 1)  # Get prediction
                        loss = criterion(outputs, labels)  # Compute loss

                        if phase == self.train_phase:
                            loss.backward()  # Backpropagate
                            optimizer.step()  # Optimize

                    # Error measures
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == self.train_phase:
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.2f}%")

                # Deep copy the model
                if phase == self.valid_phase and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # torch.save(best_model_wts, self.dirs.weights_dir / "path")

            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def show_predictions(self, model, num_images=9):
        # Set model to eval mode
        model.eval()

        # Get class names from dataloader
        dataloader = self.dataloaders[self.valid_phase]
        class_names = dataloader.dataset.classes

        # Get a batch of images and labels from the dataloader
        images, labels = next(iter(dataloader))
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Make predictions on the batch
        with torch.no_grad():
            preds = model(images)
            preds = torch.argmax(preds, dim=1)

        images = [self.normalize_image(image.cpu()) for image in images]
        self.plot_predictions(images, preds, labels, class_names, num_images)

    def plot_predictions(self, images, preds, labels, class_names, num_images, cols=4):
        rows = -(-num_images // cols)  # ceil(num_images / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 7))

        for i in range(num_images):
            row, col = divmod(i, cols)
            if i >= len(images):  # If there are fewer images than axes, remove the axis
                axes[row, col].axis('off')
                continue

            axes[row, col].imshow(images[i])
            axes[row, col].axis('off')
            pred_class = class_names[preds[i]]
            true_class = class_names[labels[i]]
            if preds[i] == labels[i]:
                axes[row, col].set_title(f'Predicted: {pred_class}', fontsize=12, color='green')
            else:
                axes[row, col].text(-20, -30, f"Predicted: {pred_class}", fontsize=12, color="red")
                axes[row, col].text(-20, -5, f"True class: {true_class}", fontsize=12, color="black")

        # Remove unused axes
        for i in range(num_images, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def normalize_image(tensor):
        image = tensor.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std*image + mean
        image = np.clip(image, 0, 1)
        return image


def main():
    if not torch.cuda.is_available():
        print("Cuda not available")
        return

    num_epochs = 1
    batch_size = 128
    device = torch.device("cuda:0")

    spc = SpeciesModel(dataset_name="Dataset", device=device)
    spc.load_data(batch_size=batch_size, weighted_sampler=True)

    model = spc.load_pretrained_model(model_name='resnet50')
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

    trained_model = spc.train_model(model=model,
                                    criterion=criterion,
                                    optimizer=optimizer_ft,
                                    scheduler=exp_lr_scheduler,
                                    num_epochs=num_epochs)

    spc.show_predictions(trained_model, num_images=12)


if __name__ == "__main__":
    main()
