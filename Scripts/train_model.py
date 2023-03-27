import time
from copy import deepcopy
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
        self.transform_mean = [0.485, 0.456, 0.406]
        self.transform_std = [0.229, 0.224, 0.225]

        self.data_transforms = {
            self.train_phase:
                transforms.Compose([
                    transforms.RandomResizedCrop(size=resize_size, scale=(0.8, 1.0)),
                    transforms.RandomRotation(degrees=rotation_degrees),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=center_crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.transform_mean, std=self.transform_std)
                ]),
            self.valid_phase:
                transforms.Compose([
                    transforms.Resize(size=resize_size),
                    transforms.CenterCrop(size=center_crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.transform_mean, std=self.transform_std)
                ]),
            self.test_phase:
                transforms.Compose([
                    transforms.Resize(size=resize_size),
                    transforms.CenterCrop(size=center_crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.transform_mean, std=self.transform_std)
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

    def load_pretrained_model(self, model_name, weights_file_name=None):
        model_func = getattr(models, model_name)
        pretrained_model = model_func(weights='DEFAULT')

        # Freeze pre-trained parameters
        for parameter in pretrained_model.parameters():
            parameter.requires_grad = False

        # Replace the final layer (requires_grad=True by default)
        in_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.4),
                                            nn.Linear(256, self.num_classes))

        # Try to load weights stored in a file (if any)
        pretrained_model = nn.DataParallel(pretrained_model)
        pretrained_model = self.load_model(pretrained_model, weights_file_name)
        return pretrained_model

    def train_model(self, model, criterion, optimizer, scheduler, num_epochs, weights_file_name=None):
        since = time.time()
        best_model_state = deepcopy(model.state_dict())
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

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.3f}%")

                # Deep copy the model
                if phase == self.valid_phase and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_state = deepcopy(model.state_dict())
                    if weights_file_name is not None:
                        self.save_best_model(best_model_state, weights_file_name)

            print()

        time_elapsed = time.time() - since
        print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc * 100:.3f}")

        # load best model weights
        model.load_state_dict(best_model_state)
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

    def normalize_image(self, tensor):
        image = tensor.numpy().transpose((1, 2, 0))
        mean = np.array(self.transform_mean)
        std = np.array(self.transform_std)
        image = std*image + mean
        image = np.clip(image, 0, 1)
        return image

    def save_best_model(self, model_state_dict, weights_file_name):
        model_file_path = Path(self.dirs.weights_dir / weights_file_name)
        Path(model_file_path.parent).mkdir(parents=True, exist_ok=True)
        torch.save(model_state_dict, model_file_path)

    def load_model(self, model, weights_file_name):
        if weights_file_name is not None:
            model_state_dict_path = self.dirs.weights_dir / weights_file_name
            if Path(model_state_dict_path).is_file():
                model.load_state_dict(torch.load(model_state_dict_path))
            else:
                print(f"Error: File '{model_state_dict_path}' not found. Initializing with random weights.")
        return model


def main():
    if not torch.cuda.is_available():
        print("Cuda not available")
        return

    num_epochs = 1
    batch_size = 128
    device = torch.device("cuda")

    spc = SpeciesModel(dataset_name="Dataset", device=device)
    spc.load_data(batch_size=batch_size, weighted_sampler=True)

    pretrained_model = spc.load_pretrained_model(model_name='resnet50', weights_file_name=None)
    model = pretrained_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters())
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

    trained_model = spc.train_model(model=model,
                                    criterion=criterion,
                                    optimizer=optimizer_ft,
                                    scheduler=exp_lr_scheduler,
                                    num_epochs=num_epochs,
                                    weights_file_name=None)


if __name__ == "__main__":
    main()
