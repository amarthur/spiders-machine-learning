import time
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from directory_structure import DirectoryStructure
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from PIL import Image
import json

class SpeciesModel:

    def __init__(self, dataset_name: str, device: torch.device) -> None:
        # Parameters
        self.dataset_name = dataset_name
        self.device = torch.device(device)

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
        self.num_classes = len(list(self.dirs.phases_dirs[self.train_phase].glob('*')))

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
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.CenterCrop(size=center_crop_size),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

    def load_data(self, batch_size, use_weighted_sampler=False):
        # Define image datasets
        self.image_datasets = {
            phase: datasets.ImageFolder(root=path, transform=self.data_transforms[phase])
            for phase, path in self.dirs.phases_dirs.items()
        }

        # Define dataloaders
        weighted_sampler = self.get_weighted_sampler() if use_weighted_sampler else None
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

        # # Freeze pre-trained parameters
        # for parameter in pretrained_model.parameters():
        #     parameter.requires_grad = False

        for name, param in pretrained_model.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False
                
        # # Replace the final layer (requires_grad=True by default)
        
        # for name, param in pretrained_model.named_parameters():
        #     if all(n not in name for n in ["features.6", "features.7"]):
        #         param.requires_grad = False
        in_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.4),
                                            nn.Linear(256, self.num_classes))
        
        # pretrained_model.fc = nn.Sequential(nn.Linear(in_features, 512),
        #                                     nn.ReLU(),
        #                                     nn.Dropout(0.25),
        #                                     nn.Linear(512, 256),
        #                                     nn.ReLU(),
        #                                     nn.Dropout(0.25),
        #                                     nn.Linear(256, self.num_classes))
        # for name, param in pretrained_model.named_parameters():
        #     if 'encoder.layers.encoder_layer_11' not in name:
        #         param.requires_grad = False
        # in_features = pretrained_model.heads.head.in_features
        
        
        # for name, param in pretrained_model.named_parameters():
        #     if all(n not in name for n in ["features.6", "features.7"]):
        #         param.requires_grad = False
        # in_features = pretrained_model.head.in_features
        # pretrained_model.head = nn.Sequential(nn.Linear(in_features, 512),
        #                                         nn.ReLU(),
        #                                         nn.Dropout(0.25),
        #                                         nn.Linear(512, 256),
        #                                         nn.ReLU(),
        #                                         nn.Dropout(0.25),
        #                                         nn.Linear(256, self.num_classes))

        # Try to load weights stored in a file (if any)
        # pretrained_model = nn.parallel.DataParallel(pretrained_model)
        pretrained_model = self.load_model_state(pretrained_model, weights_file_name)
        return pretrained_model.to(self.device)

    def train_model(self,
                    model,
                    num_epochs,
                    batch_size,
                    criterion,
                    optimizer,
                    scheduler,
                    use_weighted_sampler=False,
                    weights_file_name=None,
                    load_data=False):

        if load_data:
            print("Loading data...")
            self.load_data(batch_size, use_weighted_sampler)
            print("Loaded data")

        since = time.time()
        best_model_state = deepcopy(model.state_dict())
        best_acc = 0.0

        sched_has_args = scheduler.step.__code__.co_argcount >= 1
    
        running_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="micro").to(self.device)
        # running_precision = MulticlassPrecision(num_classes=self.num_classes, average="macro").to(self.device)
        # running_recall = MulticlassRecall(num_classes=self.num_classes + 1, average="macro").to(self.device)
        # running_f1score = MulticlassF1Score(num_classes=self.num_classes, average="macro").to(self.device)
        
        # running_accuracy_micro = MulticlassAccuracy(num_classes=self.num_classes, average="micro").to(self.device)
        # running_precision_micro = MulticlassPrecision(num_classes=self.num_classes, average="micro").to(self.device)
        # running_recall_micro = MulticlassRecall(num_classes=self.num_classes + 1, average="micro").to(self.device)
        # running_f1score_micro = MulticlassF1Score(num_classes=self.num_classes, average="micro").to(self.device)

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
                        preds = torch.argmax(outputs, dim=1)  # Get prediction
                        loss = criterion(outputs, labels)  # Compute loss
                        running_accuracy.update(preds, labels)
                        # running_precision.update(preds, labels)
                        # running_recall.update(preds.view(-1) , labels.view(-1))
                        # running_f1score.update(preds, labels)
                        
                        # running_accuracy_micro.update(preds, labels)
                        # running_precision_micro.update(preds, labels)
                        # running_recall_micro.update(preds.view(-1) , labels.view(-1))
                        # running_f1score_micro.update(preds, labels)

                        if phase == self.train_phase:
                            loss.backward()  # Backpropagate
                            optimizer.step()  # Optimize

                    # Error measures
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                ds_len = self.dataset_sizes[phase]
                epoch_loss = running_loss / ds_len
                epoch_acc = running_corrects.double() / ds_len

                epoch_accuracy = running_accuracy.compute()
                # epoch_precision = running_precision.compute()
                # epoch_recall = running_recall.compute()
                # epoch_f1score = running_f1score.compute()

                # epoch_accuracy_micro = running_accuracy_micro.compute()
                # epoch_precision_micro = running_precision_micro.compute()
                # epoch_recall_micro = running_recall_micro.compute()
                # epoch_f1score_micro = running_f1score_micro.compute()

                if phase == self.train_phase and not sched_has_args:
                    scheduler.step()
                if phase == self.valid_phase and sched_has_args:
                    scheduler.step(epoch_loss)

                print(f"Phase: {phase} | Loss: {epoch_loss:.4f}")
                print(f"Acc: {epoch_accuracy * 100:.2f}%")
                # print(f"       Acc    Prec   Rec    F1")
                # print(f"Macro: {epoch_accuracy * 100:.2f}% {epoch_precision * 100:.2f}% {epoch_recall * 100:.2f}% {epoch_f1score * 100:.2f}%")
                # print(f"Micro: {epoch_accuracy_micro * 100:.2f}% {epoch_precision_micro * 100:.2f}% {epoch_recall_micro * 100:.2f}% {epoch_f1score_micro * 100:.2f}%")
                print()
                
                
                running_accuracy.reset()
                # running_precision.reset()
                # running_recall.reset()
                # running_f1score.reset()
                
                # running_accuracy_micro.reset() 
                # running_precision_micro.reset()
                # running_recall_micro.reset()   
                # running_f1score_micro.reset()  

                # Deep copy the model
                if phase == self.valid_phase and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_state = deepcopy(model.state_dict())
                    if weights_file_name is not None:
                        self.save_model_state(best_model_state, weights_file_name)

        time_elapsed = time.time() - since
        print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc * 100:.2f}%")

        # load best model weights
        model.load_state_dict(best_model_state)
        return model

    def show_me(self, model, img_path, weights_file_name=None):        
        model.eval()
        
        if weights_file_name:
            model = self.load_model_state(model, weights_file_name).to(self.device)
        image = Image.open(img_path)
        
        model_weights = []
        conv_layers = []
        model_children = list(model.children())
        counter = 0
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)

        
        image = self.data_transforms[self.valid_phase](image)
        image = image.unsqueeze(0).to(self.device)
        
        outputs = []
        names = []
        for layer in conv_layers[0:]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))
        
        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        
        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(8, 8, i+1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(names[i].split('(')[0], fontsize=30)
        plt.savefig(img_path.stem, bbox_inches='tight')


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

        _, axes = plt.subplots(rows, cols, figsize=(15, 7))

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

    def save_model_state(self, model_state_dict, weights_file_name):
        model_file_path = Path(self.dirs.weights_dir / weights_file_name)
        Path(model_file_path.parent).mkdir(parents=True, exist_ok=True)
        torch.save(model_state_dict, model_file_path)

    def load_model_state(self, model, weights_file_name):
        if weights_file_name is not None:
            model_state_dict_path = self.dirs.weights_dir / weights_file_name
            if Path(model_state_dict_path).is_file():
                model.load_state_dict(torch.load(model_state_dict_path))
            else:
                print(f"File '{weights_file_name}' not found. Initializing with random weights.")
        return model


def main():
    device = torch.device("cuda:0")
    model_name = "convnext_base"
    model_func = getattr(models, model_name)
    pretrained_model = model_func(weights='DEFAULT')
    print(pretrained_model)
    for name, param in pretrained_model.named_parameters():
        print(name)
    #     if "features.7.0" in name:
    #         print(f"{name}")
    # spc = SpeciesModel(dataset_name="Dataset", device=device)
    # m = spc.load_pretrained_model('resnet50')
    # print(m)


if __name__ == "__main__":
    main()
