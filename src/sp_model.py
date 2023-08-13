import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision import models


class SpModel(pl.LightningModule):
    def __init__(self, num_classes, criterion: torch.nn.Module = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.criterion = criterion
        self.feature_extractor, self.classifier = self.get_model()

        self.train_acc = MulticlassAccuracy(self.num_classes)
        self.train_precision = MulticlassPrecision(self.num_classes)
        self.train_recall = MulticlassRecall(self.num_classes)
        self.train_f1 = MulticlassF1Score(self.num_classes)
        self.train_conf_mat = MulticlassConfusionMatrix(self.num_classes)

        self.val_acc = MulticlassAccuracy(self.num_classes)
        self.val_precision = MulticlassPrecision(self.num_classes)
        self.val_recall = MulticlassRecall(self.num_classes)
        self.val_f1 = MulticlassF1Score(self.num_classes)
        self.val_conf_mat = MulticlassConfusionMatrix(self.num_classes)

        self.train_metrics = {
            "Train Accuracy": self.train_acc,
            "Train Precision": self.train_precision,
            "Train Recall": self.train_recall,
            "Train F1": self.train_f1,
        }

        self.val_metrics = {
            "Val Accuracy": self.val_acc,
            "Val Precision": self.val_precision,
            "Val Recall": self.val_recall,
            "Val F1": self.val_f1,
        }

    def get_model(self):
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        feature_extractor = nn.Sequential(*layers)
        classifier = nn.Sequential(
            nn.Linear(num_filters, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, self.num_classes),
        )

        return feature_extractor, classifier

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("Train Loss", loss, on_step=False, on_epoch=True)
        self.train_conf_mat(preds, y)

        for name, metric in self.train_metrics.items():
            metric(preds, y)
            self.log(name, metric, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("Val Loss", loss, on_step=False, on_epoch=True)
        self.val_conf_mat(preds, y)

        for name, metric in self.val_metrics.items():
            metric(preds, y)
            self.log(name, metric, on_step=False, on_epoch=True)

    def log_confusion_matrix(self, phase, cf, epoch):
        conf_mat = cf.compute().detach().cpu().numpy()
        title = f"Confusion Matrix: {phase}"
        self.logger.experiment.log_confusion_matrix(
            matrix=conf_mat,
            epoch=epoch,
            title=title,
            file_name=f"{title}.json",
        )

    def on_train_epoch_end(self):
        self.log_confusion_matrix("Training", self.train_conf_mat, self.current_epoch)
        self.train_conf_mat.reset()

    def on_validation_epoch_end(self):
        self.log_confusion_matrix("Validation", self.val_conf_mat, self.current_epoch)
        self.val_conf_mat.reset()
