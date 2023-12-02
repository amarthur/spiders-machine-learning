import pytorch_lightning as pl
import torch
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassConfusionMatrix,
                                         MulticlassF1Score)

import sp_models


class SpModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = None,
        model_params: dict = None,
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])  # Save init args

        self.num_classes = num_classes
        self.model = self.get_model(model_params)
        self.criterion = criterion

        self.train_macro_acc = MulticlassAccuracy(self.num_classes, average="macro")
        self.train_micro_acc = MulticlassAccuracy(self.num_classes, average="micro")
        self.train_macro_f1 = MulticlassF1Score(self.num_classes, average="macro")

        self.val_macro_acc = MulticlassAccuracy(self.num_classes, average="macro")
        self.val_micro_acc = MulticlassAccuracy(self.num_classes, average="micro")
        self.val_macro_f1 = MulticlassF1Score(self.num_classes, average="macro")
        self.val_conf_mat = MulticlassConfusionMatrix(self.num_classes)

        self.test_macro_acc = MulticlassAccuracy(self.num_classes, average="macro")
        self.test_micro_acc = MulticlassAccuracy(self.num_classes, average="micro")
        self.test_macro_f1 = MulticlassF1Score(self.num_classes, average="macro")
        self.test_conf_mat = MulticlassConfusionMatrix(self.num_classes)

        self.train_metrics = {
            "Train Macro Accuracy": self.train_macro_acc,
            "Train Micro Accuracy": self.train_micro_acc,
            "Train Macro F1": self.train_macro_f1,
        }

        self.val_metrics = {
            "Val Macro Accuracy": self.val_macro_acc,
            "Val Micro Accuracy": self.val_micro_acc,
            "Val Macro F1": self.val_macro_f1,
        }

        self.test_metrics = {
            "Test Macro Accuracy": self.test_macro_acc,
            "Test Micro Accuracy": self.test_micro_acc,
            "Test Macro F1": self.test_macro_f1,
        }

    def get_model(self, model_params):
        model_type = model_params["model_type"]
        architecture = model_params["architecture"]
        unfreeze_layers = model_params.get("unfreeze_layers", [])
        weights = model_params.get("weights", "DEFAULT")

        model_mapping = {
            "resnet": sp_models.SpResnet,
            "convnet": sp_models.SpConv,
            "vit": sp_models.SpViT,
            "maxvit": sp_models.SpMaxvit,
            "swin": sp_models.SpSwin,
        }

        try:
            ModelClass = model_mapping[model_type]
        except KeyError:
            raise ValueError(f"Unknown model_type: {model_type}")

        model_args = (self.num_classes, architecture, unfreeze_layers, weights)
        sp_model = ModelClass(*model_args)
        return sp_model.pretrained_model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("Train Loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for name, metric in self.train_metrics.items():
            metric(preds, y)
            self.log(name, metric, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("Val Loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_conf_mat(preds, y)

        for name, metric in self.val_metrics.items():
            metric(preds, y)
            self.log(name, metric, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)

        self.log("Test Loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        for name, metric in self.test_metrics.items():
            metric(preds, y)
            self.log(name, metric, on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        preds = self(x)
        return preds

    def log_conf_mat(self, conf_matrix, phase):
        cf = conf_matrix.compute().detach().cpu().numpy()
        if self.logger:
            self.logger.experiment.log_confusion_matrix(
                matrix=cf,
                epoch=self.current_epoch,
                title=f"{phase} Confusion Matrix",
                file_name=f"{phase}-confusion-matrix.json",
            )
        conf_matrix.reset()

    def on_validation_epoch_end(self):
        self.log_conf_mat(self.val_conf_mat, "Val")

    def on_test_epoch_end(self) -> None:
        self.log_conf_mat(self.test_conf_mat, "Test")
