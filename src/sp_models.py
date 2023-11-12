import torch.nn as nn
from torchvision import models


class PretrainedModel:
    def __init__(self, num_classes, model_name, grad_layers, weights = "DEFAULT"):
        self.num_classes = num_classes
        self.grad_layers = grad_layers

        self.model = getattr(models, model_name)
        self.pretrained_model = self.model(weights=weights)

    def get_default_classifier(self, in_features: int, dropout: float = 0.45) -> nn.Sequential:
        classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.num_classes),
        )
        return classifier

    def freeze_layers(self) -> None:
        for name, param in self.pretrained_model.named_parameters():
            if all(grad_layer not in name for grad_layer in self.grad_layers):
                param.requires_grad = False


class SpResnet(PretrainedModel):
    def __init__(self, *args):
        super().__init__(*args)
        in_features = self.pretrained_model.fc.in_features
        self.freeze_layers()
        self.pretrained_model.fc = self.get_default_classifier(in_features)


class SpConv(PretrainedModel):
    def __init__(self, *args):
        super().__init__(*args)
        in_features = self.pretrained_model.classifier[2].in_features
        self.freeze_layers()
        self.pretrained_model.classifier[2] = self.get_default_classifier(in_features)

class SpSwin(PretrainedModel):
    def __init__(self, *args):
        super().__init__(*args)
        in_features = self.pretrained_model.head.in_features
        self.freeze_layers()
        self.pretrained_model.head = self.get_default_classifier(in_features)
        
class SpMaxvit(PretrainedModel):
    def __init__(self, *args):
        super().__init__(*args)
        in_features = self.pretrained_model.classifier[5].in_features
        self.freeze_layers()
        self.pretrained_model.classifier[5] = self.get_default_classifier(in_features)

class SpViT(PretrainedModel):
    def __init__(self, *args):
        super().__init__(*args)
        in_features = self.pretrained_model.heads.head.in_features
        self.freeze_layers()
        self.pretrained_model.heads.head = self.get_default_classifier(in_features)
