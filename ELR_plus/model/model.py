import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from .ResNet_Zoo import ResNet, BasicBlock
from .PreResNet import PreActResNet, PreActBlock
import torchvision.models as models
from .InceptionResNetV2 import InceptionResNetV2


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    #return models.resnet34(num_classes=10)


def resnet50(num_classes=14):
    import torchvision.models as models
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def PreActResNet34(num_classes=10) -> PreActResNet:
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)
def PreActResNet18(num_classes=10) -> PreActResNet:
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)
