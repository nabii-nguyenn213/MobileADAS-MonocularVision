import torch

from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights, 
)


class ResNetBackbone(torch.nn.Module):

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
    ):
        super().__init__()

        if backbone == "resnet18":
            weights = (
                ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )

            model = resnet18(
                weights=weights
            )

        elif backbone == "resnet34":
            weights = (
                ResNet34_Weights.DEFAULT
                if pretrained
                else None
            )

            model = resnet34(
                weights=weights
            )

        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(
        self,
        x: torch.Tensor,
    ):

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        feature1 = self.layer1(x)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)

        return [
            feature1,
            feature2,
            feature3,
            feature4,
        ]
