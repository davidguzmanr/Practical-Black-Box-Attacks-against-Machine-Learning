"""
BlackBoxModel architecture for MNIST dataset. It is kinda like AlexNet 
but modified for MNIST. This will be used as an oracle for the substitute model.
"""

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
class BlackBoxModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(BlackBoxModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        # out = F.log_softmax(out, dim=1)

        return out
