import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.model(x)
