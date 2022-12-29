import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        N = x.size(0)
        x = x.view(N, -1)
        scores = self.classifier(x)
        return scores

