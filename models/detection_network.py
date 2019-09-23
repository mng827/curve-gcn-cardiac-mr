import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectionNetwork(nn.Module):
    def __init__(self, feats_grid_size, in_channels, hidden_channels, out_channels):
        super(DetectionNetwork, self).__init__()
        self.feats_grid_size = feats_grid_size

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = 3,
            padding = 1
        )

        self.fc = nn.Linear(
            in_features =(feats_grid_size // 2) ** 2 * hidden_channels,
            out_features = out_channels
        )


    def forward(self, features):
        batch_size = features.size(0)
        x = self.conv(features)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.fc(x.view(batch_size, -1))

        return x
