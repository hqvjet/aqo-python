import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)

        # Shortcut layer to match dimensions
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.GroupNorm(num_groups=32, num_channels=out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity  # Add residual connection
        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, input_size, num_res_blocks=6):
        super(CNN, self).__init__()
        self.name = "CNN"

        # Initial Convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=16, num_channels=64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Residual Blocks
        self.res_blocks = nn.ModuleList()
        in_channels, out_channels = 64, 64
        for i in range(num_res_blocks):
            if i == num_res_blocks // 2:  # Increase channel size halfway
                out_channels *= 2
            self.res_blocks.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(out_channels, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # Initial Convolution
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Apply ResNet blocks
        for block in self.res_blocks:
            x = block(x)

        # Global Average Pooling
        x = self.global_pool(x).squeeze(-1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


