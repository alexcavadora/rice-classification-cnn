import torch
from torch import nn
from torch import flatten
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, nChannels, nClases):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(nChannels, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, nClases)

    def forward(self, x):
        # Input: nChannels x H x W
        x = torch.tanh(self.conv1(x))      # 6 x (H-4) x (W-4)
        x = F.avg_pool2d(x, 2, 2)          # 6 x (H-4)/2 x (W-4)/2
        x = torch.tanh(self.conv2(x))      # 16 x (H-12)/2 x (W-12)/2
        x = F.avg_pool2d(x, 2, 2)          # 16 x (H-12)/4 x (W-12)/4
        x = torch.tanh(self.conv3(x))      # 120 x 1 x 1 (or close)
        x = x.view(x.size(0), -1)          # Flatten
        x = torch.tanh(self.fc1(x))        # 84
        x = self.fc2(x)                    # nClases
        return x

class LeNetModified(nn.Module):
    def __init__(self, nChannels, nClases):
        super(LeNetModified, self).__init__()

        self.conv1 = nn.Conv2d(nChannels, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nClases)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
