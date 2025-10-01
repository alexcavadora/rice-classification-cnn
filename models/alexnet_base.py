from torch.nn import Module, MaxPool2d, Conv2d, Linear, ReLU, BatchNorm2d, BatchNorm1d, Dropout
from torch import flatten

class AlexNet(Module):
    def __init__(self, nChannels, nClases):
        super(AlexNet, self).__init__()

        self.conv1 = Conv2d(nChannels,96, 11, stride = 4, padding=2)
        self.conv2 = Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5 = Conv2d(384, 256, 3, stride=1, padding=1)
        self.maxpool = MaxPool2d(3, stride=2)
        self.relu = ReLU()
        self.fc1  = Linear(256*6*6, 4096)
        self.fc2  = Linear(4096, 4096)
        self.fc3  = Linear(4096, nClases)
        self.batchnorm1d_1 = BatchNorm1d(4096)
        self.batchnorm1d_2 = BatchNorm1d(4096)
        self.dropout = Dropout(0.5)
        self.batchnorm2d_1 = BatchNorm2d(96)
        self.batchnorm2d_2 = BatchNorm2d(256)
        self.batchnorm2d_3 = BatchNorm2d(384)
        self.bacthnorm2d_4 = BatchNorm2d(384)
        self.bacthnorm2d_5 = BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm2d_1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.batchnorm2d_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batchnorm2d_3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bacthnorm2d_4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bacthnorm2d_5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = flatten(x, 1)

        x = self.fc1(x)
        x = self.batchnorm1d_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batchnorm1d_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x
