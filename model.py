
import torch.nn 
from torch.nn import Module, Conv2d, Linear, MaxPool2d,  ReLU, BatchNorm1d, BatchNorm2d,  Dropout
from torch import flatten


class CustomLenet(Module):

    def __init__(self, nChannels, nClasses):
        super(CustomLenet, self).__init__()
        self.conv1 = Conv2d(nChannels, 32, kernel_size=5, stride=1, padding=1)
        # self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        # self.bn2 = BatchNorm2d(8)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        
        self.fc1 = Linear(62 * 62 * 8, 32)
        # self.bn3 = BatchNorm1d(32)
        self.dropout_fc1 = Dropout(0.5)
        self.fc2 = Linear(32, 16)
        # self.bn4 = BatchNorm1d(16)
        # self.dropout_fc2 = Dropout(0.5)
        self.fc3 = Linear(16, nClasses)
       

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = flatten(x, 1)

        x = self.fc1(x)
        # x = self.bn3(x)
        # x = self.dropout_fc1(x)

        x = self.fc2(x)
        # x = self.bn4(x)
        # x = self.dropout_fc2(x)

        x = self.fc3(x)
        return x
    

# from torch.nn import Module, Conv2d, Linear, MaxPool2d,  ReLU, BatchNorm1d, BatchNorm2d,  Dropout
# from torch import flatten

# import torch
# from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, BatchNorm1d, BatchNorm2d, Dropout, Dropout2d
# from torch import flatten

class VGGNet(Module):

    def __init__(self, nChanels, nClasses):
        super(VGGNet, self).__init__()

        # --- Bloque 1 ---
        self.conv1_1 = Conv2d(nChanels, 8, kernel_size=3, padding=1)
        self.conv1_2 = Conv2d(8, 8, kernel_size=3, padding=1)
        self.batchnorm2d_1 = BatchNorm2d(8)
        # self.dropout2d_1 = Dropout2d(0.2)

        # --- Bloque 2 ---
        self.conv2_1 = Conv2d(8, 16, kernel_size=3, padding=1) 
        self.conv2_2 = Conv2d(16, 16, kernel_size=3, padding=1) 
        self.batchnorm2d_2 = BatchNorm2d(16)
        # self.dropout2d_2 = Dropout2d(0.25)

        # --- Bloque 3 ---
        self.conv3_1 = Conv2d(16, 32, kernel_size=3, padding=1) 
        self.conv3_2 = Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3_3 = Conv2d(32, 32, kernel_size=3, padding=1)  
        self.batchnorm2d_3 = BatchNorm2d(32)
        # self.dropout2d_3 = Dropout2d(0.3)

        # --- Bloque 4 ---
        self.conv4_1 = Conv2d(32, 64, kernel_size=3, padding=1) 
        self.conv4_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4_3 = Conv2d(64, 64, kernel_size=3, padding=1)  
        self.batchnorm2d_4 = BatchNorm2d(64)
        # self.dropout2d_4 = Dropout2d(0.35)

        # --- Bloque 5 ---
        self.conv5_1 = Conv2d(64, 64, kernel_size=3, padding=1) 
        self.conv5_2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5_3 = Conv2d(64, 64, kernel_size=3, padding=1)  
        self.batchnorm2d_5 = BatchNorm2d(64)
        # self.dropout2d_5 = Dropout2d(0.4)

        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()
        
        # --- Fully Connected ---
        self.fc1 = Linear(64 * 7 * 7,64) 
        self.bn_fc1 = BatchNorm1d(64)
        self.dropout_fc1 = Dropout(0.9)

        self.fc2 = Linear(64, 16)
        self.bn_fc2 = BatchNorm1d(16)
        self.dropout_fc2 = Dropout(0.9)

        self.fc3 = Linear(16, nClasses)


    def forward(self, x):
        # --- Bloque 1 ---
        x = self.relu(self.batchnorm2d_1(self.conv1_1(x)))
        # x = self.dropout2d_1(x)
        x = self.relu(self.batchnorm2d_1(self.conv1_2(x)))
        # x = self.dropout2d_1(x)
        x = self.maxpool(x)

        # --- Bloque 2 ---
        x = self.relu(self.batchnorm2d_2(self.conv2_1(x)))
        # x = self.dropout2d_2(x)
        x = self.relu(self.batchnorm2d_2(self.conv2_2(x)))
        # x = self.dropout2d_2(x)
        x = self.maxpool(x)

        # --- Bloque 3 ---
        x = self.relu(self.batchnorm2d_3(self.conv3_1(x)))
        # x = self.dropout2d_3(x)
        x = self.relu(self.batchnorm2d_3(self.conv3_2(x)))
        # x = self.dropout2d_3(x)
        x = self.relu(self.batchnorm2d_3(self.conv3_3(x)))
        # x = self.dropout2d_3(x)
        x = self.maxpool(x)

        # --- Bloque 4 ---
        x = self.relu(self.batchnorm2d_4(self.conv4_1(x)))
        # x = self.dropout2d_4(x)
        x = self.relu(self.batchnorm2d_4(self.conv4_2(x)))
        # x = self.dropout2d_4(x)
        x = self.relu(self.batchnorm2d_4(self.conv4_3(x)))
        # x = self.dropout2d_4(x)
        x = self.maxpool(x)

        # --- Bloque 5 ---
        x = self.relu(self.batchnorm2d_5(self.conv5_1(x)))
        # x = self.dropout2d_5(x)
        x = self.relu(self.batchnorm2d_5(self.conv5_2(x)))
        # x = self.dropout2d_5(x)
        x = self.relu(self.batchnorm2d_5(self.conv5_3(x)))
        # x = self.dropout2d_5(x)
        x = self.maxpool(x)

        # --- Fully connected ---
        x = flatten(x,1)

        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)

        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)

        x = self.fc3(x)
        return x


# import torch.nn as nn
# from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, Flatten, Dropout2d, BatchNorm2d, BatchNorm1d

# class CustomLenet(Module):

#     def __init__(self, nChannels, nClasses):
#         super(CustomLenet, self).__init__()
#         self.conv1 = Conv2d(nChannels, 32, kernel_size=5, stride=1, padding=1)
#         self.bn1 = BatchNorm2d(32)
#         self.conv2 = Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
#         self.bn2 = BatchNorm2d(8)
#         self.maxpool = MaxPool2d(kernel_size=2, stride=2)
#         self.relu = ReLU()
#         self.fc1 = Linear(6*6*8, 32)
#         self.bn3 = BatchNorm1d(32)
#         self.dropout_fc1 = Dropout2d(0.5)
#         self.fc2 = Linear(32, 16)
#         self.dropout_fc2 = Dropout2d(0.5)
#         self.bn4 = BatchNorm1d(16)
#         self.fc3 = Linear(16, nClasses)
#         self.flatten = nn.Flatten()

#     def forward(self, x):
        
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.flatten(x)
        
#         x = self.fc1(x)
#         x = self.bn3(x)
#         x = self.dropout_fc1(x)

#         x = self.fc2(x)
#         x = self.bn4(x)
#         x = self.dropout_fc2(x)
    
#         x = self.fc3(x)
#         return x