import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

class lenet_mnist(nn.Module):
    def __init__(self):
        super(lenet_mnist, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(1, 20, 5),
        nn.BatchNorm2d(20),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(20, 50, 5),
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        )

        self.classifier = nn.Sequential(
        nn.Linear(4*4*50, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        )

    def forward(self, input):
        output_c = self.features(input)
        output = self.classifier(output_c.view(-1, 4*4*50))
        return output
#class lenet_mnist(nn.Module):
#    def __init__(self):
#        super(lenet_mnist, self).__init__()
#        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
#        self.relu1 = nn.ReLU()
#        self.pool1 = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
#        self.relu2 = nn.ReLU()
#        self.pool2 = nn.MaxPool2d(2, 2)
#        #self.conv2_drop = nn.Dropout2d()
#        self.fc1 = nn.Linear(4*4*50, 64)
#        self.relu3 = nn.ReLU()
#        self.fc2 = nn.Linear(64, 10)
#        #self.relu = nn.ReLU(inplace = True)
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.relu1(x)
#        x = self.pool1(x)
#        x = self.conv2(x)
#        x = self.relu2(x)
#        x = self.pool2(x)
#        x = x.view(-1, 4*4*50)
#        x = self.fc1(x)
#        x = self.relu3(x)
#        #x = F.dropout(x, training=self.training)
#        x = self.fc2(x)
#        return x
        #return F.log_softmax(x, dim=1)
"""
class lenet_mnist(nn.Module):
    def __init__(self):
        super(lenet_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*16, 10)
        #self.relu3 = nn.ReLU()
        #self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 4*4*16)
        x = self.fc1(x)
        #x = self.relu3(x)
        #x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""