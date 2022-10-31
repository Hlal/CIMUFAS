'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG8': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 1024, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_nobatch': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG_k2': [64, 64, 64, 64, 'M', 64, 64, 'M', 64, 64],
}


class vgg_cifar10(nn.Module):
    def __init__(self, vgg_name='VGG16'):
        super(vgg_cifar10, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], vgg_name)
        self.classifier = nn.Linear(1024, 10)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 1024),
		# 	nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
		# 	nn.ReLU(inplace=True),
        #     nn.Linear(1024, 10)
        # )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #print(out.size)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, vgg_name):
        layers = []
        in_channels = 3
        if 'nobatch' in vgg_name:
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        elif 'k2' in vgg_name:
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=2, padding=0),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        else:
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        
        return nn.Sequential(*layers)

#def vgg8_cifar10():
#    return vgg_cifar10('VGG8')
class vgg8_cifar10(nn.Module):
    def __init__(self):
        super(vgg8_cifar10, self).__init__()
        self.features =  nn.Sequential(
              nn.Conv2d(3, 128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(128, 256, kernel_size=3, padding=1),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, padding=1),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(256, 512, kernel_size=3, padding=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 512, kernel_size=3, padding=1),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(512, 1024, kernel_size=3, padding=0),
              nn.BatchNorm2d(1024),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(1024, 10)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 1024),
		# 	nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
		# 	nn.ReLU(inplace=True),
        #     nn.Linear(1024, 10)
        # )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        #print(out.size)
        out = self.classifier(out)
        return out

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
