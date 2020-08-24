import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Bottkeneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottkeneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=planes * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(num_features=self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, planes=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, num_blocks=num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)

        # lateral layers
        self.lateral_layer1 = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.lateral_layer2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.lateral_layer3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        # smooth layers
        self.smooth1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """
        Upsample and add two feature maps
        :param x: top feature map to be upsampled
        :param y: lateral feature map
        :return: added feature map
        Use the 'bilinear' mode to upsample
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
        #return F.upsample(x, size=(H, W), mode='nearest') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.lateral_layer1(c5)
        p4 = self._upsample_add(p5, self.lateral_layer2(c4))
        p3 = self._upsample_add(p4, self.lateral_layer3(c3))
        # smooth layer
        p5 = self.smooth1(p5)
        p4 = self.smooth2(p4)
        p3 = self.smooth3(p3)
        return p3, p4, p5, p6, p7


def getFPN(name):
    if name =="FPN50":
        return FPN(Bottkeneck, [3, 4, 6, 3])
    elif name == 'FPN101':
        return FPN(Bottkeneck, [2, 4, 23, 3])
    else:
        raise name +" is not implemented. Try using 'FPN50' or 'FPN101'" 

