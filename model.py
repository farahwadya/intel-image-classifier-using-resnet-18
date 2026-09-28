import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,  # fixed to avoid miniizing the image again
            padding=1,
            bias=False
        )

        # no relu cuz we want to sum before add non-linearity
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(identity)

        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, num_classes=5):

        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(  # stemblock
            3,  # in_channels(RGB)
            64,  # out_channels = numofilter = num of ftr map
            kernel_size=7,  # filsize=7*7*3
            stride=2,  # 2pixperstep
            padding=3,  # based on rule of k-1/2
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(  # minimize size, keep the highest num on the sapce
            kernel_size=3,
            stride=2,  # reduce size to the half
            padding=1
        )

        self.layer1 = self.make_layer(  # first layer
            64,
            64,
            blocks=2,
            stride=1
        )

        self.layer2 = self.make_layer(  # seclayer
            64,
            128,
            blocks=2,
            stride=2
        )

        self.layer3 = self.make_layer(  # third layer
            128,
            256,
            blocks=2,
            stride=2
        )

        self.layer4 = self.make_layer(  # fourthlayer
            256,
            512,
            blocks=2,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # adaptive to fix img size 521*1*1

        self.fc = nn.Linear(  # dense or fullyconnected --> tie label with img
            512,  # out of last layer
            num_classes
        )

    def make_layer(self, in_channels, out_channels, blocks, stride):

        layers = []

        layers.append(  # resblock1
            ResidualBlock(
                in_channels,
                out_channels,
                stride
            )
        )

        for _ in range(
                1, blocks):  # this loops is used incase of there is more than 1 but here only one cycle will rounded

            layers.append(  # resblock2
                ResidualBlock(
                    out_channels,
                    out_channels
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x
