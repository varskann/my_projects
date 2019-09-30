"""
models.py: model defnition(class) goes here
"""
__author__ = "Kanishk Varshney"
__date__ = "Sun Sep 24 22:56:12 IST 2019"

import sys
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    define model architectures
    """
    def __init__(self):
        super(Generator, self).__init__()

        ## Downsizing layers
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3, stride=1)
        self.bn1 = nn.InstanceNorm2d(64)
        self.reflectPad1 = nn.ReflectionPad2d(3)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn2 = nn.InstanceNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 7, padding=3, stride=2)
        self.bn3 = nn.InstanceNorm2d(256)

        self.conv_up1 = nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=1)
        self.bn_up1 = nn.InstanceNorm2d(128)

        self.conv_up2 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1)
        self.bn_up2 = nn.InstanceNorm2d(64)

        self.conv_up3 = nn.ConvTranspose2d(64, 3, 7)
        self.reflectPad_up3 = nn.ReflectionPad2d(3)


    def residualBlock(self, x):
        self.conv = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.bn = nn.InstanceNorm2d(256)
        self.reflectPad = nn.ReflectionPad2d(1)

        return self.bn(self.conv(self.reflectPad(F.relu(self.bn(self.conv(self.reflectPad(x)))))))

    def forward(self, x):
        """
        forward pass for the model
        :param x (np.array): _input image for the forward pass
        :return:
            network output after the forward pass
        """
        sys.stdout.flush()
        ##Downsizing
        x = F.relu(self.bn1(self.conv1(self.reflectPad1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        for _ in range(9):
            x = F.relu(x + self.residualBlock(x))
        ##Upsizing
        x = F.relu(self.bn_up1(self.conv_up1(x)))
        x = F.relu(self.bn_up2(self.conv_up2(x)))
        x = nn.Tanh(self.conv_up3(self.reflectPad_up3(x)))
        return x


class Descriminator(nn.Module):
    """define descriminator network"""

    def __init__(self):
        super(Descriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, padding=1, stride=2)

        self.conv2 = nn.Conv2d(64, 128, 4, padding=1, stride=2)
        self.bn2 = nn.InstanceNorm2d(128)

        self.conv3 = nn.Conv2d(128, 246, 4, padding=1, stride=2)
        self.bn3 = nn.InstanceNorm2d(128)

        self.conv4 = nn.Conv2d(256, 512, 4, padding=1, stride=2)
        self.bn4 = nn.InstanceNorm2d(128)

        self.conv5 = nn.Conv2d(512, 1, 4, padding=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        return x
