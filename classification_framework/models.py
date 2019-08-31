"""
models.py: model defnition(class) goes here
"""
__author__ = "Kanishk Varshney"
__date__ = "Sun Sep  1 22:56:12 IST 2019"

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    define model architectures
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        forward pass for the model
        :param x (np.array): _input image for the forward pass
        :return:
            network output after the forward pass
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
