"""
dataset.py: Dataset loader(class) goes here
"""

__author__ = "Kanishk Varshney"
__date__ = "Tue Sep 10 20:50:51 IST 2019"

import os
import csv
import glob

import cv2
import torch
from torch.utils import data

class Dataset(data.Dataset):
    """Load CycleGAN dataset"""

    def __init__(self, root_dir, datatype="trainA", datafolder="horse2zebra"):
        """
        initialization
        :param root_dir (str): path to training/validation/test data images for class A
        :param datatype (str): train / test
        :param datafolder (str): task / data name
        """
        self.root_dir = root_dir
        self.datatype = datatype
        self.data = {}

        self.datafolder = "{}/{}/{}".format(root_dir, datafolder, datatype)
        images = os.listdir(self.datafolder)

        if datatype in ["trainA", "testA"]:
            label = 0
        elif datatype in ["trainB", "testB"]:
            label = 1

        for image in images:
            self.data[idx] = [image, label]

    def __len__(self):
        """denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """
        generates one sample of data
        :param index (int): index of the data point / sample to fetch
                            data loader implicitly calls this function
        :return:
            sample (dict): {"image": <np.array>, "label": <int>}
        """
        ## Select sample
        image_name, label = self.data[index]

        ## load image
        image = cv2.imread(os.path.join(self.datafolder, image_name))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        return {"image": image, "label": label}
