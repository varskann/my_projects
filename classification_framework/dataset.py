"""
dataset.py: Dataset loader(class) goes here
"""

__author__ = "Kanishk Varshney"
__date__ = "Sun Sep  1 22:56:12 IST 2019"

import os
import csv

import cv2
import torch
from torch.utils import data

class Dataset(data.Dataset):
    """characterizes a dataset for PyTorch"""

    def __init__(self, root_dir, filepath, delimiter=","):
        """
        initialization
        :param root_dir (str): path to training/validation/test data images
        :param filepath (str): path to txt / csv file containing image, label list
        """
        self.root_dir = root_dir
        self.filepath = filepath
        self.data = {}

        with open(self.filepath) as fdata:
            reader = csv.reader(fdata, delimiter=delimiter)
            for idx, row in enumerate(reader):
                self.data[idx]  = [row[0], int(row[1])]

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
        image = cv2.imread(os.path.join(self.root_dir, image_name))
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        return {"image": image, "label": label}
