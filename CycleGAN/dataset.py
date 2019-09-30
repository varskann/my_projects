"""
dataset.py: Dataset loader(class) goes here
"""

__author__ = "Kanishk Varshney"
__date__ = "Tue Sep 10 20:50:51 IST 2019"

import os
import csv
import glob
import random

import cv2
import torch
from torch.utils import data

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class ImageDataset(Dataset):
    """Load CycleGAN dataset"""

    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        image_A = cv2.imread(self.files_A[index % len(self.files_A)]).transpose((2, 0, 1))
        image_A = torch.from_numpy(image_A).float()
        image_A = F.to_pil_image(image_A)
        item_A = self.transform(image_A)

        if self.unaligned:
            image_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B) - 1)]).transpose((2, 0, 1))
            image_B = torch.from_numpy(image_B).float()
            image_B = F.to_pil_image(image_B)
            item_B = self.transform(image_B)
        else:
            item_B = self.transform(cv2.imread(self.files_B[index % len(self.files_B)]).transpose((2, 0, 1)))

        # if list(item_A.shape) != [3,256,256] or list(item_B.shape) !=[3,256,256]:
        #     print(self.files_B[random.randint(0, len(self.files_B) - 1)])
        #     print("item_A=%s; item_B=%s" % (item_A.shape, item_B.shape))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


# class Dataset(data.Dataset):
#     """Load CycleGAN dataset"""
#
#     def __init__(self, root_dir, datatype="trainA", datafolder="horse2zebra"):
#         """
#         initialization
#         :param root_dir (str): path to training/validation/test data images for class A
#         :param datatype (str): train / test
#         :param datafolder (str): task / data name
#         """
#         self.root_dir = root_dir
#         self.datatype = datatype
#         self.data = {}
#
#         self.datafolder = "{}/{}/{}".format(root_dir, datafolder, datatype)
#         images = os.listdir(self.datafolder)
#
#         if datatype in ["trainA", "testA"]:
#             label = 0
#         elif datatype in ["trainB", "testB"]:
#             label = 1
#
#         for idx, image in enumerate(images):
#             self.data[idx] = [image, label]
#
#     def __len__(self):
#         """denotes the total number of samples"""
#         return len(self.data)
#
#     def __getitem__(self, index):
#         """
#         generates one sample of data
#         :param index (int): index of the data point / sample to fetch
#                             data loader implicitly calls this function
#         :return:
#             sample (dict): {"image": <np.array>, "label": <int>}
#         """
#         ## Select sample
#         image_name, label = self.data[index]
#
#         ## load image
#         image = cv2.imread(os.path.join(self.datafolder, image_name))
#         image = image.transpose((2, 0, 1))
#         image = torch.from_numpy(image).float()
#
#         return {"image": image, "label": label}