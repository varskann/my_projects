"""
download and extract MNIST dataset:
relevant reads:
    - http://yann.lecun.com/exdb/mnist/
    - https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
"""
__author__ = "KanishkVarshney"
__date__ = "Sat Aug 31 19:56:32 IST 2019"

import os
import csv
import gzip
import struct
import argparse

import cv2
import requests
import numpy as np

def download(urls, outdir):
    """
    download images and labels binary
    :param urls (list): urls to download images and labels from: [imagesurl, labelsurl]
    :param outdir (str): output directory to download data to
    :return:
    """
    for url in urls:
        filepath = os.path.join(outdir, os.path.basename(url))
        if not os.path.exists(filepath):
            res = requests.get(url)
            if res.status_code == 200:
                with open(filepath, 'wb') as fout:
                    fout.write(res.content)


def load(paths, outdir, label):
    """
    load data from ubyte files

    :param paths (list): urls to download images and labels from: [imagesurl, labelsurl]
    :param outdir (str): output directory to download data to

    :return:

    :raises RuntimeError: raises this excpetion if labels and images size mismatch
    """
    image_path, labels_path = paths

    ## setup output paths
    labels_out_path = os.path.join(outdir, "labels_{}.txt".format(label))
    if os.path.exists(labels_out_path):
        os.remove(labels_out_path)

    data_out_path = os.path.join(outdir, label)
    if not os.path.exists(data_out_path):
        os.makedirs(data_out_path)

    with open(labels_out_path, "a+") as out_flabels:
        writer = csv.writer(out_flabels, delimiter=",")
        with gzip.open(image_path) as fimages, gzip.open(labels_path) as flabels:
            ## read out first 4 bytes - magic numbers
            fimages.read(4)
            flabels.read(4)

            ## read number of elements (# images == #labels)
            num_labels, = struct.unpack('>i', flabels.read(4))
            num_images, = struct.unpack('>i', fimages.read(4))
            if num_images != num_labels:
                raise RuntimeError('Mismatch between images and labels count (%s != %s)'%(num_images, num_labels))

            ## read-out shape of images(next two-4 bytes)
            im_width, = struct.unpack(">i", fimages.read(4))
            im_height, = struct.unpack(">i", fimages.read(4))

            temp_image = np.empty((im_width, im_height), dtype=np.uint8)

            for i in range(num_labels):
                filename = "{}/{}.jpg".format(label, i)

                temp_label = ord(flabels.read(1))
                writer.writerow([filename, temp_label])

                for _px in range(im_width):
                    for _py in range(im_height):
                        temp_image[_px][_py] = ord(fimages.read(1))

                cv2.imwrite(os.path.join(outdir, filename), temp_image)


def pipeline(outdir, label="train"):
    """
    pipeline takes care of complete
    download and extraction of images and labels
    :param outdir (str): directory path to save data tourls
    :param label (str): train/test
    :return:
    """
    if label == "test":
        label = "t10k"

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ## download MNIST binaries
    urls = ["http://yann.lecun.com/exdb/mnist/{}-images-idx3-ubyte.gz".format(label),
            "http://yann.lecun.com/exdb/mnist/{}-labels-idx1-ubyte.gz".format(label)]

    download(urls, outdir)

    ## extract and load MNIST binaries
    paths = [os.path.join(outdir, os.path.basename(urls[0])),
             os.path.join(outdir, os.path.basename(urls[1]))]

    load(paths, outdir, label)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download MNIST dataset",
                                     usage="python download_mnist.py --outdir <path(optional)>")

    parser.add_argument("--outdir", default="mnist_data", type=str, help="path to store the dataset")

    args = parser.parse_args()

    pipeline(args.outdir, label="train")
    pipeline(args.outdir, label="test")
