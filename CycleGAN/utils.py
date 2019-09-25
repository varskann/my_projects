"""
utils.py: General utility functions
"""

__author__ = "Kanishk Varshney"
__date__ = "Sun Sep 2 22:56:12 IST 2019"


import configparser
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def get_config(section, option, config_file="model_config.ini", _type=None):
    """
    read confic from model_config.ini
    :param section (str): section name in config file
    :param option (str): option to pick from the section
    :param _type (str): data type of the value

    :return:
        value (_type): value of the variable typecasted to _type

    :raises:
        ValueError: section / option doesn't exist
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    try:
        value = _type(config.get(section, option))

    except Exception as e:
        raise ValueError(e)

    return value


def get_class_names(config_file="model_config.ini"):
    """
    get class names and labels from config file
    :return:
        class_mapping (dict): mapping of class names and labels from model_config.ini
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    return dict(config.items('class_map'))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    :param y_true (np.array): array/list of ground truth labels
    :param y_pred (np.array): array/list of predicted labels
    :param classes (dict): list of class labels
    :param normalize (bool): Normalization can be applied by setting `normalize=True`.
    :param title (str): plot title
    :param cmap (enum): color map for the matrix
    :return:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes.keys(), yticklabels=classes.keys(),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
