"""
train.py: training and evaluation logic goes here
"""

__author__ = "Kanishk Varshney"
__date__ = "Sun Sep  1 22:56:12 IST 2019"

import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
# import torchvision

import wandb
from sklearn.metrics import confusion_matrix
import numpy as np

import utils
import models
import dataset


def main(args):
    """
    main function to trigger training
    :param args (argparse.Namespace): command line arguments
    :return:
    """
    ## read all parameters and call the training function
    params = dict()
    params["model_name"] = args.model_name
    wandb.init(config=args, project="my-project")

    ## read_config
    learning_rate = utils.get_config("training", "learning_rate", _type=float)
    momentum = utils.get_config("training", "momentum", _type=float)
    num_workers = utils.get_config("training", "num_workers", _type=int)
    train_batch_size = utils.get_config("training", "batch_size", _type=int)
    val_batch_size = utils.get_config("validation", "batch_size", _type=int)
    max_epochs = utils.get_config("training", "epoch", _type=int)
    params["num_classes"] = utils.get_config("classes", "num_classes", _type=int)
    params["log_interval"] = utils.get_config("code", "log_interval", _type=int)
    params["classes"] = utils.get_class_names()

    ## fetch the model
    params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["net"] = models.Model().to(params["device"])
    wandb.watch(params["net"], log="all")

    params["optimizer"] = optim.SGD(params["net"].parameters(),
                                    lr=learning_rate,
                                    momentum=momentum)

    params["loss_fn"] = nn.CrossEntropyLoss()


    ## load data
    train_data = dataset.Dataset(root_dir=args.data, filepath=args.train)
    params["train_data_loader"] = data.DataLoader(train_data,
                                                  num_workers=num_workers,
                                                  batch_size=train_batch_size,
                                                  shuffle=True,
                                                  pin_memory=True)

    val_data = dataset.Dataset(root_dir=args.data, filepath=args.validation)
    params["val_data_loader"] = data.DataLoader(val_data,
                                                num_workers=num_workers,
                                                batch_size=val_batch_size,
                                                shuffle=True,
                                                pin_memory=True)

    ## trigger training
    for _epoch in range(max_epochs):
        train(params, _epoch)
        test(params, _epoch)


def train(params, epoch):
    """
    train the network

    :param params (dict): dictionary holding training parameters
    :param epoch (int): epoch counter

    :return:
    """
    ## set mode to train
    params["net"].train()
    train_loss_total = 0.0
    correct_predictions_total = 0.0
    correct_arr = [0.0] * params["num_classes"]
    total_arr = [0.0] * params["num_classes"]
    target_labels_epoch = []
    predicted_labels_epoch = []
    for batch_idx, sample in enumerate(params["train_data_loader"]):
        images = sample["image"].to(params["device"])
        target_labels = sample["label"].to(params["device"])
        target_labels_epoch.extend(target_labels)

        ## train the network iteration
        params["optimizer"].zero_grad()
        output_probs = params["net"](images)
        train_loss = params["loss_fn"](output_probs, target_labels)

        train_loss.backward()
        params["optimizer"].step()

        # calculate and log TRAINING performance metrics (PER ITERATION)
        _, predicted_labels = torch.max(output_probs.data, 1)
        predicted_labels_epoch.extend(predicted_labels)

        correct_predictions = (predicted_labels == target_labels).sum()
        total_predictions = target_labels.size(0)

        train_loss_total += train_loss.item()
        accuracy = 100.0 * correct_predictions / total_predictions

        training_metrics_per_batch = {'training accuracy (per batch)': accuracy,
                                      'training loss (per batch)': train_loss.item()}
        wandb.log(training_metrics_per_batch)

        ## Log trainign progress at every LOG_INTERVAL
        if batch_idx % params["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}'.format(epoch,
                                        batch_idx*len(images),
                                        len(params["train_data_loader"].dataset),
                                        100. * batch_idx / len(params["train_data_loader"]),
                                        train_loss.item()))


    ## calculate and log TRAINING performance metrics (PER EPOCH)
    correct_predictions_total = (np.array(predicted_labels_epoch) == np.array(target_labels_epoch)).sum()

    confusion_matrix =  wandb.Image(utils.plot_confusion_matrix(target_labels_epoch,
                                                                predicted_labels_epoch,
                                                                classes=params["classes"]))

    train_loss_total /= len(params["train_data_loader"].dataset)
    train_accuracy_epoch = 100.0 * correct_predictions_total / len(params["train_data_loader"].dataset)
    training_metrics_per_epoch = {"training loss (per epoch)": train_loss_total,
                                  "training accuracy (per epoch)": train_accuracy_epoch,
                                  "training confusion matrix (per epoch)": confusion_matrix}
    wandb.log(training_metrics_per_epoch)

    """
    =================================== save the model after every epoch ===============================================
    """
    model_state = {
        'epoch': epoch,
        'state_dict': params["net"].state_dict(),
        'optimizer': params["optimizer"].state_dict()
    }

    model_dir = "checkpoints/{}".format(params["model_name"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_outfile = os.path.join(model_dir, "{}_{}.pth".format(params["model_name"], epoch))
    torch.save(model_state, model_outfile)
    """
    ____________________________________________________________________________________________________________________
    """


def test(params, epoch):
    """
    evaluate the network

    :param params (dict): dictionary holding training parameters
    :param epoch (int): epoch counter

    :return:
    """
    ## set mode to evaluation
    params["net"].eval()
    test_loss_total = 0
    example_images = []

    target_labels_epoch = []
    predicted_labels_epoch = []
    class_accuracy = {}

    correct_predictions_total_sum = 0.0

    with torch.no_grad():
        for _, sample in enumerate(params["val_data_loader"]):
            images = sample["image"].to(params["device"])
            target_labels = sample["label"].to(params["device"])
            target_labels_epoch.extend(target_labels)

            output_probs = params["net"](images)
            test_loss = params["loss_fn"](output_probs, target_labels)

            ## calculate the TEST performance metrics (PER BATCH)
            _, predicted_labels = torch.max(output_probs.data, 1)
            predicted_labels_epoch.extend(predicted_labels)

            test_loss_total += test_loss.item()

            correct_predictions_total_sum += (np.array(predicted_labels) == np.array(target_labels)).sum()

            # example_images.append(wandb.Image(images[0]))
                                  # , caption="Pred: {} Truth: {}".format(output_label[0].item(),
                                  #                                                           target_label[0])))

    ## calculate and log TEST performance metrics (PER EPOCH)
    correct_predictions_total = (np.array(predicted_labels_epoch) == np.array(target_labels_epoch)).sum()
    # for class_label in params["classes"].values():
    #     class_name = [name for name, label in params["classes"].items() if label == class_label]
    #     class_predictions = predicted_labels_epoch.count(int(class_label))
    #     class_total_gt = target_labels_epoch.count(int(class_label))
    #     class_correct_predictions = ((np.array(predicted_labels_epoch) == np.array(target_labels_epoch)).astype(int) +
    #                                   (np.array(target_labels_epoch) == int(class_label)).astype(int) == 2).sum()
    #     class_accuracy["test accuracy of %5s" %class_name] = \
    #         100 * class_correct_predictions / (class_predictions + class_total_gt)

    confusion_matrix =  wandb.Image(utils.plot_confusion_matrix(target_labels_epoch,
                                                                predicted_labels_epoch,
                                                                classes=params["classes"]))

    test_loss_total /= len(params["val_data_loader"].dataset)
    test_accuracy_epoch = 100.0 * correct_predictions_total / len(params["val_data_loader"].dataset)
    test_metrics_per_epoch = {"test loss (per epoch)": test_loss_total,
                              "test accuracy (per epoch)": test_accuracy_epoch,
                              "test confusion matrix (per epoch)": confusion_matrix}
    wandb.log(test_metrics_per_epoch)

    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss_total,
                                               correct_predictions_total,
                                               len(params["val_data_loader"].dataset),
                                               test_accuracy_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trigger training and evaluation")

    parser.add_argument("--mode", type=str, default="train", help="train / test")
    parser.add_argument("--train", type=str, default="", help="path to training file")
    parser.add_argument("--validation", type=str, default="", help="path to validation file")
    parser.add_argument("--test", type=str, default="", help="path to test file for evaluation")
    parser.add_argument("--data", type=str, default="", help="path to data in training/validation/test file")
    parser.add_argument("--model_name", type=str, required=True, help="model versioning")

    cmd_args = parser.parse_args()

    main(cmd_args)
