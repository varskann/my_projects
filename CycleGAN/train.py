"""
train.py: training and evaluation logic goes here
"""

__author__ = "Kanishk Varshney"
__date__ = "Sun Sep 24 22:56:12 IST 2019"

import os
import argparse
import itertools

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torchvision.transforms as transforms
# import torchvision

from PIL import Image
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
    decay_epoch = utils.get_config("training", "decay_epoch", _type=int)
    params["num_classes"] = utils.get_config("classes", "num_classes", _type=int)
    params["log_interval"] = utils.get_config("code", "log_interval", _type=int)
    params["classes"] = utils.get_class_names()

    ## fetch the model
    params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load models
    params["netG_A2B"] = models.Generator().to(params["device"])
    params["netG_B2A"] = models.Generator().to(params["device"])
    params["netD_A"] = models.Descriminator().to(params["device"])
    params["netD_B"] = models.Descriminator().to(params["device"])

    params["netG_A2B"].apply(utils.weights_init_normal)
    params["netG_B2A"].apply(utils.weights_init_normal)
    params["netD_A"].apply(utils.weights_init_normal)
    params["netD_B"].apply(utils.weights_init_normal)

    # wandb.watch(params["net"], log="all")

    # Lossess
    params["criterion_GAN"] = nn.MSELoss()
    params["criterion_cycle"] = nn.L1Loss()
    params["criterion_identity"] = nn.L1Loss()

    # Optimizers & LR schedulers
    params["optimizer_G"] = optim.Adam(itertools.chain(params["netG_A2B"].parameters(),
                                                       params["netG_B2A"].parameters()),
                                       lr=learning_rate, betas=(0.5, 0.999))
    params["optimizer_D_A"] = optim.Adam(params["netD_A"].parameters(), lr=learning_rate, betas=(0.5, 0.999))
    params["optimizer_D_B"] = optim.Adam(params["netD_B"].parameters(), lr=learning_rate, betas=(0.5, 0.999))

    params["lr_scheduler_G"] = optim.lr_scheduler.LambdaLR(params["optimizer_G"],
                                                           lr_lambda=utils.LambdaLR(max_epochs, 0, decay_epoch).step)
    params["lr_scheduler_D_A"] = optim.lr_scheduler.LambdaLR(params["optimizer_D_A"],
                                                             lr_lambda=utils.LambdaLR(max_epochs, 0, decay_epoch).step)
    params["lr_scheduler_D_B"] = optim.lr_scheduler.LambdaLR(params["optimizer_D_B"],
                                                             lr_lambda=utils.LambdaLR(max_epochs, 0, decay_epoch).step)

    # Dataset loader
    transforms_ = [transforms.Resize(int(256 * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(256),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,), (0.5,))]

    ## load data
    train_data = dataset.ImageDataset(args.data,
                                      transforms_=transforms_,
                                      unaligned=True,
                                      mode="train")
    params["train_data_loader"] = data.DataLoader(train_data,
                                                  num_workers=num_workers,
                                                  batch_size=train_batch_size,
                                                  shuffle=True,
                                                  pin_memory=True)

    # ## load data
    # val_data = dataset.ImageDataset(args.data,
    #                                 transforms_=transforms_,
    #                                 unaligned=True,
    #                                 mode="test")
    # params["val_data_loader"] = data.DataLoader(val_data,
    #                                             num_workers=num_workers,
    #                                             batch_size=train_batch_size,
    #                                             shuffle=True,
    #                                             pin_memory=True)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    params["input_A"] = Tensor(train_batch_size, 3, 256, 256)
    params["input_B"] = Tensor(train_batch_size, 3, 256, 256)
    params["target_real"] = Variable(Tensor(train_batch_size).fill_(1.0), requires_grad=False)
    params["target_fake"] = Variable(Tensor(train_batch_size).fill_(0.0), requires_grad=False)

    params["fake_A_buffer"] = utils.ReplayBuffer()
    params["fake_B_buffer"] = utils.ReplayBuffer()


    # params["logger"] = utils.Logger(max_epochs, len(params["train_data_loader"]))
    ## trigger training
    for _epoch in range(max_epochs):
        train(params, _epoch)
        # test(params, _epoch)


def train(params, epoch):
    """
    train the network

    :param params (dict): dictionary holding training parameters
    :param epoch (int): epoch counter

    :return:
    """
    ## set mode to train

    for _, batch in enumerate(params["train_data_loader"]):
        real_A = Variable(params["input_A"].copy_(batch['A']))
        real_B = Variable(params["input_B"].copy_(batch['B']))

        ####### train the generator network iteration ########
        params["optimizer_G"].zero_grad()

        ## Identity loss
        # G_A2B(B) should equal B if real B is fed

        same_B = params["netG_A2B"](real_B)
        loss_identity_B = params["criterion_identity"](same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = params["netG_B2A"](real_A)
        loss_identity_A = params["criterion_identity"](same_A, real_A) * 5.0

        # GAN loss
        fake_B = params["netG_A2B"](real_A)
        pred_fake = params["netD_B"](fake_B)
        loss_GAN_A2B = params["criterion_GAN"](pred_fake, params["target_real"])

        fake_A = params["netG_B2A"](real_B)
        pred_fake = params["netD_A"](fake_A)
        loss_GAN_B2A = params["criterion_GAN"](pred_fake, params["target_real"])

        # Cycle loss
        recovered_A = params["netG_B2A"](fake_B)
        loss_cycle_ABA = params["criterion_cycle"](recovered_A, real_A) * 10.0

        recovered_B = params["netG_A2B"](fake_A)
        loss_cycle_BAB = params["criterion_cycle"](recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        params["optimizer_G"].step()
        #########################################

        ####### train discriminator #######

        ## Discriminator A
        params["optimizer_D_A"].zero_grad()

        pred_real = params["nedD_A"](real_A)
        loss_D_real = params["criterion_GAN"](pred_real, params["target_real"])

        # Fake loss
        fake_A = params["fake_A_buffer"].push_and_pop(fake_A)
        pred_fake = params["netD_A"](fake_A.detach())
        loss_D_fake = params["criterion_GAN"](pred_fake, params["target_fake"])

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        params["optimizer_D_A"].step()

        ## Discriminator B
        params["optimizer_D_B"].zero_grad()

        pred_real = params["nedD_B"](real_B)
        loss_D_real = params["criterion_GAN"](pred_real, params["target_real"])

        # Fake loss
        fake_B = params["fake_B_buffer"].push_and_pop(fake_B)
        pred_fake = params["netD_B"](fake_B.detach())
        loss_D_fake = params["criterion_GAN"](pred_fake, params["target_fake"])

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        params["optimizer_D_B"].step()


        # Progress report (http://localhost:8097)

        wandb.log({'loss_G': loss_G,
                   'loss_G_identity': (loss_identity_A + loss_identity_B),
                   'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                   'loss_D': (loss_D_A + loss_D_B)},
                    images={'real_A': wandb.Image(real_A), 'real_B': wandb.Image(real_B),
                            'fake_A': wandb.Image(fake_A), 'fake_B': wandb.Image(fake_B)})


    """
    =================================== save the model after every epoch ===============================================
    """
    # Update learning rates
    params["lr_scheduler_G"].step()
    params["lr_scheduler_D_A"].step()
    params["lr_scheduler_D_B"].step()

    # Save models checkpoints
    torch.save(params["netG_A2B.state_dict()"], 'checkpoints/netG_A2B.pth')
    torch.save(params["netG_B2A.state_dict()"], 'checkpoints/netG_B2A.pth')
    torch.save(params["netD_A.state_dict()"], 'checkpoints/netD_A.pth')
    torch.save(params["netD_B.state_dict()"], 'checkpoints/netD_B.pth')
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
    parser.add_argument("--model_name", type=str, required=False, help="model versioning")

    cmd_args = parser.parse_args()

    main(cmd_args)
