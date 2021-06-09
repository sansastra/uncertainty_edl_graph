# -*- coding: utf-8 -*-
# @Time    : 29.03.21 11:51
# @Author  : sing_sd
import os
import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np
import argparse
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
#from data import dataloaders, digit_one
from train import train_model
from test import test_data
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from net import Net_OOS
from src.classification.load_data_last_seen import load_all_data, load_test_data, load_test_data_new
from sklearn.model_selection import train_test_split

def main():
    features = ['x', 'y', 'cog', 'sog']
    dim = len(features)
    timesteps = 60  # number of sequential features per sample
    CLASSES = 2
    # load data

    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--train", default=False, action="store_true",
                            help="To train the network.")
    mode_group.add_argument("--test", default=True, action="store_true",
                            help="To test the network.")
    parser.add_argument("--epochs", default=5, type=int,
                        help="Desired number of epochs.")
    parser.add_argument("--dropout", default=True, action="store_true",
                        help="Whether to use dropout or not.")
    parser.add_argument("--uncertainty", default = True, action="store_true",
                        help="Use uncertainty or not.")
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument("--mse", default = True, action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.")
    uncertainty_type_group.add_argument("--digamma", default = False, action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.")
    uncertainty_type_group.add_argument("--log", default = False, action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.")
    args = parser.parse_args()



    if args.train:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = CLASSES
        model = Net_OOS(dropout=args.dropout)

        if use_uncertainty:
            if args.digamma:
                criterion = edl_digamma_loss
            elif args.log:
                criterion = edl_log_loss
            elif args.mse:
                criterion = edl_mse_loss
            else:
                parser.error(
                    "--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

        exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)

        data_normal, Y_data = load_all_data(timesteps, dim, features, CLASSES)
        np.random.seed(1234)
        x_train, x_test, y_train, y_test = train_test_split(data_normal, Y_data, test_size=0.10)

        model, metrics = train_model(model, x_train, x_test, y_train, y_test, num_classes, criterion,
                                     optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs,
                                     device=device, uncertainty=use_uncertainty)

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if use_uncertainty:
            if args.digamma:
                torch.save(state, "./results/model_uncertainty_digamma.pt")
                print("Saved: ./results/model_uncertainty_digamma.pt")
            if args.log:
                torch.save(state, "./results/model_uncertainty_log.pt")
                print("Saved: ./results/model_uncertainty_log.pt")
            if args.mse:
                torch.save(state, "./results/oos_model_uncertainty_mse_epochs_"+str(num_epochs)+".pt")
                print("Saved: ./results/oos_model_uncertainty_mse_epochs_"+str(num_epochs)+".pt")
                torch.save(metrics, "./results/oos_metrics_mse_epochs_"+str(num_epochs)+".pt")
                print("Saved: ./results/oos_metrics_mse_epochs_" + str(num_epochs) + ".pt")

        else:
            torch.save(state, "./results/oos_model.pt")
            print("Saved: ./results/oos_model.pt")

    elif args.test:

        use_uncertainty = args.uncertainty
        device = get_device()
        model = Net_OOS()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())

        if use_uncertainty:
            if args.digamma:
                checkpoint = torch.load(
                    "./results/model_uncertainty_digamma.pt")
                filename = "./results/rotate_uncertainty_digamma.jpg"
            if args.log:
                checkpoint = torch.load("./results/model_uncertainty_log.pt")
                filename = "./results/rotate_uncertainty_log.jpg"
            if args.mse:
                checkpoint = torch.load("./results/oos_model_uncertainty_mse_epochs_50.pt")
                metrics = torch.load("./results/oos_metrics_mse_epochs_50.pt")
                print("testing with metrics of 50 epochs")

        else:
            checkpoint = torch.load("./results/model.pt")
            filename = "./results/rotate.jpg"

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.eval()

        x_test, y_test = load_test_data_new(timesteps, dim, features, CLASSES)
        test_data("oos", model, metrics, x_test, y_test, uncertainty=use_uncertainty, device=None)


if __name__ == "__main__":
    main()
