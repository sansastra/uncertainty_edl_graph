import torch
import torch.nn as nn
import numpy as np
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from torch.autograd import Variable


def statistics(epoch, phase, losses, accuracy, evidences_succ, evidences_fail, epoch_loss, epoch_acc,
               evidence_succ, evidence_fail):
    losses["loss"].append(epoch_loss)
    losses["phase"].append(phase)
    losses["epoch"].append(epoch)

    accuracy["accuracy"].append(epoch_acc)
    accuracy["epoch"].append(epoch)
    accuracy["phase"].append(phase)

    evidences_succ["evidence_succ"].append(evidence_succ)
    evidences_succ["phase"].append(phase)
    evidences_succ["epoch"].append(epoch)

    evidences_fail["evidence_fail"].append(evidence_fail)
    evidences_fail["phase"].append(phase)
    evidences_fail["epoch"].append(epoch)

def train_model(model, x_train, x_test, y_train, y_test, num_classes, criterion, optimizer, scheduler=None,
                num_epochs=25, device=None, uncertainty=False):

    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}
    evidences_succ = {"evidence_succ":[],"phase":[], "epoch":[]}
    evidences_fail = {"evidence_fail":[],"phase":[], "epoch":[]}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                data_x = Variable(torch.from_numpy(np.array(x_train)).float())
                data_y = Variable(torch.from_numpy(np.array(y_train)).float())
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                data_x = Variable(torch.from_numpy(np.array(x_test)).float())
                data_y = Variable(torch.from_numpy(np.array(y_test)).float())
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0
            total_evidence_succ = 0
            total_evidence_fail = 0
            nr_success = 0
            nr_fail = 0
            acc = 0

            # Iterate over data.
            batch_size = 8
            total_run = len(data_x)//batch_size - 1
            for i in range(total_run):
                inputs = data_x[i*batch_size:(i+1)*batch_size,:]
                labels = data_y[i*batch_size:(i+1)*batch_size,:]
                #
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        # y = one_hot_embedding(labels, num_classes)
                        y = labels
                        # y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device)

                        match = torch.reshape(torch.eq(
                            preds, labels[:, 1]).float(), (-1, 1))
                        acc += torch.mean(match)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        total_evidence_succ += torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match)
                        nr_success += torch.sum(match ) # + 1e-20
                        total_evidence_fail += torch.sum(torch.sum(evidence, 1, keepdim=True) * (1 - match))
                        nr_fail +=  torch.sum(torch.abs(1 - match)) ## + 1e-20

                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels[:,1].data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            epoch_loss = running_loss / total_run
            epoch_acc = acc / total_run

            statistics(epoch, phase, losses, accuracy, evidences_succ, evidences_fail, epoch_loss, epoch_acc,
                       total_evidence_succ/nr_success, total_evidence_fail/nr_fail)

            print("{} loss: {:.4f} acc: {:.4f}".format(
                phase.capitalize(), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy, evidences_succ, evidences_fail)

    return model, metrics
