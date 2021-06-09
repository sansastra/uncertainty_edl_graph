import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from matplotlib import pyplot as plt

from losses import relu_evidence
from helpers import rotate_img, one_hot_embedding, get_device

def plot_train_metrics(metrics, K):
    num_epochs = 2*(metrics[0]["epoch"][-1] + 1) # for training and validation
    train_acc1 = metrics[1]["accuracy"][0:num_epochs:2]
    train_ev_s = metrics[2]["evidence_succ"][0:num_epochs:2]
    train_ev_f = metrics[3]["evidence_fail"][0:num_epochs:2]
    test_acc1 = metrics[1]["accuracy"][1:num_epochs:2]
    test_ev_s = metrics[2]["evidence_succ"][1:num_epochs:2]
    test_ev_f = metrics[3]["evidence_fail"][1:num_epochs:2]
    # calculate uncertainty for training and testing data for correctly and misclassified samples
    train_u_succ = K / (K + np.array(train_ev_s))
    train_u_fail = K / (K + np.array(train_ev_f))
    test_u_succ = K / (K + np.array(test_ev_s))
    test_u_fail = K / (K + np.array(test_ev_f))

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches([6, 4])

    axs[0].plot(train_u_succ, c='r', marker='+')
    axs[0].plot(train_u_fail, c='k', marker='x')
    axs[0].plot(train_acc1, c='blue', marker='*')
    axs[0].set_title('Train Data')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Estimated uncertainty and accuracy')
    axs[0].legend(['Correct clasifications', 'Misclasifications', 'Accuracy'])
    axs[0].set_ybound(0, 1)

    axs[1].plot(test_u_succ, c='r', marker='+')
    axs[1].plot(test_u_fail, c='k', marker='x')
    axs[1].plot(test_acc1, c='blue', marker='*')
    axs[1].set_title('Val Data')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Estimated uncertainty and accuracy')
    axs[1].legend(['Correct clasifications', 'Misclasifications', 'Accuracy'])
    axs[1].set_ybound(0, 1)

    # axs[1, 0].plot(train_ev_s, c='r', marker='+')
    # axs[1, 0].plot(train_ev_f, c='k', marker='x')
    # axs[1, 0].set_title('Train Data')
    # axs[1, 0].set_xlabel('Epoch')
    # axs[1, 0].set_ylabel('Estimated total evidence for classification')
    # axs[1, 0].legend(['Correct Clasifications', 'Misclasifications'])
    #
    # axs[1, 1].plot(test_ev_s, c='r', marker='+')
    # axs[1, 1].plot(test_ev_f, c='k', marker='x')
    # axs[1, 1].set_title('Test Data')
    # axs[1, 1].set_xlabel('Epoch')
    # axs[1, 1].set_ylabel('Estimated total evidence for classification')
    # axs[1, 1].legend(['Correct Clasifications', 'Misclasifications'])



    fig.tight_layout()
    plt.savefig("./results/oos_all.png")
    plt.savefig("./results/oos_all.pdf")
    plt.show()

def plot_accuracy_uncertainty(pred_prob, unc, y_data, experiment):
    x = np.linspace(0.1, 1, 10)
    acc = np.zeros_like(x)
    acc_ano = np.zeros_like(x)
    match = torch.eq(pred_prob.argmax(axis=1), y_data.argmax(axis=1)).float()
    match_ano = torch.eq(pred_prob[y_data[:, 1] == 1, :].argmax(axis=1), y_data[y_data[:, 1] == 1, :].argmax(axis=1)).float()
    for i in range(10):
        unc_thr = x[i]
        #match[unc[:, 0] > unc_thr] = np.nan
        acc[i] = torch.sum(match[unc[:, 0] <= unc_thr])/len(y_data) # mean over all samples
        acc_ano[i] = torch.sum(match_ano[unc[y_data[:, 1] == 1, 0] <= unc_thr])/torch.sum(y_data[:, 1] == 1)
        if not acc[i] >= 0: # this ensures acc[i] is not nan due to all unc > unc_thr
            acc[i] = 0
        if not acc_ano[i] >= 0:
            acc_ano[i] = 0

    torch.save(torch.tensor(acc), "./results/"+experiment+"_accuracy_uncertainty1.pt")
    torch.save(torch.tensor(acc_ano), "./results/" + experiment + "_accuracy_uncertainty_ano_examples1.pt")
    print("Saved: ./results/model_accuracy_uncertainty.pt")

    fig = plt.figure(figsize=[6.2, 5])
    plt.plot(x, acc, "r-+")
    plt.plot(x, acc_ano, "b-*")
    plt.xlabel("Uncertainty Threshold")
    plt.ylabel("Accuracy")
    plt.legend("All samples", "Anomaly samples")
    plt.xlim(0.1, 1.0)
    plt.ylim(np.min(acc)-0.01, 1.00001)
    ax = plt.gca()
    ax.grid(True)
    # plt.savefig("./results/turn30_acc_unc.png")
    # plt.savefig("./results/turn30_acc_unc.pdf")
    plt.show()




def test_data(experiment, model, metrics, x_test, y_test, uncertainty=False, device=None):
    if not device:
        device = get_device()
    num_classes = y_test.shape[1]
    data_x = Variable(torch.from_numpy(np.array(x_test)).float())
    data_y = Variable(torch.from_numpy(np.array(y_test)).float())
    print("Testing...")


    if uncertainty:
        # plot_train_metrics(metrics, num_classes)
        output = model(data_x)
        evidence = relu_evidence(output)
        alpha = evidence + 1
        unc = torch.divide(num_classes*torch.ones_like(alpha[:,0].reshape((len(evidence), 1))), torch.sum(alpha, dim=1, keepdim=True))
        _, preds = torch.max(output, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        match = torch.reshape(torch.eq(
            preds, data_y[:, 1]).float(), (-1, 1))
        acc = torch.mean(match)
        print("overall accuracy is ", acc)
        plot_accuracy_uncertainty(prob, unc, data_y, experiment)
        # output = output.flatten()
        # prob = prob.flatten()
        # preds = preds.flatten()


    else:

        output = model(data_x)
        _, preds = torch.max(output, 1)
        prob = F.softmax(output, dim=1)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)


    fig = plt.figure(figsize=[6.2, 5])
    plt.plot(unc.detach().numpy(), "k.")
    plt.xlabel("Validation samples")
    plt.ylabel("Uncertainty")
    fig.tight_layout()
    plt.savefig("./results/turn30_uncertaint.png")



