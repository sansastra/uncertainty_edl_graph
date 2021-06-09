# -*- coding: utf-8 -*-
# @Time    : 14.04.21 15:36
# @Author  : sing_sd
import torch
import numpy as np
from matplotlib import pyplot as plt

acc_oos = torch.load("./results/oos_accuracy_uncertainty1.pt").detach().cpu().numpy()
acc_turn30 = torch.load("./results/turn_30_accuracy_uncertainty1.pt").detach().cpu().numpy()
acc_oos_ano = torch.load("./results/oos_accuracy_uncertainty_ano_examples1.pt").detach().cpu().numpy()
acc_turn30_ano = torch.load("./results/turn_30_accuracy_uncertainty_ano_examples1.pt").detach().cpu().numpy()
x = np.linspace(0.1, 1, 10)
y_lower = 0.0 # min(min(acc_oos),min(acc_turn30),min(acc_oos_ano), min(acc_turn30_ano))-0.01

plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(1, 1)
fig.set_size_inches([6, 6])

axs.plot(x, acc_oos, c='b',linestyle='-', marker='>')
axs.plot(x, acc_oos_ano, c='b',linestyle='-.', marker='*')
axs.plot(x, acc_turn30, c='k',linestyle='-', marker='o')
axs.plot(x, acc_turn30_ano, c='k',linestyle='-.', marker='+')
axs.set_xlabel('Uncertainty Threshold')
axs.set_ylabel('Accuracy')
axs.legend(['OOS, All Samples','OOS, Anomaly Samples', 'UT, All Samples', 'UT, Anomaly Samples'])
axs.set_xbound(0.1, 1.0)
axs.set_ybound(y_lower, 1.001)
axs.grid(True)
plt.pause(0.001)
plt.savefig("./results/acc_unc_all.png")
plt.savefig("./results/acc_unc_all.pdf")
plt.show()

exit(0)
##############################################################
fig, axs = plt.subplots(1, 2)
fig.set_size_inches([8, 3])

axs[0].plot(x, acc_oos, c='r', marker='+')
axs[0].plot(x, acc_turn30, c='blue', marker='*')
axs[0].set_title("All data samples")
axs[0].set_xlabel('Uncertainty Threshold')
axs[0].set_ylabel('Accuracy')
axs[0].legend(['AIS OOS', 'Unusual Turn'])
axs[0].set_xbound(0.1, 1.0)
axs[0].set_ybound(y_lower, 1.001)
axs[0].grid(True)

axs[1].plot(x, acc_oos_ano, c='r', marker='+')
axs[1].plot(x, acc_turn30_ano, c='blue', marker='*')
axs[1].set_title("Anomalous data samples")
axs[1].set_xlabel('Uncertainty Threshold')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['AIS OOS', 'Unusual Turn'])
axs[1].set_xbound(0.1, 1.0)
axs[1].set_ybound(y_lower, 1.001)
axs[1].grid(True)
plt.pause(0.001)
plt.savefig("./results/acc_unc.png")
plt.savefig("./results/acc_unc.pdf")
plt.show()