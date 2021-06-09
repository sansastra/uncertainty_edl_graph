# -*- coding: utf-8 -*-
# @Time    : 20.05.21 09:29
# @Author  : sing_sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
def main():
    plot_prediction_rg_left_right_center()
    # plot_uncertainties_rg_left_right_center()


def plot_prediction_rg_left_right_center():
    fig1, axs = plt.subplots(1, sharex=True, figsize=(8, 6))
    data_labels =["Ground truth, left", "Ground truth, normal", "Ground truth, right"]
    pred_labels = ["Prediction, left", "Prediction, normal", "Prediction, right"]
    orig_file = ["ano_left", "data", "ano_right"]
    data_names =["left", "normal", "right"]
    markers = ["*", "+", ">"]
    colors = ["r","g","b"]
    data_linestyles = ["dotted", "-", "--"]
    dataset = "1"
    cluster = True
    for i in range(3):
        with open("./resources/gedsar_"+orig_file[i]+".csv", "r") as f:
            data = pd.read_csv(f, sep=',', header=None, names = ["x", "y", "cog","sog"])
            data = data.iloc[:, 0:2]
        with open("./resources/pred_rg_data_"+data_names[i]+"_graph_"+str(cluster)+"_ds"+dataset+".csv", "r") as f:
            pred = pd.read_csv(f, sep=',', header=None, names = ["x", "y", "cog","sog", "cluster"])
            pred= pred.iloc[:, 0:2]



        axs.plot(data.iloc[:, 0], data.iloc[:, 1], linestyle= data_linestyles[i], marker=".", c=colors[i], label=data_labels[i])
        axs.plot(pred.iloc[:, 0], pred.iloc[:, 1],  c=colors[i], marker=markers[i], label=pred_labels[i])
        plt.pause(0.001)


    axs.set_ylabel('Latitude [deg]')
    axs.set_xlabel('Longitude [deg]')
    plt.legend()
    # axs.set_title("Trajectories predicted with a graph-based model")
    #axs.legend(loc="upper right")
    plt.pause(0.001)

    plt.savefig("./results/Pred_rg_all_graph_"+ str(cluster)+"_ds_"+ dataset +".png")
    plt.savefig("./results/Pred_rg_all_graph_"+ str(cluster)+"_ds_"+ dataset +".pdf")
    plt.show()




def plot_uncertainties_rg_left_right_center():
    fig1, axs = plt.subplots(4, sharex=True, figsize=(8, 6))
    data_names =["left", "normal", "right"]
    markers = ["*", "+", "o"]
    colors = ["r", "g", "b"]
    dataset = "1"
    cluster = True
    for i in range(3):
        with open("./resources/unc_rg_data_"+data_names[i]+"_graph_"+str(cluster)+"_ds"+dataset+".csv", "r") as f:
            unc = pd.read_csv(f)
            axs[0].plot(unc.iloc[:, 0], linestyle="-", c=colors[i],marker=markers[i], label=data_names[i])
            axs[1].plot(unc.iloc[:, 1], linestyle="-", c=colors[i],marker=markers[i], label=data_names[i])
            axs[2].plot(unc.iloc[:, 2], linestyle="-", c=colors[i],marker=markers[i], label=data_names[i])
            axs[3].plot(unc.iloc[:, 3], linestyle="-", c=colors[i],marker=markers[i], label=data_names[i])

    axs[0].set_ylabel('Unc_Lon')
    axs[1].set_ylabel('Unc_Lat')
    axs[2].set_ylabel('Unc_COG')
    axs[3].set_ylabel('Unc_SOG')
    plt.xlabel('Time index')
    axs[0].set_title("Uncertainty obtained without a graph-based model")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper right")
    axs[3].legend(loc="upper right")
    plt.pause(0.001)

    # plt.savefig("./results/Unc_rg_all_graph_"+ str(cluster)+"_ds_"+ dataset +".png")
    # plt.savefig("./results/Unc_rg_all_graph_"+ str(cluster)+"_ds_"+ dataset +".pdf")
    plt.show()

if __name__ == "__main__":
    main()

