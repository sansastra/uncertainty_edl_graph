# -*- coding: utf-8 -*-
# @Time    : 26.04.21 11:56
# @Author  : sing_sd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import src.common_functions as cf
import src.clustering.COStransforms as ct

import src.clustering.cluster_association as ca
plt.rcParams.update({'font.size': 12})

def main():
    data = pd.DataFrame(get_ais_data_interpol_1min())
    data.columns = ["x", "y", "cog", "sog", "cluster", "mmsi"]

    with open("../resources/graph_nodes_refined.pkl", 'rb') as f:
        nodes_new = pickle.load(f)
    print("Graph nodes import succeeded")

    with open("../resources/graph_edges_refined.pkl", 'rb') as f:
        edges_new = pickle.load(f)
    print("Graph edges import succeeded")

    plot_refined_graph_association(data, nodes_new, edges_new)






def get_ais_data_interpol_1min():
    nr_of_actual_vessels = -1
    path = Path('/home/sing_sd/Desktop/anomaly_detection/PythonCode/Trajectory_Prediction/')
    filename = path / 'all_sampled_tracks_interpol_1minute.csv'
    with open(filename, 'rb') as f:
        data = pd.read_csv(f)
    data["cluster"] = 0
    data["mmsi"] = 0
    data = np.array(data)

    filename = path / 'data_len_sampled_tracks_interpol_1minute.pkl'
    with open(filename, 'rb') as f:
        data_all_tracks = pickle.load(f)
    overall_data = np.ones_like(data)
    nr_track = 1000000
    index = 0
    data_index = 0
    dim = 4
    for track_nr in range(len(data_all_tracks)):
        nr_data = data_all_tracks[track_nr]
        if 20 < data_all_tracks[track_nr] < 500:
            nr_track += 1
            overall_data[index: index + nr_data, 0:dim] = data[data_index: data_index + nr_data, 0:dim]
            overall_data[index: index + nr_data, dim + 1] = nr_track
            index += nr_data
        data_index += nr_data
    # get rostock-gedsar data # above code does not get rostock-gedsar data
    filename = path / 'rostock_gedsar_interpol_1min.csv'
    with open(filename, 'rb') as f:
        data = np.array(pd.read_csv(f))
    overall_data[index: index + len(data), 0:dim] = data[:, 0:dim]
    overall_data[index: index + len(data), dim + 1] = nr_track + 1
    index += len(data)
    overall_data = np.delete(overall_data, range(index, len(overall_data)), axis=0)
    return overall_data

def plot_refined_graph_association(data, nodes_new, edges_new):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([7, 5])
    plt.pause(0.0001)
    colour_array = ["r", "g", "b", "y", "c", "m", "#9475FC", "k"]  # an extra k
    for mmsi in data.mmsi.unique():  # [vessel_nr:vessel_nr+1]: #
        idx_all = data['mmsi'] == mmsi
        decoded_mmsi = data[data['mmsi'] == mmsi]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        # assignment = ca.assign_to_graph(decoded_mmsi[["x", "y"]])#
        assignment = ca.get_assignment(decoded_mmsi[["x", "y"]])
        for cluster in np.unique(assignment):
            idx = assignment == cluster
            axs.scatter(np.array(decoded_mmsi.iloc[idx, 0]), np.array(decoded_mmsi.iloc[idx, 1]), c=colour_array[cluster],
                     marker=".", s=0.5) # s=0.05
            plt.pause(0.00001)
        data.iloc[idx_all, data.columns.get_loc("cluster")] = assignment #
    data.to_csv("./resources/ais_data_1min_graph.csv", index=False)

    for ee, e in enumerate(edges_new):
        axs.plot([nodes_new[e[0], 0], nodes_new[e[1], 0]], [nodes_new[e[0], 1], nodes_new[e[1], 1]], linestyle="-",
                 color="black", linewidth=4)
        axs.scatter([nodes_new[e[0], 0], nodes_new[e[1], 0]], [nodes_new[e[0], 1], nodes_new[e[1], 1]], marker=".",
                 color="black", s=10)
        plt.pause(0.0001)
    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    plt.pause(0.001)
    plt.savefig("./results/graph_association_dataset2.png")
    plt.savefig("./results/graph_association_dataset2.pdf")
    plt.show()

if __name__ == "__main__":
    main()



