
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import src.common_functions as cf
import src.clustering.COStransforms as ct

import src.clustering.cluster_association as ca
plt.rcParams.update({'font.size': 12})

try:
    with open("../resources/ais_data_rostock_2016_processed.csv", 'rb') as f:
        data = pd.read_csv(f)
    print("AIS data import succeeded")

    with open("./results/graph_nodes_new.pkl", 'rb') as f:
        nodesENU = pickle.load(f)
    print("Graph nodes import succeeded")

    with open("./results/graph_edges_new.pkl", 'rb') as f:
        edges = pickle.load(f)
    print("Graph edges import succeeded")

except (ModuleNotFoundError, ImportError) as e:
    print("{} failure".format(type(e)))


data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                     'nav_status': 'float', 'ship_type': 'float', 'destination': 'str'})
# print(len(data))

### Select the date range
nr_of_vessels = -1 #200
WINDOW = [11.5, 54.2, 12.5, 54.5] #[11.5, 54.2, 12.5, 54.5]
SOG_LIMIT = [0, 30]
NAV_STATUS = 0.0
SHIP_TYPES = [60, 90]
idx = cf.get_filtered_data_idx(data, WINDOW, SOG_LIMIT, NAV_STATUS, SHIP_TYPES)
data = data.iloc[idx, :]

print("Starting, processed data=", len(data))

ref = {'lon': 12.0, 'lat': 54.35, 'ECEF': ct.WGS84toECEF(12.0, 54.35)}

nodesENU = np.array(nodesENU)
nodesENU[:, 2] = 0.0
nodesENU = nodesENU.astype(np.float)
nodes = ct.ENUtoWGS84(tENU= nodesENU.T, tREF=ref)
nodes[len(nodes)-1, :] = 0.0 # useful for refining nodes later
nodes = np.transpose(nodes)

edges = np.array(edges, dtype=np.int)
tkeys = data.mmsi.unique()


def main():
    # plot_graph_with_data() # plot original graph created by data in graph_cluster.py

    # plot_nodes() # plot the numbered nodes in original graph
    nodes_tobe_deleted = [6]
    nodes_new = np.delete(nodes, nodes_tobe_deleted, axis=0)
    # with open("./results/graph_nodes_refined.pkl", "wb") as fp:  # Pickling
    #    pickle.dump(nodes_new, fp)
    # plot_refined_nodes(nodes_new) # plot refined nodes to fined which edges are relevant
    edges_new = [[1, 2], [0, 1], [0, 3], [0, 4], [1, 4], [4, 6], [0, 5]] #
    # with open("./results/graph_edges_refined.pkl", "wb") as fp:  # Pickling
    #     pickle.dump(edges_new, fp)
    # plot_refined_graph(nodes_new, edges_new) # finally plot the refined graph
    plot_refined_graph_association(nodes_new, edges_new)


def plot_graph_with_data(): # in WGS84 (in degree):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([7, 5])
    for mmsi in tkeys[:nr_of_vessels]:  # [vessel_nr:vessel_nr+1]: #
        decoded_mmsi = data[data['mmsi'] == mmsi]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        axs.plot(np.array(decoded_mmsi["x"]), np.array(decoded_mmsi["y"]), c="blue", linestyle= "dotted", alpha=0.5, linewidth=0.2)
    for ee, e in enumerate(edges):
        axs.plot([nodes[e[0], 0], nodes[e[1], 0]], [nodes[e[0],1], nodes[e[1], 1]], linestyle="-", color="black", linewidth=2)
        axs.plot([nodes[e[0], 0], nodes[e[1], 0]], [nodes[e[0],1], nodes[e[1], 1]], marker=".", color="black", markersize=15)
        plt.pause(0.0001)

    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    axs.set_xlim(WINDOW[0], WINDOW[2])
    axs.set_ylim(WINDOW[1], WINDOW[3])
    plt.pause(0.001)
    plt.savefig("./results/graph_original.png")
    plt.savefig("./results/graph_original.pdf")
    plt.show()


def plot_nodes():
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([7, 5])
    axs.scatter(nodes[:, 0], nodes[:, 1])
    for i, n in enumerate(nodes):
        axs.annotate(i, (n[0], n[1]))
        plt.pause(0.0001)

    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    plt.pause(0.001)
    plt.savefig("./results/numbered_nodes_original.png")
    # plt.savefig("./results/numbered_nodes_original.pdf")
    plt.show()


def plot_refined_nodes(nodes_new):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([7, 5])
    axs.scatter(nodes_new[:, 0], nodes_new[:, 1])
    for i, n in enumerate(nodes_new):
        axs.annotate(i, (n[0], n[1]))
        plt.pause(0.0001)
    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    plt.pause(0.001)
    plt.savefig("./results/numbered_nodes_refined.png")
    # plt.savefig("./results/numbered_nodes_original.pdf")
    plt.show()


def plot_refined_graph(nodes_new, edges_new):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([7, 5])
    for mmsi in tkeys[:nr_of_vessels]:  # [vessel_nr:vessel_nr+1]: #
        decoded_mmsi = data[data['mmsi'] == mmsi]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        axs.plot(np.array(decoded_mmsi["x"]), np.array(decoded_mmsi["y"]),
                 marker =".", c="blue", markersize=0.001, alpha=0.5)
    for ee, e in enumerate(edges_new):
        axs.plot([nodes_new[e[0], 0], nodes_new[e[1], 0]], [nodes_new[e[0], 1], nodes_new[e[1], 1]], linestyle="-", color="black",
                 linewidth=2)
        axs.plot([nodes_new[e[0], 0], nodes_new[e[1], 0]], [nodes_new[e[0], 1], nodes_new[e[1], 1]], marker=".", color="black",
                 markersize=15)
        plt.pause(0.0001)
    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    plt.pause(0.001)
    plt.savefig("./results/graph_refined.png")
    plt.savefig("./results/graph_refined.pdf")
    plt.show()


def plot_refined_graph_association(nodes_new, edges_new):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([8, 6])
    plt.pause(0.0001)
    colour_array = ["r", "g", "b", "y", "c", "m", "#9475FC", "grey"]  # an extra k
    for mmsi in tkeys[:nr_of_vessels]:  # [vessel_nr:vessel_nr+1]: #
        decoded_mmsi = data[data['mmsi'] == mmsi]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        # assignment = ca.assign_to_graph(decoded_mmsi[["x", "y"]])#
        assignment = ca.get_assignment(decoded_mmsi[["x", "y"]])
        for cluster in np.unique(assignment):
            idx = assignment == cluster
            axs.scatter(np.array(decoded_mmsi.iloc[idx, 0]), np.array(decoded_mmsi.iloc[idx, 1]), c=colour_array[cluster],
                     marker=".", s=0.01, alpha=1) # s=0.05
            plt.pause(0.00001)
    for ee, e in enumerate(edges_new):
        axs.plot([nodes_new[e[0], 0], nodes_new[e[1], 0]], [nodes_new[e[0], 1], nodes_new[e[1], 1]], linestyle="-",
                 color="black", linewidth=5) # colour_array[ee]
        axs.plot([nodes_new[e[0], 0], nodes_new[e[1], 0]], [nodes_new[e[0], 1], nodes_new[e[1], 1]],
                    marker=".", color="black", markersize=30)
        plt.pause(0.0001)
    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    axs.set_xlim(WINDOW[0], WINDOW[2])
    axs.set_ylim(WINDOW[1], WINDOW[3])
    plt.pause(0.001)
    plt.savefig("./results/graph_association_60_90.png")
    plt.savefig("./results/graph_association_60_90.pdf")
    plt.show()



if __name__ == "__main__":
    main()