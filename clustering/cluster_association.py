# -*- coding: utf-8 -*-
# @Time    : 23.03.21 14:42
# @Author  : sing_sd

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from rdp import rdp
import pickle
import src.clustering.COStransforms as ct
from frechetdist import frdist

ref = {'lon': 12.0, 'lat': 54.35, 'ECEF': ct.WGS84toECEF(12.0, 54.35)}
colour_array = ["r", "g", "b", "y", "c", "m", "#9475FC", "k"]  # an extra k
# pylab.ion()
# pylab.clf()

########## A Graph associatiation method by frdist distance
def assign_to_graph(data1):
    ENUcoord, nodes, edges = get_data_nodes_edges(data1)
    data = ENUcoord.tolist()
    assignments = np.zeros(shape=(len(data), 1), dtype=int)
    idx_start = 0
    EPSILON = 1000
    idx_end = idx_start + data1.shape[0]
    ENUcoord_mmsi = ENUcoord[idx_start:idx_end, :]
    ENUcoord_subset = rdp(ENUcoord_mmsi, EPSILON)
    rdpNodes = np.array(ENUcoord_subset)
    cost_matrix = np.zeros((len(rdpNodes) - 1, len(edges)))
    start = idx_start
    for i in range(1, len(rdpNodes)):
        ends = np.where((ENUcoord_mmsi == rdpNodes[i, :]).all(axis=1))[0]
        end = ends[np.argmax(ends > start)] + idx_start
        for e_idx, e in enumerate(edges):
            cost_matrix[i-1, e_idx] = frdist([rdpNodes[i-1, :], rdpNodes[i, :]], [nodes[e[0],:], nodes[e[1],:]])

        e_idx = np.argmin(cost_matrix[i-1, :])  # edge id
        if min(cost_matrix[i-1, :]) > 25000:
            e_idx = len(edges)  # ensures color as black
        assignments[start:end, 0] = e_idx
        start = end
    assignments[end, 0] = assignments[end-1, 0]

    return assignments #np.array(assignments[:, 0]).reshape(len(assignments),1)


def get_data_nodes_edges(data1):
    clm_idx = 0
    data1 = np.array(data1)
    nData1 = data1.shape[0]
    lon = np.array(data1[:, clm_idx], dtype=float).reshape([1, nData1])
    lat = np.array(data1[:, clm_idx + 1], dtype=float).reshape([1, nData1])

    ENUcoord = ct.WGS84toENU(lon, lat, ref)
    ENUcoord = np.transpose(ENUcoord)
    ENUcoord = np.delete(ENUcoord, np.s_[2], axis=1)

    with open("../resources/graph_nodes_refined.pkl", 'rb') as f:
        nodesWGS = pickle.load(f)
    nodes = ct.WGS84toENU(nodesWGS[:, 0].T, nodesWGS[:, 1].T, ref)
    nodes = np.transpose(nodes)
    nodes = np.delete(nodes, np.s_[2], axis=1)

    with open("../resources/graph_edges_refined.pkl", 'rb') as f:
        edges = pickle.load(f)
        # i. e., now same as before edges = [[1, 2], [0, 1], [0, 3], [0, 4], [1, 4], [4, 6], [0, 5]]
    return ENUcoord, nodes, edges


def point2edge(point, edge_start, edge_end):
    line_vec = np.subtract(edge_end, edge_start)
    pnt_vec = np.subtract(point, edge_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / np.linalg.norm(line_vec)
    pnt_vec_scaled = np.multiply(pnt_vec, 1.0 / line_len)
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = np.multiply(line_vec, t)
    distance = np.linalg.norm(np.subtract(nearest, pnt_vec))
    return distance


def cosine_angle(a, b, c): # angle at b
    return np.dot(a-b, c-b) / (np.linalg.norm(a-b) * np.linalg.norm(c-b))


def get_slope_edges(edges, nodes):
    n_edges = len(edges)
    slope_edges = [0.0]*(n_edges+1) # +1 for outlier edge
    for e in range(n_edges):
        slope_edges[e] = (nodes[edges[e][0]][1] - nodes[edges[e][1]][1]) / (nodes[edges[e][0]][0] - nodes[edges[e][1]][0])
    return np.array(slope_edges)

def get_assignment(data1):
    DIST_THR = 7000
    ENUcoord, nodes, edges = get_data_nodes_edges(data1)
    data = ENUcoord.tolist()
    cost_matrix = np.zeros((len(data), len(edges)+1))
    cost_matrix[:, len(edges)] = DIST_THR + 1
    assignments = []

    for ii, points in enumerate(data):
        for jj, items in enumerate(edges):
            start = nodes[edges[jj][0]][0:2]
            end = nodes[edges[jj][1]][0:2]
            cost_matrix[ii, jj] = point2edge(points, start, end)
        e_idx = np.argmin(cost_matrix[ii, :])  # edge id
        if min(cost_matrix[ii, :]) > DIST_THR: # 5000:
            e_idx = len(edges)  # ensures color as black
        assignments.append((ii, e_idx))

    assignments = np.array(assignments)
    EPSILON = 500 #1000

    if data1.shape[1] == 2:
        idx_start = 0
        idx_end =  data1.shape[0]
        ENUcoord_mmsi = ENUcoord[idx_start:idx_end, :]
        ENUcoord_subset = rdp(ENUcoord_mmsi, EPSILON)
        rdpNodes = np.array(ENUcoord_subset)
        start = idx_start
        slope_edges = get_slope_edges(edges, nodes)
        for i in range(1, len(rdpNodes)):
            ends = np.where((ENUcoord_mmsi == rdpNodes[i, :]).all(axis=1))[0]
            end = ends[np.argmax(ends > start)] + idx_start
            unique_labels = np.unique(assignments[start:end, 1]) # numpy unique does not preserve order
            max_num = np.inf
            slope_rdpnodes = (rdpNodes[i, 1] - rdpNodes[i - 1, 1]) / (rdpNodes[i, 0] - rdpNodes[i - 1, 0])
            label = unique_labels[np.argmin(np.abs(slope_rdpnodes - slope_edges[unique_labels]))]  # unique_labels[0]
            for j in unique_labels:
                if max(cost_matrix[start,j], cost_matrix[end, j]) <  DIST_THR:
                    if abs(slope_rdpnodes - slope_edges[j]) < max_num:
                    # and not (-0.8 < slope_edges[j]*slope_rdpnodes < -1.2):
                        label = j
                        max_num = abs(slope_rdpnodes - slope_edges[j])
            assignments[start:end, 1] = label
            start = end
        assignments[end, 1] = assignments[end - 1, 1]  # assign the cluster number to last point

    else:
        print("need to be updated like single vessel data code in Else loop")
        exit(0)
        angle_thr = 0.2
        # mmsi_index = data1.columns.get_loc("mmsi")
        for mmsi in data1.mmsi.unique():
            idx_start = np.argmax(data1["mmsi"] == mmsi)
            idx_end = idx_start + np.sum(data1["mmsi"] == mmsi)
            ENUcoord_mmsi = ENUcoord[idx_start:idx_end, :]
            ENUcoord_subset = rdp(ENUcoord_mmsi, EPSILON)
            rdpNodes = np.array(ENUcoord_subset)

            start = idx_start
            for i in range(1, len(rdpNodes)):
                ends = np.where((ENUcoord_mmsi == rdpNodes[i, :]).all(axis=1))[0]
                end = ends[np.argmax(ends>start)] + idx_start
                unique_labels = np.unique(assignments[start:end, 1])
                label = unique_labels[0]
                max_num = 0
                for j in unique_labels:
                    if max_num < sum(assignments[start:end, 1] == j):
                        label = j
                        max_num = sum(assignments[start:end, 1] == j)

                # if label != len(edges) and \
                #         (-angle_thr < cosine_angle(rdpNodes[i-1, :], nodes[edges[label][0]][0:2],
                #                                   rdpNodes[i, :]) < angle_thr and \
                #         -angle_thr < cosine_angle(rdpNodes[i-1, :], nodes[edges[label][1]][0:2],
                #                                   rdpNodes[i, :]) < angle_thr):
                #     label = len(edges)

                assignments[start:end, 1] = label
                start = end
            assignments[end, 1] = assignments[end - 1, 1]

    print('data associated')
    return assignments[:, 1] #np.array(assignments[:, 1]).reshape(len(assignments),1)