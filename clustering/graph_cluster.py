# -*- coding: utf-8 -*-
# @Time    : 26.02.21 11:23
# @Author  : sing_sd

import COStransforms as ct
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotmean_cont as pm
import random
from rdp import rdp
from pathlib import Path
from sklearn.cluster import DBSCAN
import src.common_functions as cf
import src.clustering.COStransforms as ct
import pickle

try:
    with open("../resources/ais_data_rostock_2016_processed.csv", 'rb') as f:
        data = pd.read_csv(f)
        print("Import succeeded")
except (ModuleNotFoundError, ImportError) as e:
    print("{} fileure".format(type(e)))


data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                    'nav_status': 'float', 'ship_type': 'float', 'destination': 'str'})

### Select the date range
nr_of_vessels = -1
WINDOW = [11.5, 54.2, 12.5, 54.5] # 54.5
SOG_LIMIT = [0, 30]
NAV_STATUS = 0.0
SHIP_TYPES = [60, 90]
idx = cf.get_filtered_data_idx(data, WINDOW, SOG_LIMIT, NAV_STATUS, SHIP_TYPES)
data = data.iloc[idx, :]
print("Starting")

ref = {'lon': 12.0, 'lat': 54.35, 'ECEF': ct.WGS84toECEF(12.0, 54.35)}

# parameters for RDP
reducedNodes = []   # to store all nodes as list/array
rdpNodes = {}   # to store all the reduced nodes with mmsi info
EPSILON = 1000  # 1000 # maximum perpendicular error distance allowed
plt.figure()

# LOAD DATA

tkeys = data.mmsi.unique()
for mmsi in tkeys[:nr_of_vessels]:  # [vessel_nr:vessel_nr+1]: #
    decoded_mmsi = data[data['mmsi'] == mmsi]
    decoded_mmsi = decoded_mmsi.reset_index(drop=True)
    ENUcoord = ct.WGS84toENU(np.array(decoded_mmsi["x"]), np.array(decoded_mmsi["y"]), ref)
    ENUcoord = np.transpose(ENUcoord)
    ENUcoord = np.delete(ENUcoord, np.s_[2], axis=1) # deleting height values [:, 2]

    # Applying Ramer-Douglas-Peucker Algorithm
    ENUcoord_subset = rdp(ENUcoord, EPSILON)
    rdpNodes[mmsi] = np.array(ENUcoord_subset)
    reducedNodes.append(ENUcoord_subset)
    plt.plot(ENUcoord[:, 0], ENUcoord[:, 1], 'k.',  markersize=0.5) # alpha=0.3,
    plt.plot(ENUcoord_subset[:, 0], ENUcoord_subset[:, 1], 'ro',alpha=0.2,  markersize=8) # alpha=0.2,
    plt.pause(0.00001)
# plt.show()


# DBSCAN Algorithm

# plt.figure()
DBnodes = np.concatenate(reducedNodes)
EPSILON = 1500 #1800
MIN_SMP = 20 #20

labels = []

db = DBSCAN(eps=EPSILON, min_samples=MIN_SMP, metric='euclidean').fit(DBnodes)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
plt.title('Estimated number of clusters: %d' % n_clusters_)

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'

    class_member_mask = (labels == k)

    xy = DBnodes[class_member_mask & core_samples_mask]
    # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

print(n_clusters_)
# plt.show()


#  CLUSTER MEANS

mu, std = pm.plotmean(labels, DBnodes)
mu = np.delete(mu, len(mu) - 1, 0)
std = np.delete(std, len(std) - 1, 0) # delete an extra empty row at the end
# plt.show()


# EDGE EXTRACTION AND REDUCTION

thsld = 9.21 # ???Chi square test score
mmsi2nodes = {k: [] for k in rdpNodes.keys()}
tmp_list = {}
# find mmsi (vessels) belonging/passing to nodes
for mmsi, val in rdpNodes.items():
    tmp_list[mmsi] = []
    for idx in range(len(val)):
        res = []
        dot = []
        enu = np.array(val[idx, :2])
        for label in range(len(mu)):
            node_mu = mu[label]
            node_cov = std[label]
            res = np.subtract(enu.reshape((2, 1)), node_mu.reshape((2, 1)))
            dot.append(np.dot(res.T, np.dot(np.linalg.linalg.inv(node_cov), res)))
        if min(dot) < thsld:
            tmp_list[mmsi].append(dot.index(min(dot)))
mmsi2nodes = tmp_list # explain how temp_list is constructed

# Check 2 same nodes consecutive
for mmsi, val in mmsi2nodes.items():
    i = 0
    while i < len(val) - 1:
        if val[i] == val[i + 1]:
            mmsi2nodes[mmsi].remove(val[i])
        else:
            i += 1

# Check if now we have only 1 node (so no edge possible)
for mmsi in list(mmsi2nodes.keys()):
    if len(mmsi2nodes[mmsi]) < 2:
        mmsi2nodes.pop(mmsi)

# Connection matrix (edges) assigns number of visits of mmsis to edges (NxN)
edge = np.zeros([len(mu), len(mu)])
routes = [] * (len(mmsi2nodes.keys()))
for mmsi, val in mmsi2nodes.items():
    for i in range(1, len(val)):
        n1 = val[i - 1]
        n2 = val[i]
        edge[n1, n2] += 1
    #        if n1==5 and n2==1:
    #            print(n1,n2, mmsi)
    routes.append(val)

# Vessel that follows the previous routes
mmsi2routes = {k: [] for k in range(len(np.unique(routes)))}
for mmsi, val in mmsi2nodes.items():
    j = 0
    for route in np.unique(routes):
        if val == route:
            mmsi2routes[j].append(mmsi)
        j += 1

count = np.zeros((len(mu), 1), dtype=int) # counts number of visits of ships for each node
i = 0
for e in edge:
    count[i] = sum(e)
    i += 1

# Categorize Nodes
N = len(edge)
mulist = mu.tolist()
sumh = np.zeros((N, 1))
sumv = np.zeros((N, 1))
maxh = np.zeros((N, 1))
maxv = np.zeros((N, 1))
border = [[-32000, 18000], [-18000, 18000]] #[[-24000, 18000], [-20000, 23000]]
inout_ratio = 1
tot = sum(sum(edge))
for m in range(len(mu)):
    sumh[m] = sum(edge[m, :])
    sumv[m] = sum(edge[:, m])
    maxh[m] = max(edge[m, :])
    maxv[m] = max(edge[:, m])
    dist = np.abs(mu[m] - border)
    if sumv[m] == 0:
        mulist[m].append('entry')
        print(m, 'entry cause 0')
    elif sumh[m] == 0:
        mulist[m].append('exit')
        print(m, 'exit cause 0')
    elif (sumv[m] / sumh[m] > inout_ratio or maxv[m] / maxh[m] > inout_ratio) and (np.any(dist < 2000)): # 2000
        mulist[m].append('exit')
        print(m, 'exit cause div')
    elif (sumh[m] / sumv[m] > inout_ratio or maxh[m] / maxv[m] > inout_ratio) and (np.any(dist < 2000)): #2000
        mulist[m].append('entry')
        print(m, 'entry cause div')
    else:
        mulist[m].append('mid')
        print(m, 'mid cause else')

# Create the final Good Edges
plt.figure()
plt.pause(0.0001)
GEdges = []
h = 0.3
value = np.zeros((N, N))
aux = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            value[i][j] = 0
            aux[i][j] = 0
            if count[i] != 0:
                value[i][j] = edge[i][j] / count[i]
            if count[j] != 0:
                aux[i][j] = edge[i][j] / count[j]
            if value[i][j] > h and (
                    mulist[i][2] != 'exit' or (mulist[i][2] == 'exit' and mulist[j][2] == 'exit') and mulist[j][
                2] != 'entry'):
                GEdges.append([i, j, mulist[i], mulist[j], edge[i][j]])
                print('first edge:', i, j, value[i][j])
            else:
                if 1 > aux[i][j] > h and (
                        mulist[i][2] != 'exit' or (mulist[i][2] == 'exit' and mulist[j][2] == 'exit') and mulist[j][
                    2] != 'entry'):
                    GEdges.append([i, j, mulist[i], mulist[j], edge[j][i]])
                    print('second edge:', i, j, aux[i][j], value[i][j])
                else:
                    print("value_",i,"_",j,"=", value[i][j])
                    print("aux_", i, "_", j, "=", aux[i][j])

assignment_edges = []
start_edges = []
end_edges = []
edges_cov = []
colour_array = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(len(GEdges)+1)] # +1 added for a trajectory point that does not fit any edges, assign default

colour_array[len(GEdges)] = "k" # assign black color to default unfit points to edges during assignment

for ee, e in enumerate(GEdges):
    lw = np.clip(10 * edge[e[0]][e[1]] / max(maxh), 0.1, 10)
    plt.plot([e[2][0], e[3][0]], [e[2][1], e[3][1]], "-", color=colour_array[ee], linewidth=lw)
    plt.pause(0.0001)
    assignment_edges.append([e[0], e[1]])
    start_edges.append([e[2][0], e[2][1]])
    end_edges.append([e[3][0], e[3][1]])
    sigmaTmp = (std[e[0]] + std[e[1]]) / 2
    edges_cov.append(np.sqrt(np.minimum(sigmaTmp[0][0], sigmaTmp[1][1])))

    # plt.plot([e[2][0], e[3][0]], [e[2][1], e[3][1]], "-", linewidth=2, color='#FF694F')


lims = ct.WGS84toENU([11.5, 12.4], [54.17, 54.7], ref).T
lims = np.delete(lims, np.s_[2], axis=1)
# lims=[lims[0,0],lims[1,0],lims[0,1],lims[1,1]]
# plt.xlim(lims[0, 0], lims[1, 0])
# plt.ylim(lims[0, 1], lims[1, 1])
# plt.xlim(-13061,25788)

# plt.show()
#    plt.plot([e[2][0],e[3][0]],[e[2][1],e[3][1]],"g-")

# plt.savefig('/home/docker/hostShare/Pictures/graph.png', format='png', dpi=300, transparent=True)

def plot_data_graph(): # in WGS84 (in degree):
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([8, 6])

    for mmsi in tkeys[:nr_of_vessels]:  # [vessel_nr:vessel_nr+1]: #
        decoded_mmsi = data[data['mmsi'] == mmsi]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        # axs.plot(np.array(decoded_mmsi["x"]), np.array(decoded_mmsi["y"]), c="k", linestyle= "--", linewidth=0.5)
        axs.scatter(np.array(decoded_mmsi["x"]), np.array(decoded_mmsi["y"]), c="blue",
                    marker=".", s=0.001, alpha=0.5)  # s=0.05
    for ee, e in enumerate(GEdges):
        # lw = np.clip(10 * edge[e[0]][e[1]] / max(maxh), 0.1, 10)
        n_wgs = ct.ENUtoWGS84(tENU=[[e[2][0], e[3][0]], [e[2][1], e[3][1]],[0, 0]], tREF=ref)
        axs.plot([n_wgs[0,0], n_wgs[0,1]], [n_wgs[1,0], n_wgs[1,1]], linestyle="-", color="black", linewidth=2)
        axs.plot([n_wgs[0,0], n_wgs[0,1]], [n_wgs[1,0], n_wgs[1,1]], marker=".", color="black", markersize=15)
        plt.pause(0.0001)

    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    axs.set_xlim(WINDOW[0], WINDOW[2])
    axs.set_ylim(WINDOW[1], WINDOW[3])
    plt.pause(0.001)
    plt.savefig("./results/cluster_graph_new.png")
    plt.savefig("./results/cluster_graph_new.pdf")
    with open("./results/graph_nodes_new.pkl", "wb") as fp:  # Pickling
        pickle.dump(mulist, fp)
    with open("./results/graph_edges_new.pkl", "wb") as fp:  # Pickling
        pickle.dump(assignment_edges, fp)
    plt.show()


plot_data_graph()
for _ in range(5000):
    i=0
exit(0)
# Point to edge assignment

def point2edge(point, edge_start, edge_end):
    line_vec = np.subtract(edge_end, edge_start)
    pnt_vec = np.subtract(point, edge_start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / np.linalg.norm(line_vec)
    pnt_vec_scaled = np.multiply(pnt_vec, 1.0/line_len)
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = np.multiply(line_vec, t)
    distance = np.linalg.norm(np.subtract(nearest, pnt_vec))
    return distance


'''
# choose random track from folder
randTrack = str(1) #167 str(random.randint(0, 203))
trackName = "/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/track"+randTrack+".pkl"

data1 = np.array(pd.read_pickle(trackName))
clm_idx = 2

########## other dataset ###################
trackName = "/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/rostock_gedsar_interpol_1min.csv"
data1 = np.array(pd.read_csv(trackName))
clm_idx = 0


nData1 = data1.shape[0]
lon = np.array(data1[:, clm_idx], dtype=float).reshape([1, nData1])
lat = np.array(data1[:, clm_idx+1], dtype=float).reshape([1, nData1])
cog = np.array(data1[:, clm_idx+2], dtype=float).reshape([1, nData1])
sog = np.array(data1[:, clm_idx+3], dtype=float).reshape([1, nData1])

'''

data1 = get_ais_data_interpol_1min()

clm_idx = 0
nData1 = data1.shape[0]
lon = np.array(data1[:, clm_idx], dtype=float).reshape([1, nData1])
lat = np.array(data1[:, clm_idx+1], dtype=float).reshape([1, nData1])

ENUcoord = ct.WGS84toENU(lon, lat, ref)
ENUcoord = np.transpose(ENUcoord)
ENUcoord = np.delete(ENUcoord, np.s_[2], axis=1)


nodes = mulist
edges = assignment_edges
data = ENUcoord.tolist()

cost_matrix = np.zeros((len(data), len(edges)))

Nall = len(data)
# cut_off = 2 * 25
order = 2
assignments = []
plt.figure()
plt.pause(0.0001)
for ii, points in enumerate(data):
    for jj, items in enumerate(edges):
        cut_off = 2 * edges_cov[jj]
        start = start_edges[jj]
        end = end_edges[jj]
        cost_matrix[ii, jj] = np.minimum(point2edge(points, start, end), cut_off) ** order

    e_idx = np.argmin(cost_matrix[ii, :])  # edge id
    if min(cost_matrix[ii, :]) > 100000:
        e_idx = len(GEdges) # ensures color as black

    assignments.append((ii, e_idx))

    plt.plot(points[0], points[1], '.', color=colour_array[e_idx], markersize=8)
    # plt.pause(0.00001)

plt.figure()
EPSILON = 2000
ENUcoord_subset = rdp(ENUcoord, EPSILON)
rdpNodes = np.array(ENUcoord_subset)
assignments = np.array(assignments)
start = 0
max_num = 0
for i in range(1,len(rdpNodes)):
    end = np.where((ENUcoord == rdpNodes[i,:]).all(axis=1))[0][0]
    unique_labels = np.unique(assignments[start:end, 1])
    label = unique_labels[0]
    max_num = 0
    for j in unique_labels:
        if max_num < sum(assignments[start:end, 1]== j):
            label = j
            max_num = sum(assignments[start:end, 1]== j)
    assignments[start:end, 1] = label
    plt.plot(ENUcoord[start:end, 0], ENUcoord[start:end, 1], ".",color=colour_array[label], markersize=8)
    plt.pause(0.00001)
    start = end
plt.plot(rdpNodes[:, 0], rdpNodes[:, 1], "rx", markersize=8)
plt.show()
print('end')