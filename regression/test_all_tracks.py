# -*- coding: utf-8 -*-
# @Time    : 26.04.21 16:41
# @Author  : sing_sd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for removing unnecessary warnings
from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('...')
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import pickle
import random
import json
import matplotlib.pyplot as plt
import src.common_functions as cf
import src.clustering.COStransforms as ct
import src.regression.process as  process
from src.regression.load_data_regress import load_test_data, load_test_interpolated_data, load_test_data_ENU, load_test_data_gedsar, load_test_data_vk2mmsi
########### check a whole track ##############
from src.evidential_deep_learning.layers.dense import DenseNormalGamma
from src.evidential_deep_learning.losses.continuous import EvidentialRegression
# edl.losses.EvidentialRegression does work, since this custom loss function has a get_config method
# that has to be changed in the argument, add **kwarg
# Also DenseNormalGamma is imported locally otherwise saved model does not recognise this layer
def EvidentialRegressionLoss(true, pred):
    return EvidentialRegression(true, pred, coeff=1e-2)

plt.rcParams.update({'font.size': 12})

vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "resources/")
results_dir = os.path.join(vb_dir, "results/")
# ref = {'lon': 12.0, 'lat': 54.35, 'ECEF': ct.WGS84toECEF(12.0, 54.35)}
# Train_all = True # its default
INPUT_LEN = 10  # same as timesteps
TARGET_LEN = 1
# for computing anomalous traj
unc_thr = 0.6 # 0.3, 0.6
pred_dist_thr = 30
segment_size = 30
cluster = True
dataset = "3"  # 1-> jan 2016, 2-> july 2019, 3=> Jan 2020, 4=> rostock_degsar
WINDOW = [11.5, 54.2, 12.5, 54.5]
SOG_LIMIT = [0, 30]
NAV_STATUS = 0.0
SHIP_TYPES = [0, 90] #[60, 90] # [0, 60] #
ml_model = "edl" # "edl", "seq"

if cluster:
    features = ['x', 'y', 'cog', 'sog', 'cluster'] # by default we convert to ENU
    model_name = ml_model+"_graph_True_ds1.h5"
else:
    features = ['x', 'y', 'cog', 'sog']
    model_name = ml_model+"_graph_False_ds1.h5"

### Select the date range
dim = len(features)
# def load_data():
if dataset == "1":
    filename = "ais_data_rostock_2016_processed.csv"
elif dataset == "3":
    filename = "ais_data_rostock_2020_processed.csv" # to be selected later

with open("../resources/"+filename, 'rb') as f:
    data = pd.read_csv(f)
data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                    'nav_status': 'float', 'ship_type': 'float', 'destination': 'str'})

idx = cf.get_filtered_data_idx(data, WINDOW, SOG_LIMIT, NAV_STATUS, SHIP_TYPES)
data = data.iloc[idx, :]
data = data.reset_index(drop=True)
print("Data loaded, total messages within window = ", len(data))
# return data
if ml_model == "edl":
    model = keras.models.load_model(data_dir+model_name,
    custom_objects={'DenseNormalGamma': DenseNormalGamma, "EvidentialRegressionLoss": EvidentialRegressionLoss})
else:
    model = keras.models.load_model(data_dir+model_name)

def main():
    # compute_prediction_error()
    # plot_cdf_error()
    compute_uncertainties()

    # exit(0)

def compute_prediction_error():
    mmsis = data.mmsi.unique()
    if SHIP_TYPES[0] < 59:
        nr_of_vessels = 0
    else:
        nr_of_vessels = int(len(mmsis)*0.6)

    # max_error = [0.0] * len(mmsis)
    # avg_error = [0.0] * len(mmsis)
    # min_error = [0.0] * len(mmsis)
    error_all = {}
    i = 0
    for mmsi in mmsis[nr_of_vessels:]:  # [vessel_nr:vessel_nr+1]: #
        data_mmsi = data[data['mmsi'] == mmsi]
        data_mmsi = data_mmsi.reset_index(drop=True)
        x_test, y_test, _ = load_test_data(data_mmsi, INPUT_LEN, TARGET_LEN, features, cluster)
        if y_test is None:
            i += 1
            continue
        x_test.shape = (x_test.shape[0], INPUT_LEN, dim)
        y_pred = model.predict(x_test)
        if ml_model == "edl":
            y_pred = get_lon_lat(y_pred)
        else: # seq
            y_pred.shape = (y_pred.__len__(), TARGET_LEN * dim)
        # invert predictions
        y_pred = process.inverse_transform_lon_lat(y_pred)
        distance_error = np.abs(np.sqrt(
            (y_test[:, 0] - y_pred[:, 0]) ** 2 + (y_test[:, 1] - y_pred[:, 1]) ** 2))*60*1.85
        error_all[i] = distance_error.tolist() # tolist() enable np ndarray for json serialization, otherwise error
        # max_error[i]= np.max(distance_error)
        # avg_error[i] = np.mean(distance_error)
        # min_error[i] = np.min(distance_error)
        i += 1
        print("track processed=", i)

    with open("./resources/dist_error_"+ml_model+"_graph_" + str(cluster) + "_ship_"+str(SHIP_TYPES[0])+
               "_"+str(SHIP_TYPES[1])+"_ds" + dataset +".json", 'w') as f:
        json.dump(error_all, f)

    # with open("./resources/dist_error_graph_60_90_all_ds"+dataset+".pkl", "wb") as fp:  # Pickling
    #     pickle.dump(error_all, fp)


def plot_cdf_error():
    f_all = {}
    if ml_model == "edl":
        f_all[0] = "dist_error_edl_graph_False_ship_0_60_ds1.json"
        f_all[1] = "dist_error_edl_graph_True_ship_0_60_ds1.json"
        f_all[2] = "dist_error_edl_graph_False_ship_60_90_ds1.json"
        f_all[3] = "dist_error_edl_graph_True_ship_60_90_ds1.json"
    else:
        f_all[0] = "dist_error_seq_graph_False_ship_0_60_ds1.json"
        f_all[1] = "dist_error_seq_graph_True_ship_0_60_ds1.json"
        f_all[2] = "dist_error_seq_graph_False_ship_60_90_ds1.json"
        f_all[3] = "dist_error_seq_graph_True_ship_60_90_ds1.json"

    dist_error = {}
    x1 = {}
    cdf = {}
    for i in f_all:
        with open("./resources/"+f_all[i], 'rb') as f:
            dist_error[i] = json.load(f)
        print("file import succeeded", i)
        x1[i], cdf[i] = compute_cdf(dist_error[i])

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([6, 5])
    axs.plot(x1[0], cdf[0], c="black", linestyle="-",  label="W/O Graph, ship_type=[0, 60)")
    axs.plot(x1[1], cdf[1], c="blue", linestyle="-.",  label="Graph, ship_type=[0, 60)")
    axs.plot(x1[2], cdf[2], c="green", linestyle="dotted", label="W/O Graph, ship_type=[60, 90)")
    axs.plot(x1[3], cdf[3], c="red", linestyle="--", label="Graph, ship_type=[60, 90)")
    axs.set_xlabel('Prediction error [km]')
    axs.set_ylabel('Cumulative density function (CDF)')
    axs.set_xlim(0, 60)
    axs.set_ylim(0, 1.01)
    plt.legend()
    axs.grid(True)
    plt.pause(0.001)
    plt.savefig("./results/cdf_error_all_"+ml_model+".png")
    plt.savefig("./results/cdf_error_all_"+ml_model+".pdf")
    plt.show()


def compute_uncertainties():
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches([7, 5])
    plt.pause(0.0001)


    mmsis = data.mmsi.unique()
    nr_of_vessels = int(len(mmsis)*0.6)  # > 300  for [60-90]
    if SHIP_TYPES[0]< 60 or dataset != "1":
        nr_of_vessels = 0
    i = 0
    for mmsi in mmsis[nr_of_vessels:]:  # [vessel_nr:vessel_nr+1]: #
        data_mmsi = data[data['mmsi'] == mmsi]
        data_mmsi = data_mmsi.reset_index(drop=True)
        x_test, y_test, lon_lat_test = load_test_data(data_mmsi, INPUT_LEN, TARGET_LEN, features, cluster)
        if y_test is None or len(y_test)==0:
            i += 1
            continue
        x_test.shape = (x_test.shape[0], INPUT_LEN, dim)
        y_pred = model.predict(x_test)

        mu_pred, uncertainty = get_predicted_mean_uncertainty(y_pred, unc_inv_scale=False)  # sigma, alpha, v
        # transformation from ENU to degree
        #lon_lat_test = ct.ENUtoWGS84(tENU=np.array([y_test[:, 0].T, y_test[:, 1].T,
        #                                             [0.0] * len(y_test)]), tREF=ref)
        # y_test[:, 0:2] = np.transpose(lon_lat_test)[:, 0:2]

        # plot_uncertainty(uncertainty)
        # plot_predicted_track(y_test, mu_pred)

        plot_anomalous_segments(y_test, mu_pred,  uncertainty, lon_lat_test, ax)

        i += 1
        print("track processed=", i)
    ax.set_xlim(xmin=11.5, xmax=12.5)
    ax.set_ylim(ymin=54.2, ymax=54.5)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    plt.savefig("./results/anoall_graph_"+str(cluster)+"_ship_"+str(SHIP_TYPES[0])+"_" + str(SHIP_TYPES[1])+
                "_ds" + dataset + "_unc_"+str(unc_thr)[2:]+".png")
    plt.savefig("./results/anoall_graph_"+str(cluster)+"_ship_"+str(SHIP_TYPES[0])+"_" + str(SHIP_TYPES[1])+
                "_ds" + dataset + "_unc_"+str(unc_thr)[2:]+".pdf")
    print("done")
    plt.show()


def plot_anomalous_segments(y_test, y_pred, uncertainty, lon_lat_test, ax):
    ax.plot(lon_lat_test[:, 0], lon_lat_test[:, 1], c="k", linestyle="--", linewidth=0.4, alpha=1)
    plt.pause(0.0001)
    # distance_error = np.abs(np.sqrt((np.square(y_test[:, 0] - y_pred[0, 0])
    #                                 + np.square( y_test[:, 1] - y_pred[:, 1]))))*60*1.85
    step = 10
    color = "#F"+''.join([random.choice('0123456789ABCDEF') for j in range(5)])
    for index in range(0, len(y_test)-segment_size, step):

        max = np.max(uncertainty[index:index + segment_size, :], axis=0)
        min = np.min(uncertainty[index:index + segment_size, :], axis=0)
        # std = np.std(np.divide(uncertainty[index:index + segment_size, :],max), axis=0)
        if np.max(1- np.divide(min[0:2],max[0:2])) > unc_thr: # and np.max(distance_error[index:index + segment_size]) > pred_dist_thr:
        # if np.max(std[0:2]) > 0.2 and np.max(np.max(uncertainty[index:index + segment_size, 0:2],axis=0))> 0.01: #unc_thr:
            ax.plot(lon_lat_test[index+INPUT_LEN-1: index + segment_size+INPUT_LEN+1, 0],
                    lon_lat_test[index+INPUT_LEN-1: index + segment_size+INPUT_LEN+1, 1],
                       linestyle="-", c=color, linewidth=2, zorder=5)

        # if np.min(std[0:4]) > unc_thr and np.max(distance_error) > pred_dist_thr:
        #     ax.scatter(y_test[index: index+segment_size, 0], y_test[index: index+segment_size, 1], marker = ".", c="r", s=1)
    plt.pause(0.0001)


def get_predicted_mean_uncertainty(y_pred, unc_inv_scale):
    n_stds = 4
    y_pred = tf.reshape(y_pred, shape=(y_pred.__len__(), TARGET_LEN * dim * n_stds))
    uncertainty = np.zeros(shape=(len(y_pred), dim))
    mu_all = np.zeros(shape=(len(y_pred), dim))
    for feature_idx in range(dim): # 0=> lon, 1 => lat, 2 => cog, 3 => sog
        mu, v, alpha, beta = tf.split(y_pred[:, feature_idx: n_stds * dim:dim], n_stds, axis=-1)
        var_model = np.sqrt(beta / (v * (alpha - 1)))
        # var_data = np.sqrt(beta / (alpha - 1))
        var_model = np.minimum(var_model, 1e2)
        uncertainty[:, feature_idx] = var_model[:, 0]
        mu_all[:, feature_idx] = mu[:, 0]
    # inverse scaling
    if cluster:
        mu_all = process.inverse_transform_lon_lat_cog_sog_cluster(mu_all)
        if unc_inv_scale:
            uncertainty = process.inverse_transform_lon_lat_cog_sog_cluster(uncertainty)
    else:
        mu_all = process.inverse_transform_lon_lat_cog_sog(mu_all)
        if unc_inv_scale:
            uncertainty = process.inverse_transform_lon_lat_cog_sog(uncertainty)
    # transform from ENU to degrees
    # lon_lat = ct.ENUtoWGS84(tENU=np.array([mu_all[:, 0].T, mu_all[:, 1].T,
    #                              [0.0] * len(mu_all)]), tREF=ref)
    # mu_all[:, 0] = lon_lat[0, :]
    # mu_all[:, 1] = lon_lat[1, :]
    return mu_all, uncertainty


def get_lon_lat(y_pred):
    n_stds = 4
    y_pred = tf.reshape(y_pred, shape=(y_pred.__len__(), TARGET_LEN * dim * n_stds))
    lon_lat = np.zeros(shape=(len(y_pred), 2))
    for feature_idx in range(2): # 0=> lon, 1 => lat
        mu, v, alpha, beta = tf.split(y_pred[:, feature_idx: n_stds * dim:dim], n_stds, axis=-1)
        lon_lat[:, feature_idx] = mu[:, 0]
    return lon_lat


def plot_predicted_track(y_test, y_pred):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches([8, 6])
    plt.pause(0.0001)
    ax.plot(y_test[:, 0], y_test[:, 1], '.k', label='Original trajectory')
    ax.plot(y_pred[:, 0], y_pred[:, 1], 'xb', label='Predicted trajectory')
    ax.set_xlim(xmin=11.5, xmax=12.5)
    ax.set_xlim(xmin=54.2, xmax=54.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # ax.legend()
    # plt.savefig('trajectory_pred1.pdf')
    plt.pause(0.001)


def plot_uncertainty(uncertainty):
    fig1, axs = plt.subplots(dim, sharex=True, figsize=(8, 6))
    axs[0].plot(uncertainty[:, 0], "r-", label="Longitude")
    axs[1].plot(uncertainty[:, 1], "g--", label="Latitude")
    axs[1].set_ylabel('Uncertainty')
    axs[2].plot(uncertainty[:, 2], "b-", label="CoG")
    axs[3].plot(uncertainty[:, 3], "k--", label="SoG")
    if cluster:
        axs[4].plot(uncertainty[:, 4], "k-", label="Cluster")
    plt.xlabel('Time index')
    axs[0].set_title("Uncertainty in the Left Anomalous Track")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper right")
    axs[3].legend(loc="upper right")
    if cluster:
        axs[4].legend(loc="upper right")
    plt.pause(0.001)

    # plt.savefig('Unc_gedsar_ano_route_graph__all.png')
    # plt.savefig('Unc_gedsar_ano_route_graph__all.pdf')
    # plt.show()


def plot_max_avg_min_errors(max_error, avg_error, min_error):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([8, 6])
    plt.pause(0.0001)
    axs.plot(max_error, linestyle="dotted", marker="+", color="red", linewidth=2, label="max_error")
    axs.plot(avg_error, linestyle="-", marker="o", color="green", linewidth=2, label="avg_error")
    axs.plot(min_error, linestyle="--", marker=".", color="blue", linewidth=2, label="min_error")
    plt.pause(0.0001)
    axs.set_xlabel('Trajectory index')
    axs.set_ylabel('Prediction error of distance (m)')
    plt.pause(0.001)
    plt.savefig("./results/distance_error_graph.png")
    plt.savefig("./results/distance_error_graph.pdf")
    plt.show()


def compute_cdf(distance_error_all):
    distance_error = []
    for key in distance_error_all:
        distance_error = np.append(distance_error, distance_error_all[key])
    # distance_error = (distance_error)*0.54 #(change from km to nautical miles)
    bins_start = int(np.floor(np.min(distance_error)))
    bins_end = int(np.ceil(np.max(distance_error)))
    # plt.hist(distance_error, bins= np.arange(bins_start, bins_end+1), density=True)
    hist1 = np.histogram(distance_error, bins=np.arange(bins_start, bins_end + 1))
    # Create some test data
    dx = 1
    X = np.arange(bins_start, bins_end, dx)
    Y = hist1[0]
    # Normalize the data to a proper PDF
    Y = Y / np.sum(dx * Y)
    # Compute the CDF
    CY = np.cumsum(Y * dx)
    return X, CY
    # Plot both
    # plt.plot(X, Y, "-g")
    # ax.plot(X, CY, color, label = legend)
    # plt.pause(0.001)


if __name__ == "__main__":
    main()

################################################################################