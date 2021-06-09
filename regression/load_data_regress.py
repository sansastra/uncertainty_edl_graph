# -*- coding: utf-8 -*-
# @Time    : 19.04.21 10:29
# @Author  : sing_sd

import os
from pathlib import Path
import numpy as np
import pandas as pd
from random import randint
from numpy import array
from numpy import argmax
from math import sin, cos, sqrt, atan2, radians
#from geopy import distance
import matplotlib.pyplot as plt
import os
import h5py
import pickle
from PIL import Image
from datetime import date, time, datetime
import src.clustering.COStransforms as ct
import src.common_functions as cf
import src.regression.process as process
import src.clustering.cluster_association as ca

vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "resources/")
interactive = True
ref = {'lon': 12.0, 'lat': 54.35, 'ECEF': ct.WGS84toECEF(12.0, 54.35)}
### Select the date range
WINDOW = [11.5, 54.2, 12.5, 54.5]
SOG_LIMIT = [0, 30]
NAV_STATUS = 0.0

def load_data(features, INPUT_LEN, PRED_LEN, graph, dataset):
    np.random.seed(10)
    dim = len(features)
    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN
    print("Data Reading")
    if dataset == "1":
        filename = "ais_data_rostock_2016_processed.csv"
    elif dataset == "3":
        filename = "ais_data_rostock_2020_processed.csv"
    else:
        filename = "ais_data_rostock_gedsar_2016.csv"
    with open("../resources/"+filename, 'rb') as f:
        data = pd.read_csv(f)
    data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                        'nav_status': 'float', 'ship_type': 'float', 'destination': 'str'})


    SHIP_TYPES = [60, 90]
    idx = cf.get_filtered_data_idx(data, WINDOW, SOG_LIMIT, NAV_STATUS, SHIP_TYPES)
    data = data.iloc[idx, :]
    data = data.reset_index(drop=True)
    print("Data loaded, total messages within window = ", len(data))
    mmsis = data.mmsi.unique()
    nr_of_vessels = int(len(mmsis)*0.6) # 60% for training
    if dataset == "4":
        nr_of_vessels = len(mmsis)
    SAMPLING_TIME = 60
    overall_data = np.full(shape=(int(2* len(data)), in_clm_len+out_clm_len), fill_value=np.nan)
    nr_track = 0
    index = 0
    for mmsi in mmsis[:nr_of_vessels]:  # [vessel_nr:vessel_nr+1]: #
        data_mmsi = data[data['mmsi'] == mmsi]
        data_mmsi = data_mmsi.reset_index(drop=True)
        nr_data = int(np.ceil((np.array(data_mmsi.iloc[-1]['time']) - data_mmsi.iloc[0]['time']) / SAMPLING_TIME) + 1)
        if nr_data > (INPUT_LEN+PRED_LEN):
            nr_track += 1
            data_mmsi['time_idx'] = data_mmsi.iloc[0]['time']
            data_mmsi['time_idx'] = np.ceil((data_mmsi['time'] - data_mmsi['time_idx']) / SAMPLING_TIME)
            overall_data[np.array(index + data_mmsi['time_idx'], dtype=np.int), 0:4] = \
                                                        np.array(data_mmsi[['x','y','cog','sog']])

            # plot_trajectory(overall_data[index: index + nr_data, 0:2])
            # index += nr_data
            # get cluster number
            if graph:
                overall_data[np.array(index + data_mmsi['time_idx'], dtype=np.int), dim - 1] = ca.get_assignment(
                                        overall_data[np.array(index + data_mmsi['time_idx'], dtype=np.int), 0:2])
            ################ convert to ENU #######################
            idx = overall_data[index:index + nr_data, 0] > 0

            # if ENU:
            #     overall_data[index + np.where(idx)[0], 0:2] = \
            #                                             convertENU(overall_data[index + np.where(idx)[0], 0:2])
            ######### Scaling #####################
            if graph:
                overall_data[index + np.where(idx)[0], 0:dim] = process.scale_lon_lat_cog_sog_cluster(
                                                                    overall_data[index + np.where(idx)[0], 0:dim])
            else:
                overall_data[index + np.where(idx)[0], 0:dim] = process.scale_lon_lat_cog_sog(
                                                                overall_data[index + np.where(idx)[0], 0:dim])
            # reverse data_mmsi now as overall_data will change np.nan, writing reversed data after putting nan
            overall_data[index+nr_data: index + 2*nr_data, 0:dim] =  \
                                                        np.flipud(overall_data[index: index+nr_data, 0:dim]) #
            # fill data horizontally if real data is there for, first time for normal and 2nd time for reverse
            for _ in range(2):
                for i in np.where(idx)[0]:
                    # eventhough idx is say 1x10 array, idx[8:15] results in only 2 values idx[8:10]
                    if sum(idx[i:i+INPUT_LEN+PRED_LEN]) == INPUT_LEN+PRED_LEN: # then no need of x-axis interpolation
                        for j in range(dim): # for each feature fill the column of the ith row
                            overall_data[index+i, j+dim:j+(INPUT_LEN+PRED_LEN)*dim:dim] = \
                                overall_data[index+i+1:index+i+INPUT_LEN+PRED_LEN, j]
                    else:
                        overall_data[index+i, 0] = np.nan
                # reverse dataset fill columns
                index += nr_data
                idx = overall_data[index:index + nr_data, 0] > 0

####################################################################################
    overall_data = overall_data[0:index,:]
    overall_data = overall_data[overall_data[:, 0] > 0]
    # for j in range(dim-1):
    #     overall_data[:, j:(INPUT_LEN+PRED_LEN)*dim:dim] = overall_data[:, j:(INPUT_LEN+PRED_LEN)*dim:dim] \
    #                      .interpolate(method='linear', columns=features, limit_direction='forward', axis=0)

    if np.isnan(overall_data).any():
        print("nan data")
        exit(0)
    print('Overall data size= ', len(overall_data))
    print("nr of tracks for training = ", nr_track)
    return overall_data[:, 0:INPUT_LEN*dim], overall_data[:, INPUT_LEN*dim:]


def load_dataset_2019(dim, INPUT_LEN, PRED_LEN, graph):
    np.random.seed(10)
    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN
    with open('./resources/ais_data_1min_graph.csv', 'rb') as f:
        data = pd.read_csv(f)
        # data.columns = ["x", "y", "cog", "sog", "cluster", "mmsi"]
    # Update cluster nr.
    # print("clustering")
    # for mmsi in data.mmsi.unique():
    #     idx = np.where(data["mmsi"]==mmsi)[0]
    #     data.iloc[idx, 4] = ca.get_assignment(data.iloc[idx, 0:2])

    # data["cluster"] = ca.get_assignment(data[["x", "y", "cog", "sog", "mmsi"]])
    idx = np.array(data['x']) > WINDOW[0]
    idx &= np.array(data['y']) > WINDOW[1]
    idx &= np.array(data['x']) < WINDOW[2]
    idx &= np.array(data['y']) < WINDOW[3]

    mmsis = data.mmsi.unique()
    data = np.array(data)
    # # convert lon, lat in ENU
    # data = convertENU(data)
    # scale data
    if graph:
        data = process.scale_lon_lat_cog_sog_cluster(data)
    else:
        data = process.scale_lon_lat_cog_sog(data)

    data[~idx, 0] = np.nan
    nr_of_vessels = int(len(mmsis)*0.6)  # max nr of vessels in this dataset is 146
    legecy_seq_len = 60
    overall_data = np.full(shape=(int(2 * len(data) + (legecy_seq_len - 1) * nr_of_vessels), in_clm_len+out_clm_len),
                           fill_value=np.nan)
    mmsis = np.unique(data[:, -1])
    nr_track = 0
    index = 0
    for track_nr in range(nr_of_vessels):  # some vessels have less data so ignored
        mmsi = mmsis[track_nr]
        data_mmsi = data[data[:, -1] == mmsi,:]
        nr_data = len(data_mmsi)
        # if INPUT_LEN + PRED_LEN < nr_data < 500:
            # plot_trajectory(data[startIndex: startIndex + nr_sampled_data, 0:2])
        nr_track += 1
        # plot_trajectory(data[startIndex: startIndex + nr_sampled_data, 0:2])
        overall_data[index: index + nr_data, 0:dim] = data_mmsi[:, 0:dim]
        # shift from top and put on remaining columns
        for clm_nr in range(1, INPUT_LEN+PRED_LEN):
            overall_data[index: index + nr_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
                overall_data[index + 1: index + nr_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]
        # clear last few rows that does not have sufficient data to make ...
        overall_data[index + nr_data - INPUT_LEN - PRED_LEN + 1: index + nr_data] = np.nan

        # reverse dataset
        index += nr_data - INPUT_LEN - PRED_LEN + 1
        overall_data[index: index + nr_data, 0:dim] =  np.flipud(data_mmsi[:, 0:dim])
        # shift from top and put on remaining columns
        for clm_nr in range(1, INPUT_LEN+PRED_LEN):
            overall_data[index: index + nr_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
                overall_data[index + 1: index + nr_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]
        # clear last few rows that does not have sufficient data to make ...
        overall_data[index + nr_data - INPUT_LEN - PRED_LEN + 1: index + nr_data] = np.nan
        index += nr_data - INPUT_LEN - PRED_LEN + 1

    overall_data = overall_data[0:index, :]
    overall_data = overall_data[~np.isnan(overall_data).any(axis=1)]
    if np.isnan(overall_data).any():
        print("nan data")
        exit(0)
    print('Overall data size= ', len(overall_data))
    print("nr of tracks for training = ", nr_track)
    return overall_data[:, :in_clm_len], overall_data[:, in_clm_len:]


def load_data_rostock_gedsar(dim, INPUT_LEN, PRED_LEN):
    np.random.seed(10)
    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN

    with open("../resources/rostock_gedsar_interpol_1min.csv", 'rb') as f:
        data = pd.read_csv(f)
        data.columns = ["x", "y", "cog", "sog"]
    idx = np.array(data['y']) > WINDOW[1]
    idx &= np.array(data['y']) < WINDOW[3]

    data = np.array(data)

    # # convert lon, lat in ENU
    # data = convertENU(data)
    # scale data
    data = process.scale_lon_lat_cog_sog(data)

    data[~idx, 0] = np.nan
    overall_data = np.full(shape=(2*len(data), in_clm_len+out_clm_len), fill_value=np.nan)
    index = 0
    nr_data = len(data)
    overall_data[index: index + nr_data, 0:dim] = data[:, 0:dim]
    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN + PRED_LEN):
        overall_data[index: index + nr_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
            overall_data[index + 1: index + nr_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]
    # clear last few rows that does not have sufficient data to make ...
    overall_data[index + nr_data - INPUT_LEN - PRED_LEN + 1: index + nr_data] = np.nan

    # reverse dataset
    index += nr_data - INPUT_LEN - PRED_LEN + 1
    overall_data[index: index + nr_data, 0:dim] = np.flipud(data[:, 0:dim])
    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN + PRED_LEN):
        overall_data[index: index + nr_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
            overall_data[index + 1: index + nr_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]
    # clear last few rows that does not have sufficient data to make ...
    overall_data[index + nr_data - INPUT_LEN - PRED_LEN + 1: index + nr_data] = np.nan
    index += nr_data - INPUT_LEN - PRED_LEN + 1

    overall_data = overall_data[0:index, :]
    overall_data = overall_data[~np.isnan(overall_data).any(axis=1)]
    if np.isnan(overall_data).any():
        print("nan data")
        exit(0)
    print('Overall data size= ', len(overall_data))
    print("nr of tracks for training = 1")
    return overall_data[:, :in_clm_len], overall_data[:, in_clm_len:]

def plot_trajectory(data):
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(data[:,0], data[:,1], 'k.',markersize=2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.pause(0.00001)

def load_test_data_gedsar(INPUT_LEN, TARGET_LEN, features,  dim, ANO_DATA, cluster):
    in_clm_len = INPUT_LEN * dim

    if ANO_DATA:
        filename = 'gedsar_ano_left.csv' #'gedsar_ano_right.csv' # 'gedsar_ano_left.csv'
    else:
        filename = 'gedsar_data.csv'
    with open("./resources/"+filename, 'rb') as f:
        data = pd.read_csv(f, sep=",", header=None)
    idx = np.array(data.iloc[:, 1]) > 54.2
    idx &= np.array(data.iloc[:, 1])< 54.5
    data = data.iloc[idx, :]
    if cluster:
        data["cluster"] = ca.get_assignment(data.iloc[:, 0:2])
    # data.iloc[:, 0:2] = convertENU(np.array(data.iloc[:, 0:2]))
    data = np.array(data)
    nr_sampled_data = len(data)
    overall_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)
    overall_data[:,0:dim] = data
    # target_data = data

    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN):
        overall_data[0: nr_sampled_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
            overall_data[1: nr_sampled_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]

    return data, overall_data, overall_data


def load_test_data_vk2mmsi(INPUT_LEN, TARGET_LEN, features,  dim, ANO_DATA):
    in_clm_len = INPUT_LEN * dim
    out_clm_len = TARGET_LEN * dim
    interpolate_interval = 60  # seconds
    SAMPLING_TIME = 1  # second
    if ANO_DATA:
        filename = os.path.join(data_dir, 'resampled_1min_ais_data_211724970.csv')
    else:
        print("no normal data")
        exit(0)
    # filename = path / 'Track167_interpolated_1min.csv'
    with open(filename, 'rb') as f:
        data = pd.read_csv(f, sep=",", header=None)
    data = np.array(data)
    data = data[1:]
    nr_sampled_data = len(data)
    overall_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)
    overall_data[:,0:dim] = data
    # target_data = data

    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN):
        overall_data[0: nr_sampled_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
            overall_data[1: nr_sampled_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]

    return data, overall_data, overall_data

def load_test_data(data_mmsi, INPUT_LEN, PRED_LEN, features, graph):
    np.random.seed(10)
    SAMPLING_TIME = 60
    dim = len(features)
    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN
    nr_data = int(np.ceil((np.array(data_mmsi.iloc[-1]['time']) - data_mmsi.iloc[0]['time']) / SAMPLING_TIME) + 1)
    overall_data = np.full(shape=(nr_data, in_clm_len + out_clm_len), fill_value=np.nan)
    index = 0
    if nr_data > INPUT_LEN + PRED_LEN:
        data_mmsi['time_idx'] = data_mmsi.iloc[0]['time']
        data_mmsi['time_idx'] = np.ceil((data_mmsi['time'] - data_mmsi['time_idx']) / SAMPLING_TIME)
        overall_data[np.array(index + data_mmsi['time_idx'], dtype=np.int), 0:4] = \
            np.array(data_mmsi[['x', 'y', 'cog', 'sog']])
        lon_lat_plot = overall_data[:, 0:2].copy() # some nan data helps in plotting with discontinuity
        # plot_trajectory(overall_data[index: index + nr_data, 0:2])
        # index += nr_data
        # get cluster number
        if graph:
            overall_data[np.array(index + data_mmsi['time_idx'], dtype=np.int), dim - 1] = ca.get_assignment(
                overall_data[np.array(index + data_mmsi['time_idx'], dtype=np.int), 0:2])
        ################ convert to ENU #######################
        idx = overall_data[index:index + nr_data, 0] > 0
        # overall_data[index + np.where(idx)[0], 0:2] = \
        #     convertENU(overall_data[index + np.where(idx)[0], 0:2])
        ######### Scaling #####################
        if graph:
            overall_data[index + np.where(idx)[0], 0:dim] = process.scale_lon_lat_cog_sog_cluster(
                overall_data[index + np.where(idx)[0], 0:dim])
        else:
            overall_data[index + np.where(idx)[0], 0:dim] = process.scale_lon_lat_cog_sog(
                overall_data[index + np.where(idx)[0], 0:dim])
        # fill data horizontally if real data is there for, first time for normal and 2nd time for reverse

        for i in np.where(idx)[0]:
            # eventhough idx is say 1x10 array, idx[8:15] results in only 2 values idx[8:10]
            if sum(idx[
                   i:i + INPUT_LEN + PRED_LEN]) == INPUT_LEN + PRED_LEN:  # then no need of x-axis interpolation
                for j in range(dim):  # for each feature fill the column of the ith row
                    overall_data[index + i, j + dim:j + (INPUT_LEN + PRED_LEN) * dim:dim] = \
                        overall_data[index + i + 1:index + i + INPUT_LEN + PRED_LEN, j]
            else:
                overall_data[index + i, 0] = np.nan

        index += nr_data
    else:
        print("not enough data for the vessel")
        return None, None, None
    ####################################################################################
    overall_data = overall_data[0:index, :]
    overall_data = overall_data[overall_data[:, 0] > 0]
    # for j in range(dim-1):
    #     overall_data[:, j:(INPUT_LEN+PRED_LEN)*dim:dim] = overall_data[:, j:(INPUT_LEN+PRED_LEN)*dim:dim] \
    #                      .interpolate(method='linear', columns=features, limit_direction='forward', axis=0)

    if np.isnan(overall_data).any():
        print("nan data")
        exit(0)
    print('Overall data size= ', len(overall_data))

    return overall_data[:, 0:INPUT_LEN * dim], \
           process.inverse_transform_lon_lat(overall_data[:, INPUT_LEN * dim:INPUT_LEN * dim + 2]), lon_lat_plot

def load_test_data_ENU(INPUT_LEN, TARGET_LEN, features,  dim, track_to_check):

    in_clm_len = INPUT_LEN*dim
    out_clm_len = TARGET_LEN * dim
    interpolate_interval = 60 # seconds
    SAMPLING_TIME = 1 # second
    data_features = ["x", "y", "cog", "sog"]
    data_dim = len(data_features)
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'

    filename = 'track{}'.format(track_to_check)

    try:
        data = pd.read_pickle(path + filename + '.pkl')
    except IOError:
        print("Error: File does not appear to exist for track ", track_to_check)
        return 0, 0
    # without interpolation
    original_data = np.array([data.x, data.y]).transpose()

    start_time = datetime.strptime(data.iloc[0]['date'] + ' ' + data.iloc[0]['time'],
                                   '%m/%d/%Y %H:%M:%S')
    end_time = datetime.strptime(data.iloc[-1]['date'] + ' ' + data.iloc[-1]['time'],
                                 '%m/%d/%Y %H:%M:%S')

    data_per_track = int((end_time - start_time).total_seconds() // interpolate_interval + 1) # interpolation interval = 2 seconds


    sampling_indices = range(0, data_per_track, SAMPLING_TIME)
    nr_sampled_data = len(sampling_indices)
    overall_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)

    temp_data = pd.DataFrame(index=range(data_per_track), columns=data_features, dtype=np.float)
    for slot_index in range(0, data.shape[0]):  # //
        current_time = datetime.strptime(data.iloc[slot_index]['date'] + ' ' + data.iloc[slot_index]['time'],
                                         '%m/%d/%Y %H:%M:%S')
        index1 = int((current_time - start_time).total_seconds()) // interpolate_interval
        temp_data.iloc[index1, 0:data_dim] = data.iloc[slot_index, 2:data_dim+2]

    # interpolate
    temp_data.iloc[0:sampling_indices[INPUT_LEN], :] = temp_data.iloc[0:sampling_indices[INPUT_LEN], :].interpolate(method='linear', limit_direction='forward', axis=0)
    resampled_data = temp_data.iloc[sampling_indices, 0:data_dim]

    # resampled_data.to_csv("Track167.csv", index=False)

    # put lon and lat data
    overall_data[0:nr_sampled_data, 0:2] = temp_data.iloc[sampling_indices, 0:2]

    temp_data_interpolated = resampled_data.interpolate(method='linear', limit_direction='forward', axis=0)
    # temp_data_interpolated.to_csv("Track167_interpolated_1min.csv",index=False)
    temp_data_interpolated = np.array(temp_data_interpolated.iloc[:, 0:data_dim])
    # aa = temp_data.iloc[sampling_indices, 0:dim]
    # aa.to_csv("Track167.csv",index=False)
    # convert lon, lat in ENU
    lon = np.transpose(overall_data[:, 0])
    lat = np.transpose(overall_data[:, 1])

    zENU = ct.WGS84toENU(lon, lat, ref)

    overall_data[:, 0] = np.transpose(zENU[0, :])
    overall_data[:, 1] = np.transpose(zENU[1, :])
    # put z data
    overall_data[:, 2] = np.transpose(zENU[2, :])  # insert zeros on 3rd column in data for
    # put cog and sog
    overall_data[:, 3:dim] = temp_data.iloc[sampling_indices, 2:data_dim]
    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN):
        overall_data[0: nr_sampled_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                                                                     1: nr_sampled_data - clm_nr + 1,
                                                                                     (clm_nr - 1) * dim:clm_nr * dim]

    return np.array(resampled_data), overall_data, temp_data_interpolated

def load_test_interpolated_data(INPUT_LEN, TARGET_LEN, features,  dim, track_to_check):
    in_clm_len = INPUT_LEN*dim
    out_clm_len = TARGET_LEN * dim
    interpolate_interval = 60 # seconds
    SAMPLING_TIME = 1
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'

    filename = 'track{}'.format(track_to_check)

    try:
        data = pd.read_pickle(path + filename + '.pkl')
    except IOError:
        print("Error: File does not appear to exist for track ", track_to_check)
        return 0, 0
    # without interpolation
    original_data = np.array([data.x, data.y]).transpose()

    start_time = datetime.strptime(data.iloc[0]['date'] + ' ' + data.iloc[0]['time'],
                                   '%m/%d/%Y %H:%M:%S')
    end_time = datetime.strptime(data.iloc[-1]['date'] + ' ' + data.iloc[-1]['time'],
                                 '%m/%d/%Y %H:%M:%S')

    data_per_track = int((end_time - start_time).total_seconds() // interpolate_interval + 1) # interpolation interval = 2 seconds

    sampling_indices = range(0, data_per_track, SAMPLING_TIME)
    nr_sampled_data = len(sampling_indices)
    overall_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)

    temp_data = pd.DataFrame(index=range(data_per_track), columns=features, dtype=np.float)
    for slot_index in range(0, data.shape[0]):  # //
        current_time = datetime.strptime(data.iloc[slot_index]['date'] + ' ' + data.iloc[slot_index]['time'],
                                         '%m/%d/%Y %H:%M:%S')
        index1 = int((current_time - start_time).total_seconds()) // interpolate_interval
        temp_data.iloc[index1, 0:dim] = data.iloc[slot_index, 2:dim+2]

    # interpolate
    temp_data = temp_data.interpolate(method='linear', limit_direction='forward', axis=0)

    overall_data[0:nr_sampled_data, 0:dim] = temp_data.iloc[sampling_indices, 0:dim]


    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN):
        overall_data[0: nr_sampled_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                           1: nr_sampled_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]

    target_data[0:nr_sampled_data - INPUT_LEN - TARGET_LEN + 1, 0:out_clm_len] = overall_data[
                 INPUT_LEN: nr_sampled_data - TARGET_LEN + 1, 0:out_clm_len]

    # clear last few rows that does not have sufficient data to make ...
    overall_data[nr_sampled_data - INPUT_LEN + 1: nr_sampled_data] = np.nan

    overall_data = overall_data[~np.isnan(overall_data[:, 0])]
    target_data = target_data[~np.isnan(target_data[:, 0])]
    target_data = np.nan_to_num(target_data)
    return original_data, overall_data, target_data


def convertENU(data):
    # # convert lon, lat in ENU
    zENU = ct.WGS84toENU(np.transpose(data[:, 0]), np.transpose(data[:, 1]), ref)
    data[:, 0] = np.transpose(zENU[0, :])
    data[:, 1] = np.transpose(zENU[1, :])
    # if no z dimension then comment next line
    # data = np.insert(data, 2, np.transpose(zENU[2, :]), axis=1)  # insert zeros on 3rd column in data for
    return data

# def load_rg_into_overall():
#     ########### Load Rostock-gedsar data trial ##########
#     with open('../resources/rostock_gedsar_interpol_1min.csv', 'rb') as f:
#         data = pd.read_csv(f)
#     data = np.array(data)
#     nr_data = len(data)
#     overall_data[index: index + nr_data, 0:dim - 1] = data[:, 0:dim - 1]
#     overall_data[index: index + nr_data, dim - 1:dim] = ca.get_assignment(
#         overall_data[index: index + nr_data, 0:2])
#     # convert to ENU #
#     overall_data[index: index + nr_data, 0:2] = \
#         convertENU(overall_data[index: index + nr_data, 0:2])
#     # Scaling #
#     overall_data[index: index + nr_data, 0:dim] = process.scale_lon_lat_cog_sog_cluster(
#         overall_data[index: index + nr_data, 0:dim])
#
#     for clm_nr in range(1, INPUT_LEN + PRED_LEN):
#         overall_data[index: index + nr_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = \
#             overall_data[index + 1: index + nr_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]
#     overall_data[index + nr_data - INPUT_LEN - PRED_LEN: index + nr_data, 0] = np.nan
#     index += nr_data
#     ###################