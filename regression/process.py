from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# from src.trajectory_pred.load_data_traj import load_test_data, load_test_interpolated_data
from src.trajectory_pred.ENUtransform import WGS84toENU, ENUtoWGS84, WGS84toECEF


geo_area = [11.5, 12.5, 54.2, 54.5] # [11, 13, 54, 55] #
# ref = {'lon': 12.0, 'lat': 54.35, 'ECEF': WGS84toECEF(12.0, 54.35)}
# geo_area_ENU = WGS84toENU(geo_area[0:2], geo_area[2:4], ref, h=0.)
# geo_area_ENU = np.delete(geo_area_ENU.T, np.s_[2],axis=1)
geo_area_ENU = np.array(geo_area).reshape(2,2).T
cluster_max = 7.0 # 0-6 for real edges + 1 for noise
sog_max = 30.0


def scale_lon_lat_cog_sog_cluster(data):
    data[:, 0] = (data[:, 0] - geo_area_ENU[0, 0]) / (geo_area_ENU[1, 0] - geo_area_ENU[0, 0])  # lon
    data[:, 1] = (data[:, 1] - geo_area_ENU[0, 1]) / (geo_area_ENU[1, 1] - geo_area_ENU[0, 1])  # lat
    data[:, 2] = data[:, 2] / 360.0
    data[:, 3] = data[:, 3] / sog_max
    data[:, 4] = data[:, 4]/cluster_max
    return data

def scale_lon_lat_cog_sog(data):
    data[:, 0] = (data[:, 0] - geo_area_ENU[0, 0]) / (geo_area_ENU[1, 0] - geo_area_ENU[0, 0])  # lon
    data[:, 1] = (data[:, 1] - geo_area_ENU[0, 1]) / (geo_area_ENU[1, 1] - geo_area_ENU[0, 1])  # lat
    data[:, 2] = data[:, 2] / 360.0
    data[:, 3] = data[:, 3] / sog_max
    return data

def scale_lon(lon):
    return (lon - geo_area_ENU[0, 0]) / (geo_area_ENU[1, 0] - geo_area_ENU[0, 0])  # lon


def scale_lat(lat):
    return (lat - geo_area_ENU[0, 1]) / (geo_area_ENU[1, 1] - geo_area_ENU[0, 1])  # lat


def scale_cog(cog):
    return cog/360.0


def scale_cluster(cluster):
    return cluster/cluster_max


def scale_sog(sog):
    return sog/cluster_max


def inverse_transform_lon_lat_cog_sog_cluster(data):
    data[:, 0] = (data[:, 0]) * (geo_area_ENU[1, 0] - geo_area_ENU[0, 0]) + geo_area_ENU[0, 0]
    data[:, 1] = (data[:, 1]) * (geo_area_ENU[1, 1] - geo_area_ENU[0, 1]) + geo_area_ENU[0, 1]
    data[:, 2] = 360*data[:,2]
    data[:, 3] = sog_max*data[:, 3]
    data[:, 4] = cluster_max*data[:, 4]
    return data

def inverse_transform_lon_lat_cog_sog(data):
    data[:, 0] = (data[:, 0]) * (geo_area_ENU[1, 0] - geo_area_ENU[0, 0]) + geo_area_ENU[0, 0]
    data[:, 1] = (data[:, 1]) * (geo_area_ENU[1, 1] - geo_area_ENU[0, 1]) + geo_area_ENU[0, 1]
    data[:, 2] = 360*data[:,2]
    data[:, 3] = sog_max*data[:, 3]
    return data


def inverse_transform_lon_lat(data):
    data[:, 0] = (data[:, 0]) * (geo_area_ENU[1, 0] - geo_area_ENU[0, 0]) + geo_area_ENU[0, 0]
    data[:, 1] = (data[:, 1]) * (geo_area_ENU[1, 1] - geo_area_ENU[0, 1]) + geo_area_ENU[0, 1]
    return data


def inverse_transform_lon(lon):
    return (lon) * (geo_area_ENU[1, 0] - geo_area_ENU[0, 0]) + geo_area_ENU[0, 0]


def inverse_transform_lat(lat):
    return (lat) * (geo_area_ENU[1, 1] - geo_area_ENU[0, 1]) + geo_area_ENU[0, 1]


def inverse_transform_cog(cog):
    return  360.0*(cog)


def inverse_transform_sog(sog):
    return sog_max* (sog)


def inverse_transform_cluster(cluster):
    return cluster_max * cluster


def reshape_data(x_train, INPUT_LEN, dim):
    # samples_train = x_train.shape[0]
    # x_train = x_train[:samples_train][:]
    print('size = ', x_train.shape[0])
    x_train.shape = (x_train.shape[0], INPUT_LEN, dim)
    return x_train