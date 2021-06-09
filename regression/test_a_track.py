import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for removing unnecessary warnings
from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('...')
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import interpolate
from tensorflow import keras
import evidential_deep_learning as edl
import time
import random
import matplotlib.pyplot as plt
import joblib
import pathlib
# from PythonCode.KF.ENUtransform import ENUtoWGS84
import src.clustering.COStransforms as ct

from src.regression.process import  scale_lon_lat_cog_sog, scale_lon_lat_cog_sog_cluster,inverse_transform_lon, inverse_transform_lat, inverse_transform_cog, inverse_transform_sog
from src.regression.load_data_regress import load_test_data, load_test_interpolated_data, load_test_data_ENU, load_test_data_gedsar, load_test_data_vk2mmsi
########### check a whole track ##############
from src.evidential_deep_learning.layers.dense import DenseNormalGamma
from src.evidential_deep_learning.losses.continuous import EvidentialRegression

# edl.losses.EvidentialRegression does work, since this custom loss function has a get_config method
# that has to be changed in the argument, add **kwarg
# Also DenseNormalGamma is imported locally otherwise saved model does not recognise this layer

vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "resources/")
results_dir = os.path.join(vb_dir, "results/")

ANO_DATA = False
cluster = True
dataset = "1"  # 1-> jan 2016, 2-> july 2019, 3=> Jan 2020, 4=> rostock_degsar

if cluster:
    features = ['x', 'y', 'cog', 'sog', 'cluster'] # by default we convert to ENU
    model_name = "edl_graph_True_ds"+dataset+".h5"
else:
    features = ['x', 'y', 'cog', 'sog']
    model_name = "edl_graph_False_ds"+dataset+".h5"

dim = len(features)
INPUT_LEN = 10  # same as timesteps
TARGET_LEN = 1
track_to_check = 0, #167  # 43, 167, 202

# path = "/home/sing_sd/Desktop/evidential/src/regression/resources/"
# model = joblib.load(model_name)
# custom_objects = {'DenseNormalGamma':edl.layers.DenseNormalGamma}
# with keras.utils.custom_object_scope(custom_objects):
#   model=keras.models.load_model(path+model_name)
def EvidentialRegressionLoss(true, pred):
    return EvidentialRegression(true, pred, coeff=1e-2)

# layer_config = layer.get_config()
# new_layer = keras.layers.Dense.from_config(layer_config)

model = keras.models.load_model(data_dir+model_name, custom_objects={'DenseNormalGamma': DenseNormalGamma, "EvidentialRegressionLoss": EvidentialRegressionLoss})
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(6, 8))


def main():
    fig1, ax1 = plt.subplots(4, figsize=(6, 8))
    # plot_uncertainties_rg_left_right_center()

    # original_data, test_data, test_data_interpolated = load_test_data(INPUT_LEN, TARGET_LEN, features, dim, track_to_check)
    original_data, test_data, test_data_interpolated = load_test_data_gedsar(INPUT_LEN, TARGET_LEN, features, dim, ANO_DATA, cluster)
    # original_data, test_data, test_data_interpolated = load_test_data_vk2mmsi(INPUT_LEN, TARGET_LEN, features, dim, ANO_DATA)
    ax.plot(original_data[:INPUT_LEN, 0], original_data[:INPUT_LEN, 1], '.k', label='Original trajectory')
    plt.pause(0.001)
    test_predict = 0
    test_predict_all = np.zeros(shape=(len(test_data)-INPUT_LEN, dim))
    uncertainty_all = np.zeros(shape=(len(test_data)-INPUT_LEN, dim))
    total_distance_error = []
    predicted_track = pd.DataFrame(columns=['x', 'y'])
    for data_index in range(len(test_data)-INPUT_LEN):

        if np.isnan(test_data[data_index, (INPUT_LEN-1)*dim]):
            for i in range(INPUT_LEN):
                if data_index+i < len(test_data):
                    test_data[data_index + i, (INPUT_LEN-1-i)*dim : (INPUT_LEN-i)*dim] = test_predict[0, 0: dim]


        X_test = test_data[data_index, :].reshape(1, INPUT_LEN, dim)
        if not cluster:
            X_test[0] = scale_lon_lat_cog_sog(X_test[0])
        else:
            X_test[0, :, 0: dim] = scale_lon_lat_cog_sog_cluster(X_test[0, :, 0:dim])
        # start_time = time.time()
        test_predict = model.predict(X_test)
        # print(time.time()-start_time)
        # invert predictions
        y_pred = tf.reshape(test_predict, shape=(test_predict.__len__(), TARGET_LEN * dim * 4))  # n_stds=4

        uncertianty = get_uncertainty(y_pred) # sigma, alpha, v
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        plot_v_alpha_beta(ax1,data_index,v,alpha, beta)

        test_predict = np.array(y_pred[:, 0:dim])
        test_predict[0, 0] = inverse_transform_lon(test_predict[0, 0])
        test_predict[0, 1] = inverse_transform_lat(test_predict[0, 1])
        test_predict.shape = (1, dim)

        # test_predict[test_predict <= 0] = np.nan

        # store values to fill var
        test_predict_all[data_index, 0] = test_predict[0, 0]
        test_predict_all[data_index, 1] = test_predict[0, 1]
        uncertainty_all[data_index, 0] = uncertianty[0, 0]
        uncertainty_all[data_index, 1] = uncertianty[0, 1]
        uncertainty_all[data_index, 2] = uncertianty[0, 2]
        uncertainty_all[data_index, 3] = uncertianty[0, 3]

        ax.plot(test_predict[0, 0], test_predict[0, 1], 'xb', label="Predicted trajectory" if data_index == 0 else None)
        ax.plot(original_data[INPUT_LEN+data_index, 0], original_data[INPUT_LEN+data_index, 1], '.k') # original track
        # plt.pause(0.001)

        # plt.gca().set_ylim(54, 55)
        # plt.gca().set_xlim(11.5, 12.5)
        predicted_track.loc[data_index] = test_predict[0,0:2]
        distance_error = np.abs(np.sqrt(
        (test_data_interpolated[INPUT_LEN+data_index, 0] - test_predict[0, 0]) ** 2 + (
                    test_data_interpolated[INPUT_LEN+data_index, 1] - test_predict[0, 1]) ** 2)*1.85*60 )
        total_distance_error.append(distance_error)


    exit(0)
    total_distance_error = np.array(total_distance_error)
    # plot_anomalous_segments(uncertainty_all, original_data[:,0:2], total_distance_error)
    print("max ", np.max(total_distance_error))
    print("min ", np.min(total_distance_error))
    print("avg ", np.mean(total_distance_error))
   # np.savetxt("./resources/pred_rg_data_left_graph_" + str(cluster) + "_ds" + dataset + ".csv", test_predict_all,
   #            delimiter=",")

    # np.savetxt("./resources/unc_rg_data_normal_graph_"+str(cluster)+"_ds"+dataset+".csv", uncertainty_all, delimiter=",")

    # kk = 1
    # plt.fill_between(original_data[INPUT_LEN:, 0], test_predict_all[:, 1] - kk * test_predict_all[:, 3],
    #                  test_predict_all[:, 1] + kk * test_predict_all[:, 3],
    #                  alpha=0.3,
    #                  edgecolor=None,
    #                  facecolor='#00aeef',
    #                  linewidth=0,
    #                  zorder=1,
    #                  label="Unc.")

    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    ax.legend()
    plt.pause(0.001)
    # plt.savefig('gedsar_route_prediction.png')
    # plt.savefig('gedsar_route_prediction.pdf')
    plot_uncertainties(uncertainty_all)


def plot_v_alpha_beta(ax1,data_index,v,alpha, beta):
    v = v[0]
    alpha = alpha[0]
    beta = beta[0]
    ax1[0].plot(data_index, v[0], "r.", label="v" )
    ax1[0].plot(data_index, alpha[0], "g.", label="alpha")
    ax1[0].plot(data_index, beta[0], "b.", label="beta")
    ax1[1].plot(data_index, v[1], "r.", label="v")
    ax1[1].plot(data_index, alpha[1], "g.", label="alpha")
    ax1[1].plot(data_index, beta[1], "b.", label="beta")
    ax1[2].plot(data_index, v[2], "r.", label="v")
    ax1[2].plot(data_index, alpha[2], "g.", label="alpha")
    ax1[2].plot(data_index, beta[2], "b.", label="beta")
    ax1[3].plot(data_index, v[3], "r.", label="v")
    ax1[3].plot(data_index, alpha[3], "g.", label="alpha")
    ax1[3].plot(data_index, beta[3], "b.", label="beta")

    plt.pause(0.0001)
    if data_index == 0:
        ax1[0].set_ylabel("Lon")
        ax1[1].set_ylabel("Lat")
        ax1[2].set_ylabel("cog")
        ax1[3].set_ylabel("sog")
        plt.xlabel("Time_index")
        ax1[0].set_title("evaluated for the normal track")
        ax1[0].legend(loc="upper right")
        ax1[1].legend(loc="upper right")
        ax1[2].legend(loc="upper right")
        ax1[3].legend(loc="upper right")
    plt.savefig("./results/rg_normal_v_alpha_beta.png")


def plot_uncertainties(unc):
    fig1, axs = plt.subplots(4, sharex=True, figsize=(8, 6))
    axs[0].plot(unc[:, 0], "r-", label="Longitude")
    axs[1].plot(unc[:, 1], "g--", label="Latitude")
    axs[1].set_ylabel('Uncertainty')
    axs[2].plot(unc[:, 2], "b-", label="CoG")
    axs[3].plot(unc[:, 3], "k--", label="SoG")

    plt.xlabel('Time index')
    axs[0].set_title("Uncertainty in the Left Anomalous Track")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper right")
    axs[3].legend(loc="upper right")
    plt.pause(0.001)

    # plt.savefig('Unc_gedsar_ano_route_graph_right_all.png')
    # plt.savefig('Unc_gedsar_ano_route_graph_right_all.pdf')
    plt.show()


def plot_anomalous_segments(uncertainty_all, lon_lat_plot, distance_error):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches([8, 6])
    plt.pause(0.0001)

    segment_size = 10
    unc_thr = 0.005
    pred_dist_thr = 10
    ax.plot(lon_lat_plot[:, 0], lon_lat_plot[:, 1], c="k", linestyle="--", linewidth=0.5)
    plt.pause(0.0001)
    step = 10
    # color = "#F" + ''.join([random.choice('0123456789ABCDEF') for j in range(5)])
    for index in range(0, len(uncertainty_all) - segment_size, step):
        # std = np.std(uncertainty[index:index+segment_size, :], axis=0)
        max = np.max(uncertainty_all[index:index + segment_size, :], axis=0)
        min = np.min(uncertainty_all[index:index + segment_size, :], axis=0)
        if np.max(1 - np.divide(min[0:2], max[ 0:2])) > unc_thr:
            # and np.max(distance_error[index:index + segment_size]) > pred_dist_thr:
            ax.scatter(lon_lat_plot[index: index + segment_size, 0], lon_lat_plot[index: index + segment_size, 1],
                       marker=".", c="r", s=15, zorder=5)

    # if np.max(std[0:4]) > unc_thr or np.max(distance_error[index:index+segment_size]) > pred_dist_thr:
        #     ax.scatter(lon_lat_plot[index: index+segment_size, 0], lon_lat_plot[index: index+segment_size, 1], marker = "+", c="r", s=15)
    plt.pause(0.0001)


def get_var(y_pred):
    n_stds = 4

    feature_idx = 0  # 0=> lon, 1 => lat, 2 => cog, 3 => sog

    mu_lon, v_lon, alpha_lon, beta_lon = tf.split(y_pred[:, feature_idx: n_stds * dim:dim], n_stds, axis=-1)
    var_lon = np.sqrt(beta_lon / (v_lon * (alpha_lon - 1)))
    # var_lon = np.minimum(var_lon, 1e3)[:, 0]  # for visualization

    feature_idx = 1  # 0=> lon, 1 => lat, 2 => cog, 3 => sog

    mu_lat, v_lat, alpha_lat, beta_lat = tf.split(y_pred[:, feature_idx: n_stds * dim:dim], n_stds, axis=-1)
    mu_lat = mu_lat[:, 0]
    var_lat = np.sqrt(beta_lat / (v_lat * (alpha_lat - 1)))
    # var_lat = np.minimum(var_lat, 1e3)[:, 0]  # for visualization

    return var_lon, var_lat


def get_uncertainty(y_pred):
    n_stds = 4
    uncertainty = np.zeros(shape=(1, dim))
    for feature_idx in range(dim): # 0=> lon, 1 => lat, 2 => cog, 3 => sog
        mu, v, alpha, beta = tf.split(y_pred[:, feature_idx: n_stds * dim:dim], n_stds, axis=-1)
        var_model = np.sqrt(beta / (v * (alpha - 1)))
        var_data = np.sqrt(beta / (alpha - 1))
        var_model = np.minimum(var_model, 1e2)
        uncertainty[0, feature_idx] = var_model[0, 0]
    return uncertainty


def plot_predicted_track(original_data, test_predict, INPUT_LEN, TARGET_LEN, dim):
    ax.plot(original_data[:, 0], original_data[:, 1],'.k', label='Original trajectory')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    ax.legend()
    plt.savefig('trajectory_pred1.pdf')
    plt.pause(0.001)

if __name__ == "__main__":
    main()

################################################################################