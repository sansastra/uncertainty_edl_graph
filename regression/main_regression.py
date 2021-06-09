# -*- coding: utf-8 -*-
# @Time    : 19.04.21 10:24
# @Author  : sing_sd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for removing unnecessary warnings
from absl import logging

logging._warn_preinit_stderr = 0
logging.warning('...')
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
import evidential_deep_learning as edl
import numpy as np
import matplotlib.pyplot as plt
import src.regression.load_data_regress as ld
import src.regression.process as  process
from src.regression.models import get_model

vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "resources/")
results_dir = os.path.join(vb_dir, "results/")

INPUT_LEN = 10 # same as timesteps
TARGET_LEN = 1
evidential = False
graph = True
dataset = "1" # 1-> jan 2016, 2-> july 2019, 3=> Jan 2020, 4=> rostock_degsar
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)
plt.rcParams.update({'font.size': 12})
if graph:
    features = ['x', 'y', 'cog', 'sog', "cluster"]
else:
    features = ['x', 'y', 'cog', 'sog']

dim = len(features)


def main():
    # input_data, target_data = ld.load_data_rostock_gedsar(dim, INPUT_LEN, TARGET_LEN)
    if dataset in ["1", "3", "4"]:
        input_data, target_data = ld.load_data(features, INPUT_LEN, TARGET_LEN, graph, dataset)
    else:
        input_data, target_data = ld.load_dataset_2019(dim, INPUT_LEN, TARGET_LEN, graph)


    x_train, x_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.10)

    x_train.shape = (x_train.shape[0], INPUT_LEN, dim)
    x_test.shape = (x_test.shape[0], INPUT_LEN, dim)
    y_train.shape = (y_train.shape[0], TARGET_LEN, dim)
    y_test.shape = (y_test.shape[0], TARGET_LEN, dim)

    # model
    model = get_model(INPUT_LEN, dim, TARGET_LEN, evidential)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # fit
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=200, callbacks=[callback]) #500 overfits

    plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(history.history["loss"], "r-", label="training")
    plt.plot(history.history["val_loss"], "b--", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    ax = plt.gca()
    ax.grid(True)
    plt.pause(0.001)
    # plt.savefig(results_dir + "train_val_loss_graph_"+str(graph)+"_ds"+dataset+".png")

    # model.summary()
    model.save(data_dir + "seq_graph_"+str(graph)+"_ds"+dataset+".h5")
    # Predict and plot using the trained model
    y_pred = model(x_test)

    if evidential:
        plot_predictions(x_train, y_train, x_test, y_test, y_pred, dim)

def plot_pred_no_evidential(y_test, y_pred):
    y_test = y_test.reshape(y_test.__len__(), TARGET_LEN * dim)
    y_test[:, 0] = process.inverse_transform_lon(y_test[:, 0])
    y_test[:, 1] = process.inverse_transform_lat(y_test[:, 1])
    y_pred.shape = (y_pred.__len__(), TARGET_LEN * dim)
    y_pred[:, 0] = process.inverse_transform_lon(y_pred[:, 0])
    y_pred[:, 1] = process.inverse_transform_lat(y_pred[:, 1])
    plt.plot(y_test[:, 0], y_test[:, 1], label="true")
    plt.plot(y_pred[:, 0], y_pred[:, 1], label="predicted")
    plt.pause(0.0001)
    plt.show()

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, dim, n_stds=4):
    x_train = x_train.reshape(x_train.__len__(), INPUT_LEN * dim )
    y_train = y_train.reshape(y_train.__len__(), TARGET_LEN * dim )
    x_test = x_test.reshape(x_test.__len__(), INPUT_LEN * dim )
    y_test = y_test.reshape(y_test.__len__(), TARGET_LEN * dim)
    y_test[:, 0] = process.inverse_transform_lon(y_test[:, 0])
    y_test[:, 1] = process.inverse_transform_lat(y_test[:, 1])

    y_pred = tf.reshape(y_pred, shape=(y_pred.__len__(), TARGET_LEN * dim * n_stds)) #

    var_idx = 1 #TARGET_LEN
    feature_idx = 0 # 0=> lon, 1 => lat, 2 => cog, 3 => sog

    mu_lon, v_lon, alpha_lon, beta_lon = tf.split(y_pred[:, (var_idx-1)*n_stds*dim+feature_idx: var_idx*n_stds*dim:dim], n_stds, axis=-1)
    mu_lon = mu_lon[:, 0]
    var_lon = np.sqrt(beta_lon / (v_lon * (alpha_lon - 1)))
    var_lon = np.minimum(var_lon, 1e3)[:, 0]  # for visualization

    feature_idx = 1  # 0=> lon, 1 => lat, 2 => cog, 3 => sog

    mu_lat, v_lat, alpha_lat, beta_lat = tf.split(y_pred[:, (var_idx-1)*n_stds*dim+feature_idx: var_idx*n_stds*dim:dim], n_stds, axis=-1)
    mu_lat = mu_lat[:, 0]
    var_lat = np.sqrt(beta_lat / (v_lat * (alpha_lat - 1)))
    var_lat = np.minimum(var_lat, 1e3)[:, 0]  # for visualization

    mu = np.zeros(shape=(len(y_pred),TARGET_LEN*dim))
    for i in range(TARGET_LEN):
        mu[:,i*dim:(i+1)*dim] = y_pred[:,i*dim*n_stds:i*dim*n_stds+dim]
    # scaling:
    mu[:, 0] = process.inverse_transform_lon(mu[:, 0])
    mu[:, 1] = process.inverse_transform_lat(mu[:, 1])

    plt.figure(figsize=(5, 3), dpi=200)
    # plt.scatter(x_train[:, 0], y_train[:, 1], s=1., c='#463c3c', zorder=0, label="Train_lon")

    plt.pause(0.001)
    plt.scatter(y_test[:, 0], y_test[:, 1], s=1., color='#163a3b',  zorder=0, label="True")
    plt.scatter(mu[:, 0], mu[:,1], s=1., color='#007cab', zorder=0, label="Pred")

    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            y_test[:, 0], (mu[:,1] - k * var_lat), (mu[:,1] + k * var_lat),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    # plt.gca().set_ylim(-10, 10)
    # plt.gca().set_xlim(-7, 7)
    plt.xlabel("Longitude [m]")
    plt.ylabel("Latitude [m]")
    plt.legend()
    plt.pause(0.001)
    # plt.savefig(results_dir + "unc_graph_all"+str(TARGET_LEN)+"bigdata.png")
    plt.show()


if __name__ == "__main__":
    main()