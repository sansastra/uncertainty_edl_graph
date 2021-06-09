# -*- coding: utf-8 -*-
# @Time    : 09.04.21 09:54
# @Author  : sing_sd

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.common_functions as cf
import csv
import ais
from datetime import datetime, timedelta, timezone
import re

vb_dir   = os.path.dirname(__file__)
data_dir = os.path.join(vb_dir, "resources/")
headers = ['x', 'y', 'cog', 'sog', 'time', 'mmsi', "nav_status", "ship_type", "destination"]
plt.rcParams.update({'font.size': 12})
def main():
    # data = generate_short_data(data_len=100000)
    filename = 'ais_data_rostock_2020.csv' # 'ais_data_rostock_2016.csv'
    generate_processed_data(filename)
    # filename = "ais_data_rostock_2016_processed.csv"
    # plot_data(filename)
    # generate_rostock_gedsar_dataset(filename)
    # decode_data()

def plot_data(filename):
    mpl.rcParams['agg.path.chunksize'] = 10000
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([8, 6])
    plt.pause(0.0001)
    with open(data_dir + filename, "r") as f:
        print("start")
        data = pd.read_csv(f)
        print("data loaded")
        data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                            'nav_status': 'float', 'ship_type': 'float', 'destination': 'str'})
        WINDOW = [11.5, 54.2, 12.5, 54.5]
        SOG_LIMIT = [0, 30]
        NAV_STATUS = 0.0
        SHIP_TYPES = [0, 90]
        idx = cf.get_filtered_data_idx(data, WINDOW, SOG_LIMIT, NAV_STATUS, SHIP_TYPES)
        data = data.iloc[idx, :]
        data = data.reset_index(drop=True)
        print("Data loaded, total messages within window = ", len(data))


#        axs.plot(data.iloc[:, 0], data.iloc[:, 1], 'b.', markersize=0.1, alpha=0.5) #, linestyle="solid", linewidth=0.1, alpha=0.5

        SAMPLING_TIME = 10

        for mmsi in data.mmsi.unique():
            data_mmsi = data.iloc[np.array(data['mmsi'] == mmsi), :]
            data_mmsi = data_mmsi.reset_index(drop=True)
            nr_data = int(np.ceil((np.array(data_mmsi.iloc[-1]['time']) - data_mmsi.iloc[0]['time']) / SAMPLING_TIME) + 1)
            overall_data = np.full(shape=(100 * len(data_mmsi), 2), fill_value=np.nan)
            data_mmsi['time_idx'] = data_mmsi.iloc[0]['time']
            data_mmsi['time_idx'] = np.ceil((data_mmsi['time'] - data_mmsi['time_idx']) / SAMPLING_TIME)
            overall_data[np.array(data_mmsi['time_idx'], dtype=np.int), 0:2] = np.array(data_mmsi[['x', 'y']])

            axs.plot(overall_data[:, 0], overall_data[:, 1],
                        linestyle="-", color="blue", linewidth=0.3, alpha=5)
            plt.pause(0.0001)
    axs.set_xlabel('Longitude [deg]')
    axs.set_ylabel('Latitude [deg]')
    axs.set_xlim(xmin=11.5, xmax=12.5)
    axs.set_ylim(ymin=54.2, ymax=54.5)
    plt.pause(0.001)
    plt.savefig("./resources/dataset2016.png")
    plt.savefig("./resources/dataset2016.pdf")
    plt.show()


def generate_processed_data(filename):
    with open(data_dir + filename, "r") as f:
        print("start")
        data_pd = pd.read_csv(f)
        print("data loaded")
        data_pd = data_pd.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                            'nav_status': 'float', 'ship_type': 'str', 'destination': 'str'})
        data = np.array(data_pd)
        for mmsi in data_pd.mmsi.unique():
            idx = data[:, 5] == mmsi
            # update the missing destination dtype = str, replace nan
            dst = data[:, -1] != "nan"
            value = np.unique(data[idx & dst, -1])
            if len(value) == 0:
                # data[idx, -1] = "UNKNOWN"
                data_pd.iloc[idx, -1] = "UNKNOWN"
            else:
                # data[idx, -1] = value[0]
                data_pd.iloc[idx, -1] = value[0]

            # and ship type
            dst = data[:, -2] != "nan"
            value = np.unique(data[idx & dst, -2])
            if len(value) == 0:
                # data[idx, -2] = np.nan
                # data_pd.iloc[idx, -2] = np.nan
                data_pd.iloc[idx, 0] = np.nan # delete those rows that does not have ship type by putting x = np.nan
            else:
                # data[idx, -2] = value[0]
                data_pd.iloc[idx, -2] = value[0]
        data_pd = data_pd[data_pd.x > 0]
        if sum(data_pd.iloc[:, -1] == "nan") + sum(data_pd.iloc[:, -2] == "nan")> 0:
            print("there are nan values")
            exit(0)
        data_pd["ship_type"] = data_pd["ship_type"].astype("float64")
        data_pd.to_csv(data_dir + "ais_data_rostock_2020_processed.csv", index=False)



        # plot_graph(data)

def generate_short_data(data_len=10000):
    data = pd.DataFrame(columns=headers)
    data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                        'nav_status': 'float', 'ship_type': 'str', 'destination': 'str'})
                        # float helps in interpolation of these features
    try:
        with open(data_dir + 'ais_data_rostock_2016.csv', "r") as my_csv:
            reader = csv.reader(my_csv)
            print("first row", next(reader))
            for i in range(data_len):
                try:
                    next_row = next(reader)
                    data = data.append(pd.Series(next_row, index=data.columns), ignore_index=True)
                except Exception as e:
                    print(str(e))

                    data.to_csv(data_dir + 'ais_data_rostock_2019_short.csv', index=False)
                    exit(0)
        # data = genfromtxt(data_dir+'ais_data_rostock_2019.csv', delimiter=',')

        data.to_csv(data_dir + 'ais_data_rostock_2016_short.csv', index=False)
        # np.savetxt(data_dir+'ais_data_rostock_2019_short.csv', data, delimiter=',')

    except Exception as e:
        print(str(e))
    return data


def generate_rostock_gedsar_dataset(filename):
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches([8, 6])
    plt.pause(0.0001)
    with open(data_dir + filename, "r") as f:
        print("start")
        data = pd.read_csv(f)
        print("data loaded")
        data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                            'nav_status': 'float', 'ship_type': 'float', 'destination': 'str'})
        WINDOW = [11, 54, 13, 56]
        SOG_LIMIT = [0, 30]
        NAV_STATUS = 0.0
        SHIP_TYPES = [60, 61]
        idx = cf.get_filtered_data_idx(data, WINDOW, SOG_LIMIT, NAV_STATUS, SHIP_TYPES)
        data = data.iloc[idx, :]
        data_rg = pd.DataFrame(columns=data.columns)
        filename = "ais_data_rostock_gedsar_2016.csv"
        print("Data loaded, total messages within window = ", len(data))
        for mmsi in data.mmsi.unique():
            if mmsi in [219000479,218780000]:
                data_mmsi = data.iloc[np.array(data['mmsi'] == mmsi), :]
                data_mmsi = data_mmsi.reset_index(drop=True)
                data_rg = pd.concat([data_rg,data_mmsi], ignore_index=True)

        data_rg.to_csv(data_dir+filename, index=False)
        plt.plot(data_rg["x"], data_rg["y"])
        plt.pause(0.0001)
        plt.show()



def decode_data():
    WINDOW = (11, 54, 13, 56)
    np.random.seed(10)
    # names = [i for i in range(20)] # chnage .. when using other input files
    headers = ['x', 'y', 'cog', 'sog', 'time', 'mmsi', "nav_status", "ship_type", "destination"]
    data = pd.DataFrame(columns=headers)
    data = data.astype({'x': 'float', 'y': 'float', 'cog': 'float', 'sog': 'float', 'time': 'float', 'mmsi': 'int',
                        'nav_status': 'float', 'ship_type': 'float',
                        'destination': 'str'})  # float helps in interpolation of these features

    filename = 'ais_data_rostock_2020.csv'
    data.to_csv(filename, index=False)
    # insert a dummy row
    to_append = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    data = data.append(pd.Series(to_append, index=data.columns), ignore_index=True)

    txt_files = sorted(os.listdir(data_dir+"/AISHUB2020/"))
    for file in txt_files:
        with open(data_dir+"/AISHUB2020/"+file, "r") as f:
            aismsg = None
            for line_num, i_line in enumerate(f.readlines()):  # [:3000] f.readlines()
                try:
                    splitted_line = i_line.split('\t')
                    ais_timestamp = splitted_line[0]
                    nmea_msg_split = splitted_line[1].split(",")

                    if nmea_msg_split[1] == "2":
                        if nmea_msg_split[2] == "1":
                            multi_line_nmea = nmea_msg_split[5]
                        if nmea_msg_split[2] == "2":
                            multi_line_nmea += nmea_msg_split[5]
                            # print(multi_line_nmea)
                            aismsg = ais.decode(multi_line_nmea, 2)
                            # print(aismsg)
                            multi_line_nmea = ""
                    else:
                        aismsg = ais.decode(nmea_msg_split[5], 0)

                    if aismsg is not None or (aismsg['id'] in [1, 2, 3, 5]):  # or aismsg['id'] == 18 or aismsg['id'] == 19
                        # if aismsg["mmsi"] == 219423000: #244239000: # getting data for a single trajectory

                        if aismsg['id'] in [1, 2, 3]:
                            if not ((aismsg['x'] < WINDOW[0]) or (aismsg['y'] < WINDOW[1]) or (aismsg['x'] > WINDOW[2]) or (
                                    aismsg['y'] > WINDOW[3])):  # aismsg['sog'] < 6 or (aismsg['sog'] > 50)
                                to_append = [aismsg['x'], aismsg['y'], aismsg['cog'], aismsg['sog'], ais_timestamp,
                                             aismsg['mmsi'], aismsg["nav_status"], np.nan,
                                             np.nan]  # class_name = nmea_msg_split[4]
                                data.iloc[0] = pd.Series(to_append, index=data.columns)
                                data.to_csv(filename, mode='a', index=False, header=None)
                                # data = data.drop(0, axis=0)
                                # file.write('\n')
                                # data = data.append(a_series, ignore_index=True)
                        elif aismsg['id'] == 5:
                            to_append = [np.nan, np.nan, np.nan, np.nan, ais_timestamp, aismsg['mmsi'], np.nan,
                                         aismsg["type_and_cargo"], re.split("@| ", aismsg["destination"])[0]]
                            data.iloc[0] = pd.Series(to_append, index=data.columns)
                            # data = data.append(pd.Series(to_append, index=data.columns), ignore_index=True)
                            data.to_csv(filename, mode='a', index=False, header=None)
                            # data = data.drop(0, axis=0)
                except Exception as ex:
                    continue
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)

        # data.to_csv(filename,index=False)
        print("completed file ", file)


if __name__ == "__main__":
    main()