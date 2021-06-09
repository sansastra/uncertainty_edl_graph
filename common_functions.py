# -*- coding: utf-8 -*-
# @Time    : 15.04.21 12:18
# @Author  : sing_sd

import numpy as np

def get_filtered_data_idx(data, WINDOW, SOG_LIMIT, NAV_STATUS, SHIP_TYPES):
    idx = np.array(data['x']) > WINDOW[0]
    idx &= np.array(data['y']) > WINDOW[1]
    idx &= np.array(data['x']) < WINDOW[2]
    idx &= np.array(data['y']) < WINDOW[3]
    idx &= (np.array(data["sog"]) > SOG_LIMIT[0]) & (np.array(data["sog"]) <= SOG_LIMIT[1])
    idx &= np.array(data["nav_status"]) == NAV_STATUS
    idx &= (np.array(data["ship_type"]) >= SHIP_TYPES[0]) & (np.array(data["ship_type"]) < SHIP_TYPES[1])
    return idx
