import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import my_lib
import math
import random
from matplotlib import pyplot as plt
from scipy.io import savemat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# system setting
# num_wd_ot = [80, 90, 100, 110, 120]     # number of Wireless devices
num_wd_ot = [120, 140, 160, 180, 200]

capacities = np.load("para21/capacities.npy")
delta_set = np.load("para21/delta_set.npy")
f_set = np.load("para21/f_set.npy")

for num_wd in num_wd_ot:
    str_cost = "para21/costs_"+str(num_wd) + ".npy"
    costs_opt_num_wd = np.load(str_cost)
    str_time = "para21/times_" + str(num_wd) + ".npy"
    times_opt_num_wd = np.load(str_time)

    num_sv = len(capacities)      # number of Edge servers
    num_time_slots = 1   # total number of time slots
    dataset_len = 100   # size of the dataset for DNN-2
    batch_size = 50     # size of each training batch

    capacities_sv = copy.deepcopy(capacities)
    # delta = 0.5 + 0.5*np.random.rand(num_wd, num_sv).astype(np.float32)
    delta = delta_set[:num_wd, :]

    for epoch in range(num_time_slots):
        # beginning of each time slot
        # observe states
        task_sizes = 0.6 + 1.9 * np.random.rand(num_wd).astype(np.float32)  # current system states

        ta = time.time()
        decision_0, cost_0, _ = my_lib.gcg_pro2(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))
        t1 = time.time() - ta
        ta = time.time()
        decision_1, cost_1, _ = my_lib.gcg_pro2_new(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))
        t2 = time.time() - ta
        print(f'difference={np.abs(cost_1 - cost_0):.6f}\t t_dif={t1 - t2:.6f}\t t_rat={(t1 - t2)/t1:.6f}')
        if np.abs(cost_1 - cost_0)>0.001:
            print("error")
