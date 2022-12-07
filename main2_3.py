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

# print((capacities.shape)) (16, 1)
# print((delta_set.shape)) (120, 16)
# print((f_set.shape)) (50, 120)


# cost under different numbers of wds
cost_alg_n = []
cost_bs0_n = []
cost_bs1_n = []
cost_opt_n = []
cost_gcg_n = []

time_alg_n = []
time_bs0_n = []
time_bs1_n = []
time_opt_n = []
time_gcg_n = []
time_trn_n = []


for num_wd in num_wd_ot:
    str_cost = "para21/costs_"+str(num_wd) + ".npy"
    costs_opt_num_wd = np.load(str_cost)
    str_time = "para21/times_" + str(num_wd) + ".npy"
    times_opt_num_wd = np.load(str_time)

    num_sv = len(capacities)      # number of Edge servers
    num_time_slots = 80   # total number of time slots
    dataset_len = 100   # size of the dataset for DNN-2
    batch_size = 50     # size of each training batch

    capacities_sv = copy.deepcopy(capacities)
    # delta = 0.5 + 0.5*np.random.rand(num_wd, num_sv).astype(np.float32)
    delta = delta_set[:num_wd, :]

    dataset0 = my_lib.MyDataset(set_length=dataset_len, num_users=num_wd, num_devices=num_sv)
    dataset1 = my_lib.MyDataset(set_length=dataset_len, num_users=num_wd, num_devices=num_sv)

    learning_rate = 0.05
    model0 = my_lib.DNN2(num_wd, num_sv).to(device)  # model using proposed approximation algorithm
    optimizer0 = torch.optim.SGD(model0.parameters(), lr=learning_rate, momentum=0.9)

    loss_alg0_ot = []
    loss_alg1_ot = []

    flag = 0  # flag indicate convergence
    # decision_gcg_pre = torch.zeros(num_wd, num_sv)

    for epoch in range(num_time_slots):
        # beginning of each time slot
        # observe states
        task_sizes = 0.6 + 1.9 * np.random.rand(num_wd).astype(np.float32)  # current system states

        # gcg solver
        decision_gcg, cost_gcg, _ = my_lib.gcg_pro2(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))

        # inference
        decision_hat_0 = model0(torch.from_numpy(task_sizes).reshape(1, num_wd).to(device)).reshape(num_wd, num_sv)
        index0 = torch.argmax(decision_hat_0, dim=1)

        decision_alg0 = torch.zeros(num_wd, num_sv)
        index0 = index0.cpu().detach().numpy()  # offloading decisions of each user
        for i in range(num_wd):
            decision_alg0[i, index0[i]] = 1
        cost_alg = my_lib.obj_val(decision_alg0, delta, capacities_sv, task_sizes)

        # get sample
        x_item, y_item = torch.from_numpy(task_sizes).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)

        # learning process model0
        if epoch == 0:
            loss0 = my_lib.cross_entropy_loss(decision_hat_0, y_item.to(device))
        else:
            if epoch < len(dataset0):
                lst1 = list(range(epoch))
                train_set = torch.utils.data.Subset(dataset0, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
            # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss0 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss0.backward()
            optimizer0.step()
            optimizer0.zero_grad()

        # update dataset
        dataset0.replace_item(x_item, y_item)

        if epoch % 20 == 0:
            print(f'\tLEARNING [{int((num_wd-120)/20)}]--[{int(100*(epoch + 1)/num_time_slots)}%], alg/gcg: {cost_alg.item()/cost_gcg:.6f}, Loss: {loss0.item()}')

    cost_alg_ot = []
    cost_bs0_ot = []
    cost_bs1_ot = []
    cost_gcg_ot = []
    cost_opt_ot = []

    time_alg_ot = []
    time_gcg_ot = []
    time_opt_ot = []
    time_bs0_ot = []
    time_bs1_ot = []
    time_trn_ot = []

    test_len = 20
    for epoch in range(test_len):
        task_sizes = f_set[epoch, :num_wd]
        # gcg solver
        t_a = time.time()
        decision_gcg, cost_gcg, _ = my_lib.gcg_pro2(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))
        t_b = time.time() - t_a
        time_gcg_ot.append(t_b)
        cost_gcg_ot.append(cost_gcg)

        # # optimal solver
        cost_opt = costs_opt_num_wd[epoch]
        time_opt = times_opt_num_wd[epoch]
        time_opt_ot.append(time_opt)
        cost_opt_ot.append(cost_opt)

        # baseline0
        t_a = time.time()
        decision_bs0, cost_bs0 = my_lib.heuristic_pro2(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))
        t_b = time.time() - t_a
        time_bs0_ot.append(t_b)
        cost_bs0_ot.append(cost_bs0)

        # baseline1
        t_a = time.time()
        decision_bs1, _, cost_bs1 = my_lib.mcmc_pro2(delta, capacities_sv.reshape(-1), task_sizes.reshape(-1))
        t_b = time.time() - t_a
        time_bs1_ot.append(t_b)
        cost_bs1_ot.append(cost_bs1)

        # inference
        t_a = time.time()
        decision_hat_0 = model0(torch.from_numpy(task_sizes.astype(np.float32)).reshape(1, num_wd).to(device)).reshape(num_wd, num_sv)
        index0 = torch.argmax(decision_hat_0, dim=1)
        t_b = time.time() - t_a
        time_alg_ot.append(t_b)
        index0 = index0.cpu().detach().numpy()  # offloading decisions of each user
        # initialize decision of our alg
        decision_alg0 = torch.zeros(num_wd, num_sv)
        for i in range(num_wd):
            decision_alg0[i, index0[i]] = 1
        cost_alg = my_lib.obj_val(decision_alg0, delta, capacities_sv, task_sizes)
        cost_alg_ot.append(cost_alg)

        if epoch % 10 == 0:
            print(f'TESTING [{int((num_wd-120)/20)}]--[{epoch + 1}/{test_len}], alg/opt: {cost_alg.item()/cost_opt:.6f}')

        # learning process model0
        t_a = time.time()
        dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
        dataiter = iter(dataloader)
        d_in_i, d_out_i = dataiter.next()
        d_in_i, d_out_i = d_in_i.to(device), d_out_i.to(device)
        outputs = model0(d_in_i)
        loss0 = my_lib.cross_entropy_loss(outputs, d_out_i)
        loss0.backward()
        optimizer0.step()
        optimizer0.zero_grad()
        t_b = time.time() - t_a
        time_trn_ot.append(t_b)

        # get sample
        x_item, y_item = torch.from_numpy(task_sizes).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)
        # update dataset
        dataset0.replace_item(x_item, y_item)

    cost_bs0_n.append(np.array(cost_bs0_ot).mean())
    cost_bs1_n.append(np.array(cost_bs1_ot).mean())
    cost_alg_n.append(np.array(cost_alg_ot).mean())
    cost_opt_n.append(np.array(cost_opt_ot).mean())
    cost_gcg_n.append(np.array(cost_gcg_ot).mean())

    time_bs0_n.append(np.array(time_bs0_ot).mean())
    time_bs1_n.append(np.array(time_bs1_ot).mean())
    time_alg_n.append(np.array(time_alg_ot).mean())
    time_opt_n.append(np.array(time_opt_ot).mean())
    time_gcg_n.append(np.array(time_gcg_ot).mean())
    time_trn_n.append(np.array(time_trn_ot).mean())

num_wd_ot = np.array(num_wd_ot)
cost_bs0_n = np.array(cost_bs0_n)
cost_bs1_n = np.array(cost_bs1_n)
cost_alg_n = np.array(cost_alg_n)
cost_opt_n = np.array(cost_opt_n)

time_bs0_n = np.array(time_bs0_n)
time_bs1_n = np.array(time_bs1_n)
time_alg_n = np.array(time_alg_n)
time_opt_n = np.array(time_opt_n)
time_gcg_n = np.array(time_gcg_n)
time_trn_n = np.array(time_trn_n)

print(f'gcg {time_gcg_n}\ntrn {time_trn_n}\nalg {time_alg_n}')

fig, ax1 = plt.subplots()
l1, = ax1.plot(num_wd_ot, cost_bs0_n/num_wd_ot, color="red", marker="*", linestyle='-')
l2, = ax1.plot(num_wd_ot, cost_bs1_n/num_wd_ot, color="blue",  marker="o", linestyle='-')
l3, = ax1.plot(num_wd_ot, cost_alg_n/num_wd_ot, color="black", marker="d", linestyle='-')
l4, = ax1.plot(num_wd_ot, cost_opt_n/num_wd_ot, color="firebrick", marker=".", linestyle='-')
l5, = ax1.plot(num_wd_ot, cost_gcg_n/num_wd_ot, color="silver", marker=".", linestyle='--')
ax1.set_ylabel('Processing Latency', fontsize=15)
ax1.set_xlabel('Number of WDs', fontsize=12)
plt.legend([l1, l2, l3, l4], ["baseline 0", "baseline 1", "alg", "opt"], loc=0)
plt.grid()
# plt.savefig("./plots/fig6_120_200.pdf", format="pdf", bbox_inches='tight')
plt.show()

