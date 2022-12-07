import sys

import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import my_lib
from scipy.special import softmax
from matplotlib import pyplot as plt
from scipy.io import savemat
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# system setting
num_wd = 180     # number of Wireless devices
num_ap = 10      # number of base stations (access points)
num_time_slots = 400   # total number of time slots
dataset_len = 100   # size of the dataset for DNN-2
batch_size = 30     # size of each training batch

bandwidth_uplink = 100 * np.random.rand(num_ap).astype(np.float32)
bandwidth_uplink[0] = 45
bandwidth_fronthaul = 1000 * np.random.rand(num_ap).astype(np.float32)

delta_uplink = 0.15 + 0.35 * np.random.rand(num_wd, num_ap).astype(np.float32)
delta_fronthaul = 0.1 + 0 * np.random.rand(num_wd, num_ap).astype(np.float32)

dataset0 = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)
dataset1 = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)
dataset2 = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)

learning_rate = 0.01
model0 = my_lib.DNN1(num_wd, num_ap).to(device)
optimizer0 = torch.optim.SGD(model0.parameters(), lr=learning_rate, momentum=0.5)
model1 = my_lib.DNN1(num_wd, num_ap).to(device)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.5)


# the devices that will be shut down
wd_shutdown_lst = random.sample(range(num_wd), 20)
indices = []
for idx in range(num_wd):
    if idx not in wd_shutdown_lst:
        indices.append(idx)
wd_shutdown = np.ones(num_wd).astype(np.float32)
wd_shutdown[wd_shutdown_lst] = np.zeros(len(wd_shutdown_lst)).astype(np.float32)
t_down = 200
t_start = 300

# performance metrics
cost_alg0_ot = []
cost_alg1_ot = []
loss_alg0_ot = []
loss_alg1_ot = []
cost_gcg_ot = []
time_gcg_ot = []
cost_opt_ot = []
time_opt_ot = []

for epoch in range(num_time_slots):
    # beginning of each time slot
    data_sizes = 3 + 7 * np.random.rand(num_wd).astype(np.float32)  # current system states
    delta_uplink = delta_uplink * (1 + 0.01 * np.random.randn(num_wd, num_ap))
    delta_uplink = np.maximum(np.minimum(delta_uplink, 0.5), 0.15).astype(np.float32)

    sys_state_i = np.zeros(num_wd + num_ap*num_wd).astype(np.float32)
    sys_state_i[:num_wd] = data_sizes
    sys_state_i[num_wd:] = delta_uplink.reshape(-1)

    if epoch == t_down:
        sd = model0.state_dict()
        model1.load_state_dict(sd)
    if epoch in range(t_down, t_start):
        data_sizes = data_sizes * wd_shutdown

    # gcg solver
    decision_gcg, cost_gcg, _ = my_lib.gcg_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, data_sizes)
    # optimal solver
    decision_opt, cost_opt = my_lib.gurobi_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, data_sizes)
    # initialize decision of our alg
    decision_alg0 = torch.zeros(num_wd, num_ap)
    decision_alg1 = torch.zeros(num_wd, num_ap)

    # inference
    decision_0_real = model0(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
    index0 = torch.argmax(decision_0_real, dim=1)
    index0 = index0.cpu().detach().numpy()  # offloading decisions of each user

    decision_1_real = model1(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
    index1 = torch.argmax(decision_1_real, dim=1)
    index1 = index1.cpu().detach().numpy()  # offloading decisions of each user

    # get sample
    x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)

    # learning process of model0
    if epoch < t_down:
        if epoch == 0:
            loss0 = my_lib.cross_entropy_loss(decision_0_real, y_item.to(device))
        else:
            if epoch < len(dataset0):
                lst1 = list(range(epoch))
                train_set = torch.utils.data.Subset(dataset0, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss0 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss0.backward()
            optimizer0.step()
            optimizer0.zero_grad()
    elif epoch in range(t_down, t_start):
        epoch1 = epoch - t_down
        if epoch1 == 0:
            loss0 = my_lib.cross_entropy_loss_down(decision_0_real, y_item.to(device), torch.IntTensor(indices).to(device))
        else:
            if epoch1 < len(dataset1):
                lst1 = list(range(epoch1))
                train_set = torch.utils.data.Subset(dataset1, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch1 if epoch1 < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss0 = my_lib.cross_entropy_loss_down(outputs, d_out_i, torch.IntTensor(indices).to(device))
            loss0.backward()
            optimizer0.step()
            optimizer0.zero_grad()
    elif epoch >= t_start:
        epoch2 = epoch - t_start
        if epoch == t_start:
            loss0 = my_lib.cross_entropy_loss(decision_0_real, y_item.to(device))
        else:
            if epoch2 < len(dataset2):
                lst2 = list(range(epoch2))
                train_set = torch.utils.data.Subset(dataset2, lst2)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch2 if epoch2 < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss0 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss0.backward()
            optimizer0.step()
            optimizer0.zero_grad()

    # learning process of model1
    if epoch < t_down:
        if epoch == 0:
            loss1 = my_lib.cross_entropy_loss(decision_1_real, y_item.to(device))
        else:
            if epoch < len(dataset0):
                lst1 = list(range(epoch))
                train_set = torch.utils.data.Subset(dataset0, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model1(d_in_i)
            loss1 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
    elif epoch in range(t_down, t_start):
        epoch1 = epoch - t_down
        if epoch1 == 0:
            loss1 = my_lib.cross_entropy_loss(decision_1_real, y_item.to(device))
        else:
            if epoch1 < len(dataset1):
                lst1 = list(range(epoch1))
                train_set = torch.utils.data.Subset(dataset1, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch1 if epoch1 < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model1(d_in_i)
            loss1 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
    elif epoch >= t_start:
        epoch2 = epoch - t_start
        if epoch == t_start:
            loss1 = my_lib.cross_entropy_loss(decision_1_real, y_item.to(device))
        else:
            if epoch2 < len(dataset2):
                lst2 = list(range(epoch2))
                train_set = torch.utils.data.Subset(dataset2, lst2)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch2 if epoch2 < batch_size else batch_size,
                                        shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset2, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model1(d_in_i)
            loss1 = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

    # update dataset
    if epoch in range(t_down, t_start):
        dataset1.replace_item(x_item, y_item)
    elif epoch >= t_start:
        dataset2.replace_item(x_item, y_item)
    else:
        dataset0.replace_item(x_item, y_item)

    # calculate objetive value (computing latency of our alg)
    # y_real = decision_0_real.cpu().detach().numpy()
    for i in range(num_wd):
        decision_alg0[i, index0[i]] = 1
        decision_alg1[i, index1[i]] = 1

    cost_alg0 = my_lib.obj_val(decision_alg0, delta_uplink, bandwidth_uplink, data_sizes) + my_lib.obj_val(decision_alg0, delta_fronthaul, bandwidth_fronthaul, data_sizes)
    cost_alg1 = my_lib.obj_val(decision_alg1, delta_uplink, bandwidth_uplink, data_sizes) + my_lib.obj_val(decision_alg1, delta_fronthaul, bandwidth_fronthaul, data_sizes)

    cost_alg0_ot.append(cost_alg0)
    cost_alg1_ot.append(cost_alg1)

    cost_gcg_ot.append(cost_gcg)
    cost_opt_ot.append(cost_opt)

    loss_alg0_ot.append(loss0.item())
    loss_alg1_ot.append(loss1.item())

    if epoch % 10 == 0:
        print(
            f'Epoch [{epoch}/{num_time_slots}], Loss0: {loss0.item():.6f}, Loss1: {loss1.item():.6f}, cost0: {cost_alg0.item() / cost_opt.item():.6f}, cost1: {cost_alg1.item() / cost_opt.item():.6f}')

# plot and plot performance
ratio0 = np.array(cost_alg0_ot).reshape(-1) / np.array(cost_opt_ot).reshape(-1)     # competitive ratio of our alg0
ratio1 = np.array(cost_alg1_ot).reshape(-1) / np.array(cost_opt_ot).reshape(-1)     # competitive ratio of our alg1
ratio1[:t_down] = ratio0[:t_down]
loss_alg1_ot[:t_down] = loss_alg0_ot[:t_down]

# calculating moving average competitive ratio
move_ave_size = 10  # moving average subset size
ratio0_mean = np.zeros(len(ratio0) - move_ave_size)
ratio0_min = np.zeros(len(ratio0) - move_ave_size)
ratio0_max = np.zeros(len(ratio0) - move_ave_size)
ratio1_mean = np.zeros(len(ratio1) - move_ave_size)
ratio1_min = np.zeros(len(ratio1) - move_ave_size)
ratio1_max = np.zeros(len(ratio1) - move_ave_size)

for i in range(len(ratio0) - move_ave_size):
    ratio0_mean[i] = ratio0[i:i + move_ave_size].mean()
    ratio0_max[i] = ratio0[i:i + move_ave_size].max()
    ratio0_min[i] = ratio0[i:i + move_ave_size].min()

    ratio1_mean[i] = ratio1[i:i + move_ave_size].mean()
    ratio1_max[i] = ratio1[i:i + move_ave_size].max()
    ratio1_min[i] = ratio1[i:i + move_ave_size].min()

ratio_cvg = ratio0[-100:].mean()

# plot the average approximation ratio and loss of DNN-1
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
l1, = ax1.plot(list(range(len(ratio0)))[100:], ratio0[100:len(ratio0)], color="black", linestyle='-')
l2, = ax1.plot(list(range(len(ratio1)))[100:], ratio1[100:len(ratio1)], color="grey", linestyle='--')
ax1.set_ylabel('Normalized Communication Latency', fontsize=15)
ax1.set_xlabel('Time Slots', fontsize=12)
ax1.set_xlim([99, len(ratio0)])
ax1.set_ylim([1, 2])
l3, = ax2.plot(list(range(len(ratio1)))[100:], loss_alg0_ot[100:len(ratio0)], color="firebrick", linestyle='-')
l4, = ax2.plot(list(range(len(ratio1)))[100:], loss_alg1_ot[100:len(ratio0)], color="tomato", linestyle='--')
ax2.set_ylim([-1, 4])
ax2.set_ylabel("Loss of DNN-1", fontsize=15)
plt.legend([l1, l2, l3, l4], ["Normalized Latency, using the loss of active WDs", "Normalized Latency, using the loss of all WDs",
                              "Loss of DNN-1, using the loss of active WDs", "Loss of DNN-1, using the loss of all WDs"], loc=1)
ax1.grid()
plt.savefig("./plots/fig3.pdf", format="pdf", bbox_inches='tight')
plt.show()
