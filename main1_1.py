import sys

import numpy as np
import torch
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

dataset = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)
learning_rate = 0.01
model = my_lib.DNN1(num_wd, num_ap).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

cost_alg_ot = []
cost_gcg_ot = []
time_gcg_ot = []
cost_opt_ot = []
time_opt_ot = []
loss_alg_ot = []

for epoch in range(num_time_slots):
    # beginning of each time slot
    sizes_data = 3 + 7 * np.random.rand(num_wd).astype(np.float32)  # current system states
    delta_uplink = delta_uplink * (1 + 0.01 * np.random.randn(num_wd, num_ap))
    delta_uplink = np.maximum(np.minimum(delta_uplink, 0.5), 0.15).astype(np.float32)

    sys_state_i = np.zeros(num_wd + num_ap*num_wd).astype(np.float32)
    sys_state_i[:num_wd] = sizes_data
    sys_state_i[num_wd:] = delta_uplink.reshape(-1)

    # gcg solver
    decision_gcg, cost_gcg, _ = my_lib.gcg_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
    # optimal solver
    decision_opt, cost_opt = my_lib.gurobi_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
    # initialize decision of our alg
    decision_alg = torch.zeros(num_wd, num_ap)
    if epoch < len(dataset):
        if epoch == 0:
            # inference
            decision_hat_real = model(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
            index = torch.argmax(decision_hat_real, dim=1)
            index = index.cpu().detach().numpy()  # offloading decisions of each user

            # get sample
            x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)

            loss = my_lib.cross_entropy_loss(decision_hat_real, y_item.to(device))
            print(f'{loss.item():.6f}')
            # print(decision_hat_real.shape, y_item.shape)
            # print(decision_hat_real.dim(), y_item.dim())
            # update dataset
            dataset.replace_item(x_item, y_item)

        elif epoch < len(dataset):
            # inference
            decision_hat_real = model(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
            index = torch.argmax(decision_hat_real, dim=1)
            index = index.cpu().detach().numpy()  # offloading decisions of each user

            # learning process
            lst1 = list(range(epoch))
            train_set = torch.utils.data.Subset(dataset, lst1)
            dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model(d_in_i)
            loss = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # get sample
            x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)

            # update dataset
            dataset.replace_item(x_item, y_item)
    else:
        # inference
        decision_hat_real = model(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
        index = torch.argmax(decision_hat_real, dim=1)
        index = index.cpu().detach().numpy()  # offloading decisions of each user

        # learning process
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        dataiter = iter(dataloader)
        d_in_i, d_out_i = dataiter.next()
        d_in_i = d_in_i.to(device)
        d_out_i = d_out_i.to(device)
        outputs = model(d_in_i)
        loss = my_lib.cross_entropy_loss(outputs, d_out_i)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # get sample
        x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)
        # update dataset
        dataset.replace_item(x_item, y_item)

    # calculate objetive value (computing latency of our alg)
    y_real = decision_hat_real.cpu().detach().numpy()
    for i in range(num_wd):
        decision_alg[i, index[i]] = 1
    cost_alg = my_lib.obj_val(decision_alg, delta_uplink, bandwidth_uplink, sizes_data) + my_lib.obj_val(decision_alg, delta_fronthaul, bandwidth_fronthaul, sizes_data)
    cost_alg_ot.append(cost_alg)
    cost_gcg_ot.append(cost_gcg)
    cost_opt_ot.append(cost_opt)
    loss_alg_ot.append(loss.item())

    if epoch % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_time_slots}], Loss: {loss.item():.6f}, cost: {cost_alg.item()/cost_opt.item():.6f}')

# plot and plot performance
ratio = np.array(cost_alg_ot).reshape(-1)/np.array(cost_opt_ot).reshape(-1)     # competitive ratio of our alg
# calculating moving average competitive ratio
move_ave_size = 20  # moving average subset size
ratio_mean = np.zeros(len(ratio) - move_ave_size)
ratio_min = np.zeros(len(ratio) - move_ave_size)
ratio_max = np.zeros(len(ratio) - move_ave_size)
for i in range(len(ratio)-move_ave_size):
    ratio_mean[i] = ratio[i:i+move_ave_size].mean()
    ratio_max[i] = ratio[i:i+move_ave_size].max()
    ratio_min[i] = ratio[i:i+move_ave_size].min()

ratio_cvg = ratio[-100:].mean()

# plot the average approximation ratio
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# plt.axhline(y=ratio_cvg, color='peru', linestyle='-', label="approximation\n ratio")
l1, = ax1.plot(list(range(num_time_slots-move_ave_size)), ratio_mean, 'k-', label="moving\naverage")
l2 = ax1.fill_between(list(range(num_time_slots-move_ave_size)), ratio_min, ratio_max, color='silver', label='range')
ax1.set_ylabel('Normalized Communication Latency', fontsize=15)
ax1.set_xlabel('Time Slots', fontsize=12)
ax1.set_xlim([-1, len(ratio)-move_ave_size])
ax1.set_ylim([1, 2])
ax1.grid(axis='both')
l3, = ax2.plot(list(range(num_time_slots-move_ave_size)), loss_alg_ot[:len(ratio_mean)], 'r-', label="Loss1")
ax2.set_ylabel('Loss 1', fontsize=15)
ax2.set_ylim([-1, math.ceil(max(loss_alg_ot))])
plt.legend([l1, l2, l3], ["Moving average", "Range of loss", "Loss of DNN-1"])
# plt.yticks([ratio_cvg, 1.1, 1.2, 1.3, 1.5, 1.7, 1.9])
# plt.savefig("./plots/fig1_1.pdf", format="pdf", bbox_inches='tight')
# plt.savefig("./plots/fig1_2.pdf", format="pdf", bbox_inches='tight')
plt.savefig("./plots/fig1.pdf", format="pdf", bbox_inches='tight')
plt.show()
