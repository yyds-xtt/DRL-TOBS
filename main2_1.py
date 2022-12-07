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
from matplotlib import pyplot as plt
from scipy.io import savemat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# system setting

num_time_slots = 400   # total number of time slots
dataset_len = 200   # size of the dataset for DNN-2
batch_size = 50     # size of each training batch

capacities = np.load("para21/capacities.npy")
num_sv = len(capacities)      # number of Edge servers
str3 = 'para21/delta_task_' + str(22) + '.npy'
delta_22 = np.load(str3)
str3 = 'para21/sizes_task_' + str(22) + '.npy'
f_22 = np.load(str3)
num_wd = delta_22.shape[0]
capacities_sv = capacities.reshape(num_sv, 1).astype(np.float32)
costs_opt = np.load('para21/costs_task_' + str(22) + '.npy')
decisions_opt = np.load('para21/decisions_task_' + str(22) + '.npy')

delta = copy.deepcopy(delta_22)

dataset = my_lib.MyDataset(set_length=dataset_len, num_users=num_wd, num_devices=num_sv)
learning_rate = 0.01
model_gcg = my_lib.DNN2(num_wd, num_sv).to(device)  # model using proposed approximation algorithm
optimizer = torch.optim.SGD(model_gcg.parameters(), lr=learning_rate, momentum=0.9)

cost_alg_ot = []
cost_gcg_ot = []
time_gcg_ot = []
cost_opt_ot = []
time_opt_ot = []
loss_alg_ot = []

for epoch in range(num_time_slots):
    # beginning of each time slot
    # f = 0.6 + 1.9 * np.random.rand(num_wd).astype(np.float32)  # current system states
    f = f_22[epoch, :].astype(np.float32)
    # gcg solver
    decision_gcg, cost_gcg, _ = my_lib.gcg_pro2(delta, capacities_sv.reshape(-1), f.reshape(-1))
    # optimal solver
    # decision_opt, cost_opt = my_lib.gurobi_pro2(delta, capacities_sv.reshape(-1), f.reshape(-1))
    cost_opt = costs_opt[epoch]
    decision_opt = decisions_opt[epoch]

    # initialize decision of our alg
    decision_alg = torch.zeros(num_wd, num_sv)
    # inference
    decision_hat_real = model_gcg(torch.from_numpy(f).reshape(1, num_wd).to(device)).reshape(num_wd, num_sv)
    index = torch.argmax(decision_hat_real, dim=1)
    index = index.cpu().detach().numpy()  # offloading decisions of each user

    for i in range(num_wd):
        decision_alg[i, index[i]] = 1
    cost_alg = my_lib.obj_val(decision_alg, delta, capacities_sv, f)

    # get sample
    x_item, y_item = torch.from_numpy(f).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)

    if epoch == 0:
        loss0 = my_lib.cross_entropy_loss(decision_hat_real, y_item.to(device))
        # update dataset
        dataset.replace_item(x_item, y_item)
    else:
        if epoch < len(dataset):
            lst1 = list(range(epoch))
            train_set = torch.utils.data.Subset(dataset, lst1)
            dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size,
                                    shuffle=True)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        dataiter = iter(dataloader)
        d_in_i, d_out_i = dataiter.next()
        d_in_i = d_in_i.to(device)
        d_out_i = d_out_i.to(device)
        outputs = model_gcg(d_in_i)
        loss0 = my_lib.cross_entropy_loss(outputs, d_out_i)
        loss0.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update dataset
        dataset.replace_item(x_item, y_item)

    cost_alg_ot.append(cost_alg)
    cost_gcg_ot.append(cost_gcg)
    cost_opt_ot.append(cost_opt)
    loss_alg_ot.append(loss0.item())

    if epoch % 10 == 0:
        print(
            f'Epoch [{epoch + 1}/{num_time_slots}], Loss: {loss0.item():.6f}, cost: {cost_alg.item()/cost_opt.item():.6f}')


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

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
l1, = ax1.plot(list(range(num_time_slots-move_ave_size)), ratio_mean, 'k-', label="moving\naverage")
l2 = ax1.fill_between(list(range(num_time_slots-move_ave_size)), ratio_min, ratio_max, color='silver', label='range')
ax1.set_ylabel('Normalized Processing Latency', fontsize=15)
ax1.set_xlabel('Time Slots', fontsize=12)
ax1.set_xlim([-1, len(ratio)-move_ave_size])
ax1.set_ylim([1, 2])
l3, = ax2.plot(list(range(num_time_slots-move_ave_size)), loss_alg_ot[:len(ratio_mean)], 'r-', label="Loss1")
ax2.set_ylabel('Loss 2', fontsize=15)
ax2.set_ylim([-1, math.ceil(max(loss_alg_ot))])
plt.legend([l1, l2, l3], ["Moving average", "Range of loss", "Loss of DNN-2"], loc=1)
ax1.grid()
plt.savefig("./plots/fig2.pdf", format="pdf", bbox_inches='tight')
plt.show()
