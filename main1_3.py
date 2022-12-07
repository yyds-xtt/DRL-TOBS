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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# system setting
# num_wd = 100     # number of Wireless devices
# num_wd_n = [80, 90, 100, 110, 120]      # number of Wireless devices
num_wd_n = [120, 140, 160, 180, 200]
num_ap = 10      # number of base stations (access points)
num_time_slots = 100   # total number of learning time slots
num_test_slots = 50   # total number of testing  time slots

dataset_len = 30   # size of the dataset for DNN-2
batch_size = 30     # size of each training batch

cost_alg_n = []
cost_bs0_n = []
cost_bs1_n = []
cost_opt_n = []
cost_gcg_n = []

time_alg_median = []
time_bs0_median = []
time_bs1_median = []
time_opt_median = []
time_gcg_median = []
time_trn_median = []

time_alg_mean = []
time_bs0_mean = []
time_bs1_mean = []
time_opt_mean = []
time_gcg_mean = []
time_trn_mean = []


bandwidth_uplink = 100 * np.random.rand(num_ap).astype(np.float32)
bandwidth_uplink[0] = 45
bandwidth_fronthaul = 1000 * np.random.rand(num_ap).astype(np.float32)
num_round = 0
delta_uplink_set = 0.15 + 0.35 * np.random.rand(num_wd_n[-1], num_ap).astype(np.float32)
for i in range(num_wd_n[-1]):
    threshold_i = np.sort(delta_uplink_set[i, :])[4]
    for k in range(num_ap):
        if delta_uplink_set[i, k] <= threshold_i:
            delta_uplink_set[i, k] = 0

delta_fronthaul_set = 0.1 + 0 * np.random.rand(num_wd_n[-1], num_ap).astype(np.float32)

sizes_data_set = 1 + 4 * np.random.rand(num_time_slots, num_wd_n[-1]).astype(np.float32)  # current system states
sizes_data_set_test = 1 + 4 * np.random.rand(num_test_slots, num_wd_n[-1]).astype(np.float32)  # current system states


for num_wd in num_wd_n:
    delta_uplink = delta_uplink_set[:num_wd, :]
    delta_fronthaul = delta_fronthaul_set[:num_wd, :]

    dataset0 = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)
    dataset1 = my_lib.Dataset_dnn1(set_length=dataset_len, num_users=num_wd, num_aps=num_ap)
    learning_rate = 0.01
    model0 = my_lib.DNN1(num_wd, num_ap).to(device)
    optimizer_gcg = torch.optim.SGD(model0.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer_gcg = torch.optim.SGD(model0.parameters(), lr=learning_rate)

    for epoch in range(num_time_slots):
        # beginning of each time slot
        # sizes_data = 3 + 7 * np.random.rand(num_wd).astype(np.float32)  # current system states
        sizes_data = sizes_data_set[epoch, :num_wd]
        delta_uplink = delta_uplink * (1 + 0.01 * np.random.randn(num_wd, num_ap))
        delta_uplink = np.maximum(np.minimum(delta_uplink, 0.5), 0.15).astype(np.float32)

        sys_state_i = np.zeros(num_wd + num_ap*num_wd).astype(np.float32)
        sys_state_i[:num_wd] = sizes_data
        sys_state_i[num_wd:] = delta_uplink.reshape(-1)

        # gcg solver
        decision_gcg, cost_gcg, _ = my_lib.gcg_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
        decision_hat_real = model0(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
        index = torch.argmax(decision_hat_real, dim=1)
        index = index.cpu().detach().numpy()  # offloading decisions of each user
        decision_alg = torch.zeros(num_wd, num_ap)
        for i in range(num_wd):
            decision_alg[i, index[i]] = 1
        cost_alg = my_lib.obj_val(decision_alg, delta_uplink, bandwidth_uplink, sizes_data) + my_lib.obj_val(decision_alg,
                                                                                                             delta_fronthaul,
                                                                                                             bandwidth_fronthaul,
                                                                                                             sizes_data)
        # get data sample
        x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)

        if epoch == 0:
            loss = my_lib.cross_entropy_loss(decision_hat_real, y_item.to(device))
        else:
            # learning process
            if epoch < len(dataset0):
                lst1 = list(range(epoch))
                train_set = torch.utils.data.Subset(dataset0, lst1)
                dataloader = DataLoader(dataset=train_set, batch_size=epoch if epoch < batch_size else batch_size, shuffle=True)
            else:
                dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
            # dataloader = DataLoader(dataset=dataset_gcg, batch_size=batch_size, shuffle=True)
            dataiter = iter(dataloader)
            d_in_i, d_out_i = dataiter.next()
            d_in_i = d_in_i.to(device)
            d_out_i = d_out_i.to(device)
            outputs = model0(d_in_i)
            loss = my_lib.cross_entropy_loss(outputs, d_out_i)
            loss.backward()
            optimizer_gcg.step()
            optimizer_gcg.zero_grad()

        dataset0.replace_item(x_item, y_item)

        if epoch % 10 == 0:
            print(
                f'\tLEARNING[{num_round+1}/5] Epoch [{int(100*(epoch + 1)/num_time_slots)} %], cost_alg: {cost_alg.item() / cost_gcg.item():.6f}, loss: {loss.item():.6f}')

    cost_alg_ot = []
    cost_gcg_ot = []
    cost_opt_ot = []
    cost_bs0_ot = []
    cost_bs1_ot = []

    time_gcg_ot = []
    time_opt_ot = []
    time_alg_ot = []
    time_bs0_ot = []
    time_bs1_ot = []
    time_trn_ot = []

    for epoch in range(num_test_slots):
        # beginning of each time slot
        # sizes_data = 3 + 7 * np.random.rand(num_wd).astype(np.float32)  # current system states
        sizes_data = sizes_data_set_test[epoch, :num_wd]
        delta_uplink = delta_uplink * (1 + 0.01 * np.random.randn(num_wd, num_ap))
        delta_uplink = np.maximum(np.minimum(delta_uplink, 0.5), 0.15).astype(np.float32)

        sys_state_i = np.zeros(num_wd + num_ap*num_wd).astype(np.float32)
        sys_state_i[:num_wd] = sizes_data
        sys_state_i[num_wd:] = delta_uplink.reshape(-1)

        # gcg solver
        ta = time.time()
        decision_gcg, cost_gcg, _ = my_lib.gcg_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
        time_gcg_ot.append(time.time()-ta)
        cost_gcg_ot.append(cost_gcg)
        # optimal solver
        ta = time.time()
        decision_opt, cost_opt = my_lib.gurobi_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
        time_opt_ot.append(time.time() - ta)
        cost_opt_ot.append(cost_opt)
        # baseline0
        t_a = time.time()
        decision_bs0, cost_bs0 = my_lib.heuristic_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
        time_bs0_ot.append(time.time() - t_a)
        cost_bs0_ot.append(cost_bs0)
        # baseline1
        t_a = time.time()
        decision_bs1, cost_bs1, _ = my_lib.mcmc_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data)
        time_bs1_ot.append(time.time() - t_a)
        cost_bs1_ot.append(cost_bs1)

        # alg
        ta = time.time()
        decision_hat_real = model0(torch.from_numpy(sys_state_i).reshape(1, -1).to(device)).reshape(num_wd, num_ap)
        time_alg_ot.append(time.time()-ta)
        index = torch.argmax(decision_hat_real, dim=1)
        index = index.cpu().detach().numpy()  # offloading decisions of each user
        decision_alg = torch.zeros(num_wd, num_ap)
        for i in range(num_wd):
            decision_alg[i, index[i]] = 1
        cost_alg = my_lib.obj_val(decision_alg, delta_uplink, bandwidth_uplink, sizes_data) + my_lib.obj_val(decision_alg,
                                                                                                             delta_fronthaul,
                                                                                                             bandwidth_fronthaul,
                                                                                                             sizes_data)
        cost_alg_ot.append(cost_alg)
        # get data sample
        x_item, y_item = torch.from_numpy(sys_state_i).reshape(-1), torch.from_numpy(decision_gcg).reshape(-1)
        # leaning process
        t_a = time.time()
        dataloader = DataLoader(dataset=dataset0, batch_size=batch_size, shuffle=True)
        dataiter = iter(dataloader)
        d_in_i, d_out_i = dataiter.next()
        d_in_i = d_in_i.to(device)
        d_out_i = d_out_i.to(device)
        outputs = model0(d_in_i)
        loss = my_lib.cross_entropy_loss(outputs, d_out_i)
        loss.backward()
        optimizer_gcg.step()
        optimizer_gcg.zero_grad()
        time_trn_ot.append(time.time() - t_a)
        # update dataset
        dataset0.replace_item(x_item, y_item)
        if epoch % 10 == 0:
            print(
                f'TESTING [{num_round+1}/5] Epoch [{int(100*(epoch + 1)/num_test_slots)} %], cost_alg: {cost_alg.item() / cost_opt:.6f}, cost_bs0: {cost_bs0.item() / cost_opt:.6f}, cost_bs1: {cost_bs1.item() / cost_opt:.6f}')
    num_round = num_round + 1
    cost_bs0_n.append(np.array(cost_bs0_ot).mean())
    cost_bs1_n.append(np.array(cost_bs1_ot).mean())
    cost_alg_n.append(np.array(cost_alg_ot).mean())
    cost_opt_n.append(np.array(cost_opt_ot).mean())
    cost_gcg_n.append(np.array(cost_gcg_ot).mean())

    time_bs0_median.append(np.median(np.array(time_bs0_ot)))
    time_bs1_median.append(np.median(np.array(time_bs1_ot)))
    time_alg_median.append(np.median(np.array(time_alg_ot)))
    time_opt_median.append(np.median(np.array(time_opt_ot)))
    time_trn_median.append(np.median(np.array(time_trn_ot)))
    time_gcg_median.append(np.median(np.array(time_gcg_ot)))

    time_bs0_mean.append(np.array(time_bs0_ot).mean())
    time_bs1_mean.append(np.array(time_bs1_ot).mean())
    time_alg_mean.append(np.array(time_alg_ot).mean())
    time_opt_mean.append(np.array(time_opt_ot).mean())
    time_gcg_mean.append(np.array(time_gcg_ot).mean())
    time_trn_mean.append(np.array(time_trn_ot).mean())

num_wd_ot = np.array(num_wd_n)
cost_bs0_n = np.array(cost_bs0_n)
cost_bs1_n = np.array(cost_bs1_n)
cost_alg_n = np.array(cost_alg_n)
cost_opt_n = np.array(cost_opt_n)
cost_gcg_n = np.array(cost_gcg_n)

print(f'gcg {time_gcg_median}\ntrn {time_trn_median}\nalg {time_alg_median}')
print(f'gcg {time_gcg_mean}\ntrn {time_trn_mean}\nalg {time_alg_mean}')


fig, ax1 = plt.subplots()
l1, = ax1.plot(num_wd_n, cost_bs0_n/num_wd_ot, color="red", marker="*", linestyle='-')
l2, = ax1.plot(num_wd_n, cost_bs1_n/num_wd_ot, color="blue",  marker="o", linestyle='-')
l3, = ax1.plot(num_wd_n, cost_alg_n/num_wd_ot, color="black", marker="d", linestyle='-')
l4, = ax1.plot(num_wd_n, cost_opt_n/num_wd_ot, color="firebrick", marker=".", linestyle='-')
# l5, = ax1.plot(num_wd_n, cost_gcg_n/num_wd_ot, color="pink", marker=".", linestyle='--')



# ax1.set_ylabel('Communication Latency', fontsize=15)
# ax1.set_xlabel('Number of WDs', fontsize=12)
# plt.legend([l1, l2, l3, l4], ["baseline 0", "baseline 1", "alg", "opt"], loc=0)
# plt.grid()
# # plt.savefig("./plots/fig5.pdf", format="pdf", bbox_inches='tight')
# plt.show()


# fig, ax1 = plt.subplots()
# l1, = ax1.plot(num_wd_n, time_bs0_n, color="red", marker="*", linestyle='-')
# l2, = ax1.plot(num_wd_n, time_bs1_n, color="blue",  marker="o", linestyle='-')
# l3, = ax1.plot(num_wd_n, time_alg_n, color="black", marker="d", linestyle='-')
# l4, = ax1.plot(num_wd_n, time_opt_n, color="firebrick", marker=".", linestyle='-')
# ax1.set_ylabel('Time Complexity', fontsize=15)
# ax1.set_xlabel('Number of WDs', fontsize=12)
# plt.legend([l1, l2, l3, l4], ["baseline 0", "baseline 1", "alg", "opt"], loc=0)
# plt.grid()
# # plt.savefig("./plots/fig5_1.pdf", format="pdf", bbox_inches='tight')
# plt.show()
#

# delta_uplink_set = np.load('para_data/delta_uplink_set.npy')
# delta_fronthaul_set = np.load('para_data/delta_fronthaul_set.npy')
# sizes_data_set = np.load('para_data/sizes_data_set.npy')
# np.save("para_data/cost_bs0_1", cost_bs0_n)
# np.save("para_data/cost_bs1_1", cost_bs1_n)
# np.save("para_data/cost_alg_1", cost_alg_n)
# np.save("para_data/cost_opt_1", cost_opt_n)
# np.save("para_data/cost_gcg_1", cost_gcg_n)
#
# np.save("para_data/time_bs0_1", time_bs0_median)
# np.save("para_data/time_bs1_1", time_bs1_median)
# np.save("para_data/time_alg_1", time_alg_median)
# np.save("para_data/time_opt_1", time_opt_median)
# np.save("para_data/time_gcg_1", time_gcg_median)
