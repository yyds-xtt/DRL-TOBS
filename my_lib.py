import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
import random
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.io import savemat
import os
import cvxpy as cp
import copy


class Net(nn.Module):
    def __init__(self, num_users, num_servers, device, capacity, delta):
        super(Net, self).__init__()
        self.l1 = nn.Linear(num_users, num_servers * num_users)
        self.l2 = nn.Linear(num_servers * num_users, num_servers * num_users)
        self.l3 = nn.Linear(num_servers * num_users, num_servers * num_users)
        self.l4 = nn.Linear(num_servers * num_users, num_users * num_servers)
        self.capacity = capacity.to(device)
        self.delta = delta.to(device)

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        return x

    def obj_fun(self, x):
        p = torch.sqrt(x.reshape(x.shape[0], x.shape[1], 1).repeat(1, 1, self.delta.shape[1]) /
                       self.delta.reshape(1, self.delta.shape[0], self.delta.shape[1]).repeat(x.shape[0], 1, 1))
        pn = torch.mul(self.forward(x).reshape(x.shape[0], self.delta.shape[0], self.delta.shape[1]), p).sum(axis=1)
        # print(pn.shape, self.capacity.reshape(1, -1).shape)
        return (torch.square(pn)/self.capacity.reshape(1, -1)).sum(axis=1)


class NetSoftMax(nn.Module):
    def __init__(self, num_users, num_servers, device, capacity, delta):
        super(NetSoftMax, self).__init__()
        self.capacity = capacity.to(device)
        self.delta = delta.to(device)
        self.vars = [num_users, num_servers]
        self.device = device
        self.l1 = nn.Linear(num_users, num_servers * num_users)
        self.l2 = nn.Linear(num_servers * num_users, num_users)
        self.l3 = nn.Linear(num_users, num_users)
        self.myList = []
        for i_softmax in range(num_users):
            self.myList.append(nn.Linear(num_users, num_servers).to(device))

    def obj_fun(self, x):
        p = torch.sqrt(x.reshape(x.shape[0], x.shape[1], 1).repeat(1, 1, self.delta.shape[1]) /
                       self.delta.reshape(1, self.delta.shape[0], self.delta.shape[1]).repeat(x.shape[0], 1, 1))
        pn = torch.mul(self.forward(x).reshape(x.shape[0], self.delta.shape[0], self.delta.shape[1]), p).sum(axis=1)
        # print(pn.shape, self.capacity.reshape(1, -1).shape)
        return (torch.square(pn)/self.capacity.reshape(1, -1)).sum(axis=1)

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        z = torch.zeros(x.shape[0], self.vars[0] * self.vars[1]).to(self.device)
        for i1 in range(self.vars[0]):
            z[:, i1 * self.vars[1]: (i1 + 1) * self.vars[1]] = torch.softmax(self.myList[i1](x.to(self.device)), dim=-1)
        return z


def my_loss_fun(outputs, out_opt, capacity, delta, x):
    p = torch.sqrt(x.reshape(x.shape[0], x.shape[1], 1).repeat(1, 1, delta.shape[1]) /
                   delta.reshape(1, delta.shape[0], delta.shape[1]).repeat(x.shape[0], 1, 1))
    pn = torch.mul(outputs.reshape(x.shape[0], delta.shape[0], delta.shape[1]), p).sum(axis=1)
    loss_val = (torch.square(pn) / capacity.reshape(1, -1)).sum(axis=1).sum()
    criterion = nn.MSELoss()
    return criterion(outputs, out_opt)


def gurobi_pro2(delta, capacities_servers, sizes_tasks):
    I, N = delta.shape

    x = cp.Variable((I, N), boolean=True)
    p = np.sqrt(sizes_tasks.reshape(I, 1) / delta)

    cost = 0
    for n in range(N):
        cost += cp.square(cp.sum(cp.multiply(x[:, n], p[:, n]))) / capacities_servers[n]
    objective = cp.Minimize(cost)

    constraints = [cp.sum(x, axis=1) == np.ones(I)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.GUROBI, MIPGap=0)
    return x.value, prob.value


def gcg_pro2(delta_tasks_server, capacities_servers, sizes_tasks):
    I, N = delta_tasks_server.shape

    v_p = np.zeros((I, N))
    for idx_n in range(N):
        delta_n = delta_tasks_server[:, idx_n]
        v_p[:, idx_n] = np.sqrt(sizes_tasks / delta_n)
    # v_p = np.sqrt(sizes_tasks.reshape(I, 1)/delta_tasks_server)

    d_x = np.zeros((I, N))
    # d_x[:, 0] = np.ones(I)
    for idx_i in range(I):
        # idx_n = random.randint(0, N-1)
        idx_n = 0
        d_x[idx_i, idx_n] = 1

    cost_min = 0
    for idx_n in range(N):
        cost_min += np.square(np.sum(np.dot(d_x[:, idx_n], v_p[:, idx_n]))) / capacities_servers[idx_n]
    x_min = copy.deepcopy(d_x)

    num = 0
    while 1:
        num = num + 1
        v_pn = np.zeros(N)
        for idx_n in range(N):
            v_pn[idx_n] = np.inner(d_x[:, idx_n], v_p[:, idx_n])
        # np.multiply(d_x, v_p).sum()
        cost = np.zeros(I)

        for idx_i in range(I):
            cost[idx_i] = np.inner(v_p[idx_i, :]*v_pn, d_x[idx_i, :])/np.inner(capacities_servers, d_x[idx_i, :])

        flag = 0
        for idx_i in range(I):
            cost_i_min = np.min((v_pn+v_p[idx_i, :]) *v_p[idx_i, :] / capacities_servers)
            if cost[idx_i] > cost_i_min:
                flag = 1
                d_x[idx_i, :] = np.zeros(N)
                idx_n = np.argmin((v_pn+v_p[idx_i, :]) *v_p[idx_i, :] / capacities_servers)
                d_x[idx_i, idx_n] = 1
                break

        cost_now = obj_val(d_x, delta_tasks_server, capacities_servers, sizes_tasks)
        if cost_now <= cost_min:
            cost_min = copy.deepcopy(cost_now)
            x_min = copy.deepcopy(d_x)
        if flag == 0:
            break

    return x_min, cost_min, num


def gcg_pro2_new(delta_tasks_server, capacities_servers, sizes_tasks):
    I, N = delta_tasks_server.shape

    v_p = np.sqrt(sizes_tasks.reshape(I, 1)/delta_tasks_server)
    d_x = np.zeros((I, N))
    d_x[:, 0] = np.ones(I)

    # cost_min = 0
    # for idx_n in range(N):
    #     cost_min += np.square(np.sum(np.dot(d_x[:, idx_n], v_p[:, idx_n]))) / capacities_servers[idx_n]
    cost_min = (np.square((d_x*v_p).sum(axis=0))/capacities_servers).sum()
    # if np.abs(cost_min_0-cost_min)> 0.0001:
    #     print("error 0")
    #     print(cost_min_0)
    #     print(cost_min)

    x_min = copy.deepcopy(d_x)

    num = 0
    while 1:
        num = num + 1
        v_pn = np.multiply(d_x, v_p).sum(axis=0)

        cost = (v_p * v_pn.reshape(1, -1) * d_x).sum(axis=1) / (capacities_servers.reshape(1, -1) * d_x).sum(axis=1)
        cost_min_of_all = np.min((v_pn.reshape(1, -1)+v_p)*v_p/capacities_servers.reshape(1, -1), axis=1)
        flag = 0
        for idx_i in range(I):
            # cost_i_min = np.min((v_pn+v_p[idx_i, :]) * v_p[idx_i, :] / capacities_servers)
            # if cost[idx_i] > cost_i_min:
            if cost[idx_i] > cost_min_of_all[idx_i]:
                flag = 1
                d_x[idx_i, :] = np.zeros(N)
                idx_n = np.argmin((v_pn+v_p[idx_i, :]) * v_p[idx_i, :] / capacities_servers)
                d_x[idx_i, idx_n] = 1
                break

        cost_now = obj_val(d_x, delta_tasks_server, capacities_servers, sizes_tasks)
        if cost_now <= cost_min:
            cost_min = copy.deepcopy(cost_now)
            x_min = copy.deepcopy(d_x)
        if flag == 0:
            break

    return x_min, cost_min, num


def gurobi_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data):
    num_devices, num_aps = delta_uplink.shape

    x = cp.Variable((num_devices, num_aps), boolean=True)
    p_uplink = np.sqrt(sizes_data.reshape(-1, 1) / delta_uplink)
    p_fronthaul = np.sqrt(sizes_data.reshape(-1, 1) / delta_fronthaul)

    cost = 0
    for k in range(num_aps):
        cost += cp.square(cp.sum(cp.multiply(x[:, k], p_uplink[:, k]))) / bandwidth_uplink[k]
        cost += cp.square(cp.sum(cp.multiply(x[:, k], p_fronthaul[:, k]))) / bandwidth_fronthaul[k]

    objective = cp.Minimize(cost)

    constraints = [cp.sum(x, axis=1) == np.ones(num_devices)]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.GUROBI, MIPGap=0)
    return x.value, prob.value


def heuristic_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data):
    num_devices, num_aps = delta_uplink.shape
    decision = np.zeros([num_devices, num_aps])
    for idx_i in range(num_devices):
        # argmax_idx = np.argmax(delta_fronthaul[idx_i, :])
        argmax_idx = np.random.choice(np.arange(0, num_aps), p=(delta_uplink[idx_i, :]) / (
                    delta_uplink[idx_i, :]).sum())
        # argmax_idx = np.random.choice(np.arange(stop=num_aps), p=(delta_uplink[idx_i, :]*bandwidth_uplink + delta_fronthaul[idx_i, :]*bandwidth_fronthaul) / (
        #     (delta_uplink[idx_i, :]*bandwidth_uplink).sum()+(delta_fronthaul[idx_i, :]*bandwidth_fronthaul).sum()))
        # argmax_idx = np.random.randint(num_aps)
        decision[idx_i, argmax_idx] = 1
    delay_uplink = obj_val(decision, delta_uplink, bandwidth_uplink, sizes_data)
    delay_fronthaul = obj_val(decision, delta_fronthaul, bandwidth_fronthaul, sizes_data)
    delay = delay_uplink + delay_fronthaul
    return decision, delay


def mcmc_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data):
    iter_num = 50  # 100
    omega = 10       # 5
    cost_ot = []
    num_devices, num_aps = delta_uplink.shape
    decision = np.zeros([num_devices, num_aps])
    for idx_i in range(num_devices):
        argmax_idx = np.random.randint(num_aps)
        decision[idx_i, argmax_idx] = 1
    for iter_val in range(iter_num):
        for idx_i in range(num_devices):
            idx_n_new = np.random.randint(num_aps)
            if decision[idx_i, idx_n_new] != 1:
                decision_new = copy.deepcopy(decision)
                decision_new[idx_i] = np.zeros(num_aps)
                decision_new[idx_i, idx_n_new] = 1
                delay_uplink0 = obj_val(decision, delta_uplink, bandwidth_uplink, sizes_data)
                delay_fronthaul0 = obj_val(decision, delta_fronthaul, bandwidth_fronthaul, sizes_data)
                cost0 = delay_uplink0 + delay_fronthaul0
                delay_uplink1 = obj_val(decision_new, delta_uplink, bandwidth_uplink, sizes_data)
                delay_fronthaul1 = obj_val(decision_new, delta_fronthaul, bandwidth_fronthaul, sizes_data)
                cost1 = delay_uplink1 + delay_fronthaul1
                r_v = np.random.rand()
                p_val = 1 / (1 + np.exp((cost1 - cost0) / omega))
                # print(p_val)
                if r_v <= p_val:
                    decision = copy.deepcopy(decision_new)
                    cost_ot.append(cost1)
                else:
                    cost_ot.append(cost0)
    delay_uplink = obj_val(decision, delta_uplink, bandwidth_uplink, sizes_data)
    delay_fronthaul = obj_val(decision, delta_fronthaul, bandwidth_fronthaul, sizes_data)
    cost = delay_uplink + delay_fronthaul
    return decision, cost, np.array(cost_ot).min()


def heuristic_pro2(delta_tasks_server, capacities_servers, sizes_tasks):
    num_wd, num_es = delta_tasks_server.shape
    decision = np.zeros([num_wd, num_es])
    for idx_i in range(num_wd):
        # argmax_idx = np.random.randint(num_es)
        # argmax_idx = np.argmax(capacities_servers*delta_tasks_server[idx_i, :])
        # argmax_idx = np.random.choice(np.arange(0, num_es), p=(capacities_servers * delta_tasks_server[idx_i, :]) / (
        #             capacities_servers * delta_tasks_server[idx_i, :]).sum())
        argmax_idx = np.random.choice(np.arange(stop=num_es), p=(delta_tasks_server[idx_i, :]) / (
                delta_tasks_server[idx_i, :]).sum())
        decision[idx_i, argmax_idx] = 1

    cost = obj_val(decision, delta_tasks_server, capacities_servers, sizes_tasks)

    return decision, cost


def mcmc_pro2(delta_tasks_server, capacities_servers, sizes_tasks):
    iter_num = 100  # 100
    omega = 5       # 5
    cost_ot = []
    num_wd, num_es = delta_tasks_server.shape
    decision = np.zeros([num_wd, num_es])
    # random initialize
    for idx_i in range(num_wd):
        argmax_idx = np.random.randint(num_es)
        decision[idx_i, argmax_idx] = 1
    for iter_val in range(iter_num):
        for idx_i in range(num_wd):
            idx_n_new = np.random.randint(num_es)
            if decision[idx_i, idx_n_new] != 1:
                decision_new = copy.deepcopy(decision)
                decision_new[idx_i] = np.zeros(num_es)
                decision_new[idx_i, idx_n_new] = 1
                cost0 = obj_val(decision, delta_tasks_server, capacities_servers, sizes_tasks)
                cost1 = obj_val(decision_new, delta_tasks_server, capacities_servers, sizes_tasks)
                r_v = np.random.rand()
                p_val = 1 / (1 + np.exp((cost1 - cost0) / omega))
                if r_v <= p_val:
                    # print(f"iter={iter_val}, idx-i={idx_i:.5f} rand={r_v:.5f}, p_val={p_val:.5f}")
                    decision = copy.deepcopy(decision_new)
                    cost_ot.append(cost1)
                else:
                    cost_ot.append(cost0)
    cost = obj_val(decision, delta_tasks_server, capacities_servers, sizes_tasks)

    return decision, cost, np.array(cost_ot).min()


def gcg_pro1(delta_uplink, delta_fronthaul, bandwidth_uplink, bandwidth_fronthaul, sizes_data):
    num_devices, num_aps = delta_uplink.shape
    p_uplink = np.sqrt(sizes_data.reshape(-1, 1) / delta_uplink)
    p_fronthaul = np.sqrt(sizes_data.reshape(-1, 1) / delta_fronthaul)

    decision = np.zeros([num_devices, num_aps])
    decision[:, 0] = np.ones(num_devices)

    if not np.array_equal(decision.sum(axis=1).reshape(-1), np.ones(num_devices)):
        print("Error in line 64 of my_lib.py\n")

    decision_opt = copy.deepcopy(decision)
    delay_uplink = obj_val(decision_opt, delta_uplink, bandwidth_uplink, sizes_data)
    delay_fronthaul = obj_val(decision_opt, delta_fronthaul, bandwidth_fronthaul, sizes_data)
    cost_opt = delay_uplink + delay_fronthaul

    num = 0
    while 1:
        num = num + 1
        pn_uplink = (decision * p_uplink).sum(axis=0)
        pn_fronthaul = (decision * p_fronthaul).sum(axis=0)

        cost_wds = np.zeros(num_devices)

        for idx_i in range(num_devices):
            delay_uplink_i = np.inner(p_uplink[idx_i, :]*pn_uplink, decision[idx_i, :])/np.inner(bandwidth_uplink, decision[idx_i, :])
            delay_fronthaul_i = np.inner(p_fronthaul[idx_i, :]*pn_fronthaul, decision[idx_i, :])/np.inner(bandwidth_fronthaul, decision[idx_i, :])
            cost_wds[idx_i] = delay_uplink_i + delay_fronthaul_i

        flag = 0
        for idx_i in range(num_devices):
            cost_i_min = np.min((pn_uplink+p_uplink[idx_i, :])*p_uplink[idx_i, :]/bandwidth_uplink +
                                (pn_fronthaul+p_fronthaul[idx_i, :])*p_fronthaul[idx_i, :]/bandwidth_fronthaul)
            if cost_wds[idx_i] > cost_i_min:
                flag = 1
                decision[idx_i, :] = np.zeros(num_aps)
                idx_k = np.argmin((pn_uplink+p_uplink[idx_i, :])*p_uplink[idx_i, :]/bandwidth_uplink +
                                  (pn_fronthaul+p_fronthaul[idx_i, :])*p_fronthaul[idx_i, :]/bandwidth_fronthaul)
                decision[idx_i, idx_k] = 1
                break
        delay_uplink = obj_val(decision, delta_uplink, bandwidth_uplink, sizes_data)
        delay_fronthaul = obj_val(decision, delta_fronthaul, bandwidth_fronthaul, sizes_data)
        cost_now = delay_uplink + delay_fronthaul

        if cost_now <= cost_opt:
            cost_opt = copy.deepcopy(cost_now)
            decision_opt = copy.deepcopy(decision)

        if flag == 0:
            break

    return decision, cost_now, num


def obj_val(x, delta, F, f):
    I, N = delta.shape
    p = np.zeros((I, N))
    for idx_n in range(N):
        delta_n = delta[:, idx_n]
        p[:, idx_n] = np.sqrt(f/delta_n)
    cost_now = 0
    for idx_n in range(N):
        cost_now += np.square(np.sum(np.dot(x[:, idx_n], p[:, idx_n]))) / F[idx_n]
    return cost_now


def obj_val1(x, p, F):
    I, N = x.shape
    cost_now = 0
    for idx_n in range(N):
        cost_now += np.square(np.sum(np.dot(x[:, idx_n], p[:, idx_n]))) / F[idx_n]
    return cost_now


class NetCrossEntropyNN1(nn.Module):
    def __init__(self, num_users, num_servers):
        super(NetCrossEntropyNN1, self).__init__()
        self.vars = [num_users, num_servers]
        self.l1 = nn.Linear(num_servers * num_users, 3*num_servers * num_users)
        self.l2 = nn.Linear(3*num_servers * num_users, num_servers * num_users)
        self.l3 = nn.Linear(num_servers * num_users, num_servers * num_users)
        self.l4 = nn.Linear(num_servers * num_users, num_servers * num_users)

    def forward(self, x):
        # check availability
        # with torch.no_grad():
        #     availability = (x < 7.9)*1

        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = (self.l4(x)).reshape(x.shape[0], self.vars[0], self.vars[1])
        # x = torch.div(x, x.sum(axis=2).reshape(x.shape[0], self.vars[0], 1))
        return x


class NetCrossEntropy(nn.Module):
    def __init__(self, num_users, num_servers, device, capacity, delta):
        super(NetCrossEntropy, self).__init__()
        self.capacity = capacity.to(device)
        self.delta = delta.to(device)
        self.vars = [num_users, num_servers]
        self.device = device
        self.l1 = nn.Linear(num_users, num_servers * num_users)
        self.l2 = nn.Linear(num_servers * num_users, num_users)
        self.l3 = nn.Linear(num_users, num_users)
        self.l4 = nn.Linear(num_users, num_servers * num_users)
        # self.liners = nn.ModuleList([nn.Linear(num_users, num_servers) for i in range(num_users)])

    def obj_fun(self, x):
        p = torch.sqrt(x.reshape(x.shape[0], x.shape[1], 1).repeat(1, 1, self.delta.shape[1]) /
                       self.delta.reshape(1, self.delta.shape[0], self.delta.shape[1]).repeat(x.shape[0], 1, 1))
        pn = torch.mul(self.forward(x).reshape(x.shape[0], self.delta.shape[0], self.delta.shape[1]), p).sum(axis=1)
        # print(pn.shape, self.capacity.reshape(1, -1).shape)
        return (torch.square(pn)/self.capacity.reshape(1, -1)).sum(axis=1)

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = self.l4(x).reshape(x.shape[0], self.vars[0], self.vars[1])
        return x


class DNN2(nn.Module):
    def __init__(self, num_users, num_servers):
        super(DNN2, self).__init__()
        self.vars = [num_users, num_servers]
        self.l1 = nn.Linear(num_users, num_servers * num_users)
        self.l2 = nn.Linear(num_servers * num_users, num_users)
        self.l3 = nn.Linear(num_users, num_users)
        self.l4 = nn.Linear(num_users, num_servers * num_users)

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = self.l4(x).reshape(x.shape[0], self.vars[0], self.vars[1])
        return x


class DNN1(nn.Module):
    def __init__(self, num_users, num_aps):
        super(DNN1, self).__init__()
        self.vars = [num_users, num_aps]
        self.l1 = nn.Linear(num_aps * num_users + num_users, num_aps * num_users)
        self.l2 = nn.Linear(num_aps * num_users, num_aps * num_users)
        self.l3 = nn.Linear(num_aps * num_users, num_aps * num_users)
        self.l4 = nn.Linear(num_aps * num_users, num_aps * num_users)

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = (self.l4(x)).reshape(x.shape[0], self.vars[0], self.vars[1])
        return x


def cross_entropy_loss(out_hat, out_opt):
    criterion = nn.CrossEntropyLoss()
    if out_hat.dim() == 3:
        loss_val = criterion(out_hat.reshape(out_hat.shape[0]*out_hat.shape[1], out_hat.shape[2]),
                         torch.argmax(out_opt.reshape(out_hat.shape[0]*out_hat.shape[1], out_hat.shape[2]), dim=1))
    elif out_hat.dim() == 2:
        loss_val = criterion(out_hat,
                             torch.argmax(out_opt.reshape(out_hat.shape[0], out_hat.shape[1]),
                                          dim=1))
    else:
        print("Error Line 358 my_lib")
    return loss_val


def cross_entropy_loss_down(out_hat, out_opt, indices):
    criterion = nn.CrossEntropyLoss()
    if out_hat.dim() == 3:
        out_down = torch.index_select(out_hat, 1, indices)
        opt_down = torch.index_select(out_opt.reshape(out_hat.shape), 1, indices)
        loss_val = criterion(out_down.reshape(out_down.shape[0] * out_down.shape[1], out_down.shape[2]),
                             torch.argmax(opt_down.reshape(out_down.shape[0] * out_down.shape[1], out_down.shape[2]),
                                          dim=1))
    elif out_hat.dim() == 2:
        out_down = torch.index_select(out_hat, 0, indices)
        opt_down = torch.index_select(out_opt.reshape(out_hat.shape), 0, indices)
        loss_val = criterion(out_down,
                             torch.argmax(opt_down.reshape(out_down.shape[0], out_down.shape[1]),
                                          dim=1))
    else:
        print("Error Line 358 my_lib")
    return loss_val


class NetSoftMax(nn.Module):
    def __init__(self, num_users, num_servers, device, capacity, delta, batch_size):
        super(NetSoftMax, self).__init__()
        self.capacity = capacity.to(device)
        self.delta = delta.to(device)
        self.vars = torch.tensor([num_users, num_servers]).to(device)
        self.device = device
        self.l1 = nn.Linear(num_users, num_servers * num_users)
        self.l2 = nn.Linear(num_servers * num_users, num_users)
        self.l3 = nn.Linear(num_users, num_users)
        # self.liners = nn.ModuleList([nn.Linear(num_users, num_servers) for i in range(num_users)])
        self.l4 = nn.Linear(num_users, num_users * num_servers)

    def obj_fun(self, x):
        p = torch.sqrt(x.reshape(x.shape[0], x.shape[1], 1).repeat(1, 1, self.delta.shape[1]) /
                       self.delta.reshape(1, self.delta.shape[0], self.delta.shape[1]).repeat(x.shape[0], 1, 1))
        pn = torch.mul(self.forward(x).reshape(x.shape[0], self.delta.shape[0], self.delta.shape[1]), p).sum(axis=1)
        # print(pn.shape, self.capacity.reshape(1, -1).shape)
        return (torch.square(pn)/self.capacity.reshape(1, -1)).sum(axis=1)

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = fun.softmax(self.l4(x).reshape(x.shape[0], self.vars[0], self.vars[1]), dim=2)
        return x.reshape(x.shape[0], -1)


class NetSoftMaxNN1(nn.Module):
    def __init__(self, num_users, num_servers):
        super(NetSoftMaxNN1, self).__init__()
        self.num_users, self.num_servers = num_users, num_servers
        self.l1 = nn.Linear(num_users * num_servers, num_users * num_servers)
        self.l2 = nn.Linear(num_users * num_servers, num_users * num_servers)
        self.l3 = nn.Linear(num_users * num_servers, num_users * num_servers)
        self.l4 = nn.Linear(num_users * num_servers, num_users * num_servers)

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = fun.softmax(self.l4(x).reshape(x.shape[0], self.num_users, self.num_servers), dim=2)
        return x.reshape(x.shape[0], -1)


class MyDataset(Dataset):
    def __init__(self, set_length, num_users, num_devices):
        self.X = torch.zeros(set_length, num_users)
        self.Y = torch.zeros(set_length, num_users * num_devices)
        self.idx = 0    # total number of samples generated in the history
        self.set_len = set_length     # length of dataset

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]

    def replace_item(self, item_x, item_y):   # replace the oldest data sample
        index = self.idx % self.set_len
        self.X[index, :] = item_x
        self.Y[index, :] = item_y
        self.idx = self.idx + 1
        # print(f'Replace Item: index = {index}, data = {X[index,0]}')

    def get_total_samples(self):
        return self.idx


class Dataset_dnn1(Dataset):
    def __init__(self, set_length, num_users, num_aps):
        self.X = torch.zeros(set_length, num_users * num_aps + num_users)
        self.Y = torch.zeros(set_length, num_users * num_aps)
        self.idx = 0    # total number of samples generated in the history
        self.set_len = set_length     # length of dataset

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]

    def replace_item(self, item_x, item_y):   # replace the oldest data sample
        index = self.idx % self.set_len
        self.X[index, :] = item_x
        self.Y[index, :] = item_y
        self.idx = self.idx + 1
        # print(f'Replace Item: index = {index}, data = {X[index,0]}')

    def get_total_samples(self):
        return self.idx


def round_x(x, p):
    if len(x.shape) == 2:
        l0, l1 = x.shape
        availability = (p < 7.9)*1
        idx_max = np.argmax(x*availability, axis=-1)
        x1 = np.zeros((l0, l1))
        for idx in range(l0):
            x1[idx, idx_max[idx]] = 1
        return x1
    else:
        l0, l1, l2 = x.shape
        availability = (p < 7.9) * 1
        idx_max = np.argmax(x*availability, axis=-1)
        x1 = np.zeros((l0, l1, l2))
        for idx0 in range(l0):
            for idx1 in range(l1):
                x1[idx0, idx1, idx_max[idx0, idx1]] = 1
        return x1


def round_rand_decision(x):
    if len(x.shape) == 2:
        l0, l1 = x.shape
        # idx_max = np.argmax(x, axis=-1)
        x1 = np.zeros((l0, l1))
        for idx in range(l0):
            idx_r = np.random.choice(np.arange(l1), p=x[idx, :])
            # idx_r = np.random.choice(np.arange(l1), p=fun.softmax(x[idx, :]))
            x1[idx, idx_r] = 1
        return x1
    else:
        l0, l1, l2 = x.shape
        # idx_max = np.argmax(x, axis=-1)
        x1 = np.zeros((l0, l1, l2))
        for idx0 in range(l0):
            for idx1 in range(l1):
                idx_r = np.random.choice(np.arange(l2), p=x[idx0, idx1, :])
                x1[idx0, idx1, idx_r] = 1
        return x1
