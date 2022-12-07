import cvxpy as cp
import numpy as np
import time
import my_lib

from matplotlib import pyplot as plt
from scipy.io import savemat
# Problem data.
# I = 100

num_wd_set = [120, 140, 160, 180, 200]
num_epoch = 50

# # capacities = np.array([3.584, 3.712, 6.656, 6.144, 6.400, 7.168, 7.424, 3.072, 6.656, 3.200, 3.584, 3.712, 6.656, 6.144, 6.400, 7.168, 7.424, 3.072, 6.656, 3.200])
capacities = np.array([3.584, 3.712, 6.656, 6.144, 6.400, 7.168, 7.424, 3.072, 3.584, 3.712, 6.656, 6.144, 6.400, 7.168, 7.424, 3.072])
num_sever = len(capacities)
capacities = capacities.reshape(num_sever, 1)
delta_set = 0.5 + 0.5*np.random.rand(num_wd_set[-1], num_sever)
f_set = 0.6 + 1.9 * np.random.rand(num_epoch, num_wd_set[-1])
np.save("para21/delta_set", delta_set)
np.save("para21/f_set", f_set)
np.save("para21/capacities", capacities)

# delta_set = np.load("para21/delta_set")
# f_set = np.load("para21/f_set")
# capacities = np.load("para21/capacities")
# num_sever = len(capacities)


decision_epochs = []
cost_epochs = []
time_epochs = []

num_count = 1
for num_wd in num_wd_set:

    delta = delta_set[:num_wd, :]
    decision_opt = []
    cost_opt = []
    time_grb = []

    for epoch in range(num_epoch):

        # f = 0.6 + 1.9 * np.random.rand(num_wd)
        f = f_set[epoch, :num_wd]
        # data_input.append(f)
        # cvx begin
        t = time.time()
        x = cp.Variable((num_wd, num_sever), boolean=True)

        p = np.zeros((num_wd, num_sever))
        for n in range(num_sever):
            delta_n = delta[:, n]
            p[:, n] = np.sqrt(f / delta_n)

        cost = 0
        for n in range(num_sever):
            cost += cp.square(cp.sum(cp.multiply(x[:, n], p[:, n]))) / capacities[n]
        objective = cp.Minimize(cost)

        constraints = [cp.sum(x, axis=1) == np.ones(num_wd)]
        prob = cp.Problem(objective, constraints)
        # cvx end
        # The optimal objective value is returned by `prob.solve()`.
        # print(cp.installed_solvers())
        result = prob.solve(solver=cp.GUROBI, MIPGap=0)
        elapsed = time.time() - t

        decision_opt.append(x.value.astype(np.float32))
        cost_opt.append(prob.value)
        time_grb.append(elapsed)
        cost_alg = my_lib.obj_val(x.value, delta, capacities, f)
        if epoch % 1 == 0:
            print(f"[{num_count}/{len(num_wd_set)}]-[{epoch}/{num_epoch}] time: {elapsed}, opt_dif={prob.value-cost_alg.item():.2f}")
    num_count += 1

    str1 = 'para21/decisions_' + str(num_wd)
    np.save(str1, decision_opt)

    str2 = 'para21/costs_' + str(num_wd)
    np.save(str2, cost_opt)

    str3 = 'para21/times_' + str(num_wd)
    np.save(str3, time_grb)

    # decision_epochs.append(decision_opt)
    # cost_epochs.append(cost_opt)
    # time_epochs.append(time_grb)
    print(f'mean time {np.array(time_grb).mean()}')

plt.hist(np.array(time_grb), density=False, bins=20)
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()


