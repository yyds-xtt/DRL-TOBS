import cvxpy as cp
import numpy as np
import time
import my_lib

from matplotlib import pyplot as plt
from scipy.io import savemat
# Problem data.
# I = 100

num_wd = 180
num_epoch = 400

capacities = np.load("para21/capacities.npy")
num_sever = len(capacities)
capacities = capacities.reshape(num_sever, 1)

delta_22 = 0.5 + 0.5*np.random.rand(num_wd, num_sever)


f_22 = 0.6 + 1.9 * np.random.rand(num_epoch, num_wd)

str3 = 'para21/delta_task_' + str(22)
np.save(str3, delta_22)
str3 = 'para21/sizes_task_' + str(22)
np.save(str3, f_22)

decision_opt = []
cost_opt = []
time_grb = []

for epoch in range(num_epoch):

    f = f_22[epoch, :]

    # data_input.append(f)
    # cvx begin
    t = time.time()
    x = cp.Variable((num_wd, num_sever), boolean=True)

    p = np.zeros((num_wd, num_sever))
    for n in range(num_sever):
        delta_n = delta_22[:, n]
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
    cost_alg = my_lib.obj_val(x.value, delta_22, capacities, f)
    if epoch % 1 == 0:
        print(f"[{epoch+1}/{num_epoch}] time: {elapsed}, opt_dif={prob.value-cost_alg.item():.2f}")

str1 = 'para21/decisions_task_' + str(22)
np.save(str1, decision_opt)

str2 = 'para21/costs_task_' + str(22)
np.save(str2, cost_opt)

str3 = 'para21/times_task_' + str(22)
np.save(str3, time_grb)

# decision_epochs.append(decision_opt)
# cost_epochs.append(cost_opt)
# time_epochs.append(time_grb)
print(f'mean time {np.array(time_grb).mean()}')

plt.hist(np.array(time_grb), density=False, bins=20)
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()



