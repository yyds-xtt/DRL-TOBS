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


time_bs0 = np.load("para21/time_bs0_max.npy")
time_bs1 = np.load("para21/time_bs1_max.npy")
time_alg = np.load("para21/time_alg_max.npy")
time_opt = np.load("para21/time_opt_max.npy")

time_complexities = [time_bs0, time_bs1, time_alg, time_opt]
print(np.array_str(np.array(time_complexities), precision=1))

cost_bs0_2 = np.load("para_data/cost_bs0_2.npy")
cost_bs1_2 = np.load("para_data/cost_bs1_2.npy")
cost_alg_2 = np.load("para_data/cost_alg_2.npy")
cost_opt_2 = np.load("para_data/cost_opt_2.npy")

cost_bs0_1 = np.load("para_data/cost_bs0_1.npy")
cost_bs1_1 = np.load("para_data/cost_bs1_1.npy")
cost_alg_1 = np.load("para_data/cost_alg_1.npy")
cost_opt_1 = np.load("para_data/cost_opt_1.npy")


cost_bs0 = cost_bs0_1 + cost_bs0_2
cost_bs1 = cost_bs1_1 + cost_bs1_2
cost_alg = cost_alg_1 + cost_opt_2
cost_opt = cost_opt_1 + cost_opt_2
num_wd_n = [120, 140, 160, 180, 200]


l1, = plt.plot(num_wd_n, cost_bs0/num_wd_n, color="red", marker="*", linestyle='-')
l2, = plt.plot(num_wd_n, cost_bs1/num_wd_n, color="blue",  marker="o", linestyle='-')
l3, = plt.plot(num_wd_n, cost_alg/num_wd_n, color="black", marker="d", linestyle='-')
l4, = plt.plot(num_wd_n, cost_opt/num_wd_n, color="firebrick", marker=".", linestyle='--')
plt.ylabel('Average Latency', fontsize=15)
plt.xlabel('Number of WDs', fontsize=12)
plt.xlim([num_wd_n[0]-1, num_wd_n[-1]+1])
plt.legend([l1, l2, l3, l4], ["ROPT", "MCMC", "DRL-TOBS", "OPTIMUM"], loc=0)
plt.grid()
plt.savefig("./plots/fig7.pdf", format="pdf", bbox_inches='tight')
plt.show()

print((cost_alg/cost_opt).mean())