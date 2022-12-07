import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X0 = np.load('input.npy')
Y0 = np.load('output.npy')
delta = np.load('para_delta.npy')
F = np.load('para_F.npy')
I, N = delta.shape

n_samples = 1000

Y_data = np.zeros((n_samples, I*N), dtype=np.float32)
X_data = np.zeros((n_samples, I), dtype=np.float32)
for i in range(n_samples):
    Y_data[i, :] = Y0[i].reshape(-1)
    X_data[i, :] = X0[i].reshape(-1)


X = torch.from_numpy(X_data.astype(np.float32))
Y = torch.from_numpy(Y_data.astype(np.float32))


class MyDataset(Dataset):
    def __init__(self, d_in, d_out):
        self.X = d_in
        self.Y = d_out
    # stuff

    def __getitem__(self, index):
        # stuff
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(I, N*I)
        self.l2 = nn.Linear(N*I, I)
        self.l3 = nn.Linear(I, I)
        self.l4 = nn.Linear(I, I*N)
        # self.myList = []
        # for i in range(I):
        #     self.myList.append(nn.Linear(I, N).to(device))

    def forward(self, x):
        x = fun.relu(self.l1(x))
        x = fun.relu(self.l2(x))
        x = fun.relu(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        return x
        # z = torch.zeros(x.shape[0], I*N).to(device)
        # for i in range(I):
        #     z[:, i*N:(i+1)*N] = torch.softmax(self.myList[i](x.to(device)), dim=-1)
        # return z


dataset = MyDataset(X, Y)
dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)

learning_rate = 0.01
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_total_steps = len(dataloader)
num_epoch = 100
for epoch in range(num_epoch):
    for i, (d_in_i, d_out_i) in enumerate(dataloader):

        d_in_i = d_in_i.to(device)
        d_out_i = d_out_i.to(device)

        outputs = model(d_in_i)
        loss = criterion(outputs, d_out_i)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.6f}')
            # print(d_in_i.shape)


# y1 = model(torch.from_numpy(X0[-1, :].astype(dtype=np.float32)).to(device)).reshape(I, N)
# y1 = y1.cpu().detach().numpy()
# for i in range(I):
#     index = np.argmax(y1[i, :])
#     y1[i, :] = np.zeros(N)
#     y1[i, index] = 1
# y1_hat = Y0[-1, :].reshape(I, N)
# performance = np.sum(np.absolute(y1-y1_hat))
# print(performance)
cost0 = []
cost1 = []
computing_time = []

f_100 = X0[-100:, :]
y_hat_100 = Y0[-100:, :]
# print(y_hat_100.shape)

for idx in range(f_100.shape[0]):

    f = f_100[idx, :]
    t1 = time.time()
    y_pre = model(torch.from_numpy(f.astype(dtype=np.float32)).to(device)).reshape(I, N)
    t2 = time.time() - t1
    computing_time.append(t2)

    # quantize
    y_pre = y_pre.cpu().detach().numpy()
    for i in range(I):
        index = np.argmax(y_pre[i, :])
        y_pre[i, :] = np.zeros(N)
        y_pre[i, index] = 1

    y1_hat = y_hat_100[idx, :]
    p = np.zeros((I, N))
    for n in range(N):
        delta_n = delta[:, n]
        p[:, n] = np.sqrt(f / delta_n)

    cost_0 = 0
    cost_1 = 0
    for n in range(N):
        cost_0 += np.square(np.sum(np.dot(y_pre[:, n], p[:, n]))) / F[n]
        cost_1 += np.square(np.sum(np.dot(y1_hat[:, n], p[:, n]))) / F[n]
    cost0.append(cost_0)
    cost1.append(cost_1)


cost0 = np.array(cost0)
cost1 = np.array(cost1)

dif = cost0 - cost1
# print(np.max(dif/cost1))
# print(np.min(dif/cost1))
# print(np.mean(dif/cost1))
print(f'percentage dif maximum = {np.max(dif/cost1):.6f}, mean = {np.mean(dif/cost1):.6f}, '
      f'minimum = {np.min(dif/cost1):.6f}')
print(f'computing time maximum = {np.max(computing_time):.6f}, mean = {np.mean(computing_time):.6f}, '
      f'minimum = {np.min(computing_time):.6f}')

plt.hist((100*dif/cost1).reshape(-1).tolist(), density=False, bins=20)
plt.ylabel('Probability Density')
plt.xlabel('Percentage Error /%')
plt.title("Performance of DNN")
plt.grid(True, color="grey", linewidth="0.2", linestyle="-.")
plt.show()


# plt.hist((1000*np.array(computing_time)).reshape(-1).tolist(), density=True, bins=100)
# plt.ylabel('Probability Density')
# plt.xlabel('Time Complexity /ms')
# plt.title("Time Complexity of DNN")
# plt.grid(True, color="grey", linewidth="0.2", linestyle="-.")
# plt.show()
