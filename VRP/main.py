import greatx
import scipy as sp
import torch
from greatx.nn.models import GCN
from greatx.training import Trainer
from torch_geometric.datasets import Planetoid, KarateClub
import torch.nn as nn

from VRP.LP import LP_fun
from VRP.active_bound import compute_active_bound
from VRP.my_model import MyModel
from greatx.nn.layers import GCNConv, Sequential

# Any PyG dataset is available!
from torch_geometric.utils import to_dense_adj

import numpy as np
import torch
from scipy.optimize import fsolve

from sympy import *
from sympy import *

activations = {}
l = {}
u = {}
lay_arr = ['gconv.0']
arr = {'gconv.0': (34, 4)}

for name in lay_arr:
    l[name] = []
    u[name] = []

    for i in range(arr[name][0]):
        l[name].append([])
        u[name].append([])
        for j in range(arr[name][1]):
            l[name][i].append(float('inf'))
            u[name][i].append(float('-inf'))


def get_activation(name):
    def hook(model, input, output):
        # print(input)
        pass
        # print(name, output.size())
        # print(output[0][0])
        # print(output)
        activations[name] = output.detach()
        # print(activations[name].size())
        print(len(activations[name]), len(activations[name][0]))
        for i in range(len(activations[name])):
            for j in range(len(activations[name][i])):
                temp = float(activations[name][i][j])

                l[name][i][j] = min(l[name][i][j], temp)
                u[name][i][j] = max(u[name][i][j], temp)
        # for r in range(len(activations[name][0])):
        #     temp = float(activations[name][0][r])
        # print("name:{},value:{}".format(name, temp))
        # l[name][r] = min(l[name][r], temp)
        # u[name][r] = max(u[name][r], temp)
        # if activations["key"]:
        #     l[name][r] = temp
        #     u[name][r] = temp

        # activations["key"] = False

    return hook


W_dic = ['gconv.0.lin.weight']
Bias_dic = ['gconv.0.bias']

# dataset = Planetoid(root='.', name='Cora')
dataset = KarateClub()
data = dataset[0]
# great_c_t(data.y)
num_features = data.num_features
# print(num_features)
hidden_feats = 10
# greatx.training.trainer.CUSTOM_LOSS = "multiclass_hinge_loss"
model = MyModel(num_features, hidden_feats, dataset.num_classes)
# print(model)
trainer = Trainer(model, device='cuda:0')  # or 'cpu'

y_t = []

trainer.fit(data, epochs=1000)

# for i in trainer.predict(data):
#     i.max


# print(model)
hooks = []
# print(model.gconv[2])
hooks.append(model.gconv[0].register_forward_hook(get_activation(lay_arr[0])))
# hooks.append(model.gconv[2].register_forward_hook(get_activation(lay_arr[1])))

# for i in range(2):
#     print(model.state_dict()[W_dic[i]].size())
#     print(model.state_dict()[Bias_dic[i]].size())
logs = trainer.evaluate(data)

for hook in hooks:
    hook.remove()

# 转换为邻接矩阵

acc = logs["acc"]

q = 1
Q = 1
R = None
S = None
Qrr = [5, 10, 20, 40, 80, 120, 160, 200]
Qrr = [3, 5, 10, 20, 30]
for Q in Qrr:
    q = 2
    # print("Q:{},q:{}".format(Q, q))
    pos = {"num": 0, "index": []}
    neg = {"num": 0, "index": []}

    for m in range(34):
        for i in range(len(lay_arr)):
            W = model.state_dict()[W_dic[i]]
            B = model.state_dict()[Bias_dic[i]]
            l_under = torch.tensor(l[lay_arr[i]])[m]
            u_under = torch.tensor(u[lay_arr[i]])[m]

            edge_index = data.edge_index
            adj = to_dense_adj(edge_index)[0]
            x = data.x
            activation = activations[lay_arr[i]][m]
            ke = compute_active_bound(adj, x, activation, m, q, Q, W, l_under, u_under, i, R, S)
            R = ke["R"]
            S = ke["S"]

        # print(data.y)
        # print(torch.argmax(trainer.predict(data), 1))
        # print(ksd)
        temp = LP_fun(adj, m, dataset, W, ke["ke"], q, Q, torch.argmax(trainer.predict(data), 1), R.cpu(), S.cpu())

        if temp > 0:
            pos["num"] += 1
            pos["index"].append(m)
        else:
            neg["num"] += 1
            neg["index"].append(m)
    print("Q:{},q:{},neg:{}".format(Q, q, pos))
