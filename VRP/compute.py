import torch
import numpy as np


def c_f(a, b, c, x):
    # print(x)
    return a * x - a * b + c

def S(x):
    return torch.sigmoid(x)


def S_p(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))


def find_multi_hop_neighbors(adj_matrix, num_hops, node_index):
    # 初始化可达性矩阵
    reachability_matrix = torch.eye(adj_matrix.size(0)).to("cuda:0")
    # reachability_matrix = reachability_matrix.to("cuda:0")

    for _ in range(num_hops):
        reachability_matrix = torch.matmul(reachability_matrix, adj_matrix)
        reachability_matrix.fill_diagonal_(0)

    neighbors = torch.where(reachability_matrix[node_index] > 0)[0]
    return neighbors.tolist()


def slice_adj(A, num_hops, node):
    rows = find_multi_hop_neighbors(A, num_hops, node)
    cols = find_multi_hop_neighbors(A, num_hops + 1, node)

    # 创建一个布尔索引矩阵
    bool_rows = torch.zeros(A.size(0), dtype=torch.bool)
    bool_cols = torch.zeros(A.size(1), dtype=torch.bool)
    bool_rows[rows] = True
    bool_cols[cols] = True
    # 应用布尔索引
    new_tensor = A[bool_rows][:, bool_cols]
    return new_tensor


def slice_X(A, X, num_hops, node):
    rows = find_multi_hop_neighbors(A, num_hops + 1, node)

    # 创建一个布尔索引矩阵
    bool_rows = torch.zeros(A.size(0), dtype=torch.bool)
    bool_rows[rows] = True

    # 应用布尔索引
    new_tensor = X[:][bool_rows]
    return new_tensor


def slice_h(A, AD, num_hops, node):
    rows = find_multi_hop_neighbors(A, num_hops, node)
    cols = find_multi_hop_neighbors(A, num_hops + 1, node)

    # 创建一个布尔索引矩阵
    bool_rows = torch.zeros(A.size(0), dtype=torch.bool)
    bool_cols = torch.zeros(A.size(1), dtype=torch.bool)
    bool_rows[rows] = True
    bool_cols[cols] = True
    # 应用布尔索引
    new_tensor = AD[bool_rows][:, bool_cols]
    return new_tensor


def normalize_digraph(A):
    A = A.cpu().numpy()
    Dl = np.sum(A, 0)  # 计算邻接矩阵的度
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)  # 由每个点的度组成的对角矩阵
    AD = np.dot(A, Dn)
    AD = torch.from_numpy(AD).to("cuda:0")
    # print(AD)
    return AD


def matrix_multiplication(A, B):
    # print("A:{},{}".format(len(A), len(A[0])))
    # print("B:{},{}".format(len(B), len(B[0])))
    nrows, ncols = len(A), len(B[0])
    result = [[0] * ncols for _ in range(nrows)]
    # print(A)
    # print(B)
    for i in range(nrows):
        for j in range(ncols):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result
