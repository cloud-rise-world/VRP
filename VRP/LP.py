import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, PULP_CBC_CMD, LpMinimize
from compute import *


def LP_fun(adj_matrix, m, dataset, W, ke, q, Q, y, R, S):
    AD = normalize_digraph(adj_matrix)
    AD_ = slice_h(adj_matrix, AD, 0, m)
    H_ = slice_X(adj_matrix, dataset[0].x, 0, m)
    n = 34
    hidden_num = 10
    R = R.numpy()
    S = S.numpy()
    X = dataset[0].x.numpy()

    temp_W = W.t().tolist()
    c = y[m]

    # temp = [[0], [0], [0], [0], [0], [0], [0]]
    arr_temp = []
    temp = [[0], [0], [0], [0]]
    # temp = [[0], [0], [0], [0], [0], [0], [0]]
    temp[c][0] = 1
    for i in range(4):
        if i == c:
            continue
        else:
            temp[i][0] = -1
            arr_temp.append(temp)
        # temp = [[0], [0], [0], [0], [0], [0], [0]]
        temp = [[0], [0], [0], [0]]
        temp[c][0] = 1

    tempt = matrix_multiplication(matrix_multiplication(AD_.tolist(), H_.tolist()), temp_W)
    # print(arr_temp)

    n_1 = AD_.size()[1]

    m_1 = 34
    arr_re = 1
    import numpy as np

    import matplotlib.pyplot as plt


    for k_t_t in arr_temp:
        # H_1 = []
        model = LpProblem(name="resource-allocation", sense=LpMinimize)
        a = [[LpVariable(f'a_{i}_{j}', lowBound=0, upBound=1) for j in range(m_1)] for i in range(n_1)]
        H_2 = [[LpVariable(f'H_2_{i}_{j}') for j in range(4)] for i in range(1)]

        H_1 = [[LpVariable(f'H_1_{i}_{j}') for j in range(m_1)] for i in range(n_1)]
        H_1_ = matrix_multiplication(matrix_multiplication(AD_.tolist(), H_1), temp_W)

        temp_ = 1
        ttt = matrix_multiplication(H_2, k_t_t)
        # print(ttt[0][0])
        model += ttt[0][0]

        for i in range(n_1):
            for j in range(m_1):
                model += (H_1[i][j] <= X[i][j] + a[i][j])
                model += (H_1[i][j] >= X[i][j] - a[i][j])
                model += (H_1[i][j] <= 1)
                model += (H_1[i][j] >= 0)

        for i in range(n_1):
            temp = (lpSum(a[i]) <= q)
            model += temp

        model += (lpSum(lpSum(a[i][j] for i in range(n_1)) for j in range(m_1)) <= Q)

        for i in range(1):
            for j in range(4):
                print("y = {}x+{}".format(ke['L'][j][0], - ke['L'][j][0] * ke['L'][j][1] + ke['L'][j][2]))
                print("y = {}x+{}".format(ke['U'][j][0], - ke['U'][j][0] * ke['U'][j][1] + ke['U'][j][2]))

                x = np.linspace(R[j], S[j], 50)

                y = ke['L'][j][0] * x - ke['L'][j][0] * ke['L'][j][1] + ke['L'][j][2]



                # x2 = np.linspace(-1, 1, 50)

                y2 = ke['U'][j][0] * x - ke['U'][j][0] * ke['U'][j][1] + ke['U'][j][2]

                y3 = 1 / (1 + np.exp(-x))
                plt.figure()

                plt.plot(x, y)
                plt.plot(x, y2)
                plt.plot(x, y3)

                plt.show()
                # R[j]
                # print(R[j], S[j])
                model += (H_1_[i][j] >= R[j])
                model += (H_1_[i][j] <= S[j])
                model += (H_2[i][j] >= c_f(ke['L'][j][0], ke['L'][j][1], ke['L'][j][2], H_1_[i][j]))
                model += (H_2[i][j] <= c_f(ke['U'][j][0], ke['U'][j][1], ke['U'][j][2], H_1_[i][j]))
                pass

        # Solve the optimization problem
        status = model.solve(PULP_CBC_CMD(msg=0))

        # Get the results
        # print(f"status: {model.status}, {LpStatus[model.status]}")
        # # print(f"objective: {model.objective.value()}")
        # print()
        if model.objective.value() <= 0 or model.status == 0:
            return -1

    return model.objective.value()
