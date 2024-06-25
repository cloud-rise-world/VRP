from sympy import *
from compute import *
from scipy.optimize import fsolve
from torch_geometric.utils import to_dense_adj


def great_c_t(y):
    num = max(y) + 1
    arr = []
    for i in y:
        brr = []
        for j in range(num):
            if i == j:
                brr.append([1])
            else:
                brr.append([0])
        arr.append(brr)
    # print(arr)
    # print(len(arr))
    # for i in arr:
    #     print(i)

    return arr


def find_intersection(m, b, initial_guess=0):
    # Define the equation to solve: sigmoid(x) - (mx + b) = 0
    # print(b)
    # print(m)
    def equation(x):
        # print(type(x))
        a = m * x + b
        y = sigmoid(x) - a
        return y

    # Use fsolve to find the root, starting from x = 0
    initial_guess = 0
    intersection_x = fsolve(equation, initial_guess)

    # Compute the corresponding y value
    intersection_y = sigmoid(intersection_x[0])

    return intersection_x[0], intersection_y


from torch.nn import functional as F


# def sum_top_Q_2(nodes, adj):
#     """
#             Compute bounds on the first layer for binary node attributes.
#             Parameters
#             ----------
#             input: torch.tensor (boolean) dimension [Num. L-1 hop neighbors, D]
#                 binary node attributes (one vector for all neighbors of the input nodes)
#                 OR: [N, D] for the whole graph when slice_input=True
#             nodes:  numpy.array, int64 dim [Num l hop neighbors,]
#                 L-l hop neighbors of the target nodes.
#             q:  int
#                 per-node constraint on the number of attribute perturbations
#             Q:  int
#                 global constraint on the number of attribute perturbations
#             lower_bound: bool
#                 Whether to compute the lower bounds (True) or upper bounds (False)
#             slice_input: bool
#                 Whether the input is the whole attribute matrix. If True, we slice the
#                 node features accordingly.
#
#             Returns
#             -------
#             bounds: torch.tensor (float32) dimension [Num. L-2 hop neighbors x H_2]
#                 Lower/upper bounds on the hidden activations in the second layer.
#             """
#
#     # Convention:
#     # N: number of nodes in the current layer
#     # N_nbs: number of neighbors of the nodes in the current layer
#     # D: dimension of the node attributes (i.e. H_1)
#     # H: dimension of the first hidden layer (i.e. H_2)
#
#     adj_slice, nbs = slice_adj(adj, 1, nodes)  # 邻居
#
#     N_nbs = len(nbs)  # 邻居数量
#     N = len(nodes)  #
#
#
#     # [N_nbs x D] => [N_nbs x D x 1]
#     input_extended = input.unsqueeze(2)
#
#     # get the positive and negative parts of the weights
#     # [D x H]  => [1 x D X H]
#     W_plus = F.relu(self.weights).unsqueeze(0)
#     W_minus = F.relu(-self.weights).unsqueeze(0)
#
#     # [N_nbs x D x H]
#     if lower_bound:
#         bounds_nbs = input_extended.mul(W_plus) + (1 - input_extended).mul(W_minus)
#     else:
#         bounds_nbs = (1 - input_extended).mul(W_plus) + input_extended.mul(W_minus)
#     # top q entries per dimension in D
#     # => [N_nbs x q x H]
#     top_q_vals = bounds_nbs.topk(q, 1)[0]
#
#     # => [N_nbs x q*H]
#     top_q_vals = top_q_vals.reshape([N_nbs, -1])
#
#     # [N x N_nbs x 1]
#     adj_extended = adj_slice.unsqueeze(2).to_dense()
#
#     # per-node bounds (after aggregating the neighbors)
#     # [N x N_nbs x q x H]
#     aggregated = adj_extended.mul(top_q_vals).reshape([N, N_nbs, q, -1])
#
#     # sum of the top Q values of the top q values per dimension
#     # [N, Q, H] => [N, H]
#     n_sel = min(Q, N_nbs * q)
#
#     top_Q_vals = aggregated.reshape([N, -1, self.H]).topk(k=n_sel, dim=1)[0].sum(1)
#
#     if lower_bound:
#         top_Q_vals *= -1
#
#     # Add the normal hidden activations for the input
#     bounds = top_Q_vals + self.forward(input, nodes)


def sum_top_Q(A, X, AD, H_P, m, q, Q, W):
    N1_m = find_multi_hop_neighbors(A, 1, m)
    # print("N1_m:{}".format(N1_m))
    zero = torch.zeros_like(W[0])
    S_arr_last = []
    R_arr_last = []
    for j in range(H_P.size()[0]):
        W_pos = torch.where(W[j] < 0, zero, W[j])
        W_neg = torch.where(W[j] > 0, zero, -W[j])
        S_arr = []
        R_arr = []
        for n in N1_m:
            S_arr_temp = (torch.ones_like(X[n]) - X[n]) * W_pos + X[n] * W_neg
            sort_S = torch.sort(S_arr_temp, descending=True).values

            R_arr_temp = X[n] * W_pos + (torch.ones_like(X[n]) - X[n]) * W_neg
            # print("s_1_arr:{}".format(s_1_arr))
            sort_R = torch.sort(R_arr_temp, descending=True).values
            for i in range(1, q + 1):
                temp = AD[m][n] * sort_S[i - 1]
                S_arr.append(temp.item())
                temp = AD[m][n] * sort_R[i - 1]
                R_arr.append(temp.item())

        S_arr.sort()

        R_arr.sort()

        S_arr_last.append(sum(S_arr[:Q]))
        R_arr_last.append(sum(R_arr[:Q]))
    # print("brr:{},drr:{}".format(brr, drr))
    # print("H_P:{}".format(H_P))

    S_arr_last = torch.Tensor(S_arr_last).to('cuda:0')

    R_arr_last = torch.Tensor(R_arr_last).to('cuda:0')

    # print(S_arr_last)
    # print(R_arr_last)
    # print("H_P:", H_P.size())
    # print("brr:", brr.size())
    # print("drr:", drr.size())

    # print("Q", Q, {"R": H_P - R_arr_last, "S": H_P + S_arr_last},"H_P",H_P)

    return {"R": H_P - R_arr_last, "S": H_P + S_arr_last}


def fun(l_over, u_over, l_under, u_under):
    S_u_under = S(u_under)
    # print(u_under.size())
    S_l_under = S(l_under)
    S_l_over = S(l_over)
    S_u_over = S(u_over)
    S_p_u_under = S_p(u_under)
    S_p_l_under = S_p(l_under)
    S_p_l_over = S_p(l_over)
    S_p_u_over = S_p(u_over)
    k = (S_u_over - S_l_over) / (u_over - l_over)
    key_U = []
    key_L = []
    L = []
    U = []

    for i in range(len(k)):
        case_t = -1
        if S_p_l_over[i] < k[i] < S_p_u_over[i]:
            # print("----------case_1----------")

            d, d_y = find_intersection(u_over[i].numpy(), S_u_over[i].numpy(), 0)
            # print("d:{}, d_y:{}".format(d, d_y))
            key_U = [k[i], u_over[i], S_u_over[i]]
            # print("key_U:{}".format(key_U))
            if l_over[i] < d:
                # print("hell---------------1")

                key_L = [S_p_l_under[i], l_under[i], S_l_under[i]]

            else:

                key_L = [sigmoid_p(d), d, sigmoid(d)]
                # print("hell---------------2")
            # print("key_L:")
            # print(key_L)
            # print("key_U:")
            # print(key_U)
            # print()
        elif S_p_l_over[i] > k[i] > S_p_u_over[i]:
            # print("----------case_2----------")
            d, d_y = find_intersection(l_under[i].numpy(), S_l_under[i].numpy(), 0)
            key_L = [k[i], l_over[i], S_l_over[i]]
            if u_over[i] > d:

                key_U = [S_p_u_over[i], u_over[i], S_u_over[i]]
            else:

                key_U = [sigmoid_p(d), d, sigmoid(d)]
        else:
            # print("----------case_3----------")
            d1, d1_y = find_intersection(l_under[i].numpy(), S_l_under[i].numpy(), 0)

            if u_over[i] > d1:

                key_U = [S_p_u_under[i], u_under[i], S_u_under[i]]
            else:

                key_U = [sigmoid_p(d1), d1, sigmoid(d1)]

            d2, d2_y = find_intersection(u_over[i].numpy(), S_u_over[i].numpy(), 0)

            if u_over[i] > d2:

                key_L = [S_p_l_under[i], l_under[i], S_l_under[i]]
            else:

                key_L = [sigmoid_p(d2), d2, sigmoid(d2)]

        temp = []

        for i_t in key_L:
            if torch.is_tensor(i_t):
                temp.append(i_t.item())
            else:
                temp.append(i_t)

        L.append(temp)
        temp = []
        for i_t in key_U:
            if torch.is_tensor(i_t):
                temp.append(i_t.item())
            else:
                temp.append(i_t)

        U.append(temp)

    """-------------------"""
    # for i in range(len(k)):
    #     if S_p_l_over[i] < k[i] < S_p_u_over[i]:
    #         # print("----------case_1----------")
    #
    #         d, d_y = find_intersection(S_p_u_over[i], S_u_over[i] - S_p_u_over[i] * u_over[i])
    #         # print("d:{}, d_y:{}".format(d, d_y))
    #         key_U = [k[i], u_over[i], S_u_over[i]]
    #         # print("key_U:{}".format(key_U))
    #         if l_over[i] < d:
    #             # print("hell---------------1")
    #             key_L = [S_p_l_under[i], l_under[i], S_l_under[i]]
    #         else:
    #             key_L = [sigmoid_p(d), d, sigmoid(d)]
    #             # print("hell---------------2")
    #         # print("key_L:")
    #         # print(key_L)
    #         # print("key_U:")
    #         # print(key_U)
    #         # print()
    #     elif S_p_l_over[i] > k[i] > S_p_u_over[i]:
    #         # print("----------case_2----------")
    #         d, d_y = find_intersection(S_p_l_over[i], S_l_over[i] - S_p_l_over[i] * l_over[i])
    #         key_L = [k[i], l_under[i], S_l_under[i]]
    #         if u_over[i] > d:
    #             key_U = [S_p_u_over[i], u_over[i], S_u_over[i]]
    #         else:
    #             key_U = [sigmoid_p(d), d, sigmoid(d)]
    #     else:
    #         # print("----------case_3----------")
    #
    #         d1, d1_y = find_intersection(S_p_l_over[i], S_l_over[i] - S_p_l_over[i] * l_over[i])
    #
    #         if u_over[i] > d1:
    #             key_U = [S_p_u_over[i], u_over[i], S_u_over[i]]
    #         else:
    #             key_U = [sigmoid_p(d1), d1, sigmoid(d1)]
    #
    #         d2, d2_y = find_intersection(S_p_u_under[i], S_u_under[i] - S_p_u_under[i] * u_under[i])
    #
    #         if u_over[i] > d2:
    #             key_L = [S_p_l_over[i], l_over[i], S_l_over[i]]
    #         else:
    #             key_L = [sigmoid_p(d2), d2, sigmoid(d2)]
    #
    #     temp = []
    #
    #     for i in key_L:
    #         if torch.is_tensor(i):
    #             temp.append(i.item())
    #         else:
    #             temp.append(i)
    #     L.append(temp)
    #     temp = []
    #     for i in key_U:
    #         if torch.is_tensor(i):
    #             temp.append(i.item())
    #         else:
    #             temp.append(i)
    #
    #     U.append(temp)

    # for i in range()
    # print(U)
    return {"L": L, "U": U}


def fun_c(l_over, u_over, l_under, u_under):
    S_u_under = S(u_under)
    # print(u_under.size())
    S_l_under = S(l_under)
    S_l_over = S(l_over)
    S_u_over = S(u_over)
    S_p_u_under = S_p(u_under)
    S_p_l_under = S_p(l_under)
    S_p_l_over = S_p(l_over)
    S_p_u_over = S_p(u_over)
    k = (S_u_over - S_l_over) / (u_over - l_over)
    key_U = []
    key_L = []
    L = []
    U = []

    for i in range(len(k)):

        if S_p_l_over[i] < k[i] < S_p_u_over[i]:
            # print("----------case_1----------")
            key_L = [S_p_l_over[i], l_over[i], S_l_over[i]]
            key_U = [k[i], l_over[i], S_l_over[i]]


        elif S_p_l_over[i] > k[i] > S_p_u_over[i]:
            key_L = [k[i], l_over[i], S_l_over[i]]
            key_U = [S_p_u_over[i], u_over[i], S_u_over[i]]


        else:
            key_U = [S_p_u_over[i], u_over[i], S_u_over[i]]
            key_L = [S_p_l_over[i], l_over[i], S_l_over[i]]
        temp = []

        for i in key_L:
            if torch.is_tensor(i):
                temp.append(i.item())
            else:
                temp.append(i)
        L.append(temp)
        temp = []
        for i in key_U:
            if torch.is_tensor(i):
                temp.append(i.item())
            else:
                temp.append(i)

        U.append(temp)

        # for i in range()
    # print(U)
    return {"L": L, "U": U}


def common_sum(adj, R, S, m, W):
    # print(adj[0])
    # adj = slice_adj(adj, 2, m)
    R = R.unsqueeze(1).expand(-1, 34).t()

    S = S.unsqueeze(1).expand(-1, 34).t()

    # print("S")
    # print(S.size())
    #
    # print("R")
    # print(R.size())
    # print(R)
    # print("adj")
    # print(adj.size())
    zero = torch.zeros_like(W[0])
    W_pos = torch.where(W < 0, zero, W).t()
    W_neg = torch.where(W > 0, zero, W).t()

    # print("W")
    # print(W_pos.size())
    # # print("R")
    # print(R.size())
    # temp = torch.matmul(R, W_pos)
    # print(temp)
    #
    # print(temp.size())

    t_R = torch.matmul(adj, (torch.matmul(R, W_pos) - torch.matmul(S, W_neg)))
    t_S = torch.matmul(adj, (torch.matmul(S, W_pos) - torch.matmul(R, W_neg)))
    # print(t_R[m].size())

    return {"R": t_R[m], "S": t_S[m]}


def compute_active_bound(adj, x, activation, m, q, Q, W, l_under, u_under, layer, R=None, S=None):
    if S is None:
        S = []
    if R is None:
        R = []
    AD = normalize_digraph(adj)

    if layer == 0:
        g_arr = sum_top_Q(adj, x, AD, activation, m, q, Q, W)
        for i in range(len(g_arr["R"])):
            if g_arr["R"][i] > g_arr["S"][i]:
                print(g_arr)
    else:
        g_arr = common_sum(adj, R, S, m, W)
    l = [l_under, g_arr["R"].cpu()]
    u = [u_under, g_arr["S"].cpu()]
    # print(l[0].size())
    # print(l[1].size())
    #
    # print("l")
    # print(l)
    # print("u")
    # print(u)
    # print("ke")
    ke = fun(l[1], u[1], l[0], u[0])
    return {"ke": ke, "R": g_arr["R"], "S": g_arr["S"]}
