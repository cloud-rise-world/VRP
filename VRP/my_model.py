import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter
import torch.nn as nn
from greatx.nn.layers import GCNConv, Sequential, activations
from greatx.utils import wrapper

EPS = 1e-10


class MyLayer(torch.nn.Module):
    def __init__(self, mask, add_self_loops=True):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.mask = mask

    def forward(self, x, edge_index):
        row, col = edge_index
        A, B = x[row], x[col]
        att_score = F.cosine_similarity(A, B)

        edge_index = edge_index[:, self.mask]
        att_score = att_score[self.mask]

        row, col = edge_index
        row_sum = scatter(att_score, col, dim_size=x.size(0))
        att_score_norm = att_score / (row_sum[row] + EPS)

        if self.add_self_loops:
            degree = scatter(torch.ones_like(att_score_norm), col, dim_size=x.size(0))
            self_weight = 1.0 / (degree + 1)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            loop_index = torch.arange(
                0, x.size(0), dtype=torch.long, device=edge_index.device
            )
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)

        att_score_norm = att_score_norm.exp()
        return edge_index, att_score_norm

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class MyModel(nn.Module):
    @wrapper
    def __init__(self, in_channels, hidden_feats, out_channels, normalize=True, bias=True):
        super().__init__()

        self.gconv = Sequential(
            GCNConv(
                in_channels,
                out_channels,
                add_self_loops=False,
                bias=bias,
                normalize=normalize,
            ),
            nn.Sigmoid(),
            # GCNConv(
            #     hidden_feats,
            #     out_channels,
            #     add_self_loops=False,
            #     bias=bias,
            #     normalize=normalize,
            # ),
            # nn.Sigmoid(),
        )

        # self.out = nn.Linear(7, 7)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = self.gconv(x, edge_index, edge_weight)
        x = x.view(x.size(0), -1)  # (batch_size,32*7*7)
        out = x

        return out

    def multiclass_hinge_loss(self, output, target, delta=1.0):
        """
        多分类铰链损失
        output: 模型输出，大小为 (N, C)，其中 N 是批量大小，C 是类别数
        target: 真实的类别标签，大小为 (N)
        delta: 边距值，默认为 1
        """
        n = output.size(0)
        correct_scores = output[torch.arange(n), target].unsqueeze(1) # 获取每个样本正确类别的得分
        margins = torch.clamp(output - correct_scores + delta, min=0) # 计算边距
        margins[torch.arange(n), target] = 0 # 正确类别的边距设置为0
        loss = margins.sum() / n
        return loss