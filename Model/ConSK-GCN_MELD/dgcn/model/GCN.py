import torch.nn as nn
from torch_geometric.nn import RGCNConv, GraphConv
import torch
import torch.nn.functional as F


class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GCN, self).__init__()
        self.device = args["device"]
        self.num_relations = 2 * args["n_speakers"] ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)
        # self.lin=nn.Linear(g_dim+h2_dim, hidden_size)
        # self.drop = nn.Dropout(args["drop_rate"])

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        # 正向传播输入值，神经网络分析出输出值
        x = self.conv1(node_features, edge_index, edge_norm)
        x = self.conv2(x, edge_index)
        # x = torch.cat((x,node_features),dim=-1).to(self.device)
        # x = self.drop(F.relu(self.lin(x)))

        return x






















