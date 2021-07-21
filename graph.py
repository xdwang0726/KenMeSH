import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv


def edge_attention(self, edges):
    # edge UDF for equation (2)
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e' : F.leaky_relu(a)}


def reduce_func(self, nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h' : h}


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


# class GAT(nn.Module):
#     def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
#         super(GAT, self).__init__()
#         g = g.to('cuda')
#         self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
#         # Be aware that the input dimension is hidden_dim*num_heads since
#         #   multiple head outputs are concatenated together. Also, only
#         #   one attention head in the output layer.
#         self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
#
#     def forward(self, h):
#         h = self.layer1(h)
#         print(h.device)
#         h = F.elu(h)
#         h = self.layer2(h)
#         return h

class GAT(nn.Module):
    def __init__(self, in_node_feats, hidden_gat_size, num_classes, num_heads=1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_node_feats, hidden_gat_size, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False,
                            activation=None, allow_zero_in_degree=True, bias=True)
        self.gat2 = GATConv(hidden_gat_size, num_classes, num_heads, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, residual=False,
                            activation=None, allow_zero_in_degree=True, bias=True)

    def forward(self, g, features):
        x = self.gat1(g, features)
        x = x.squeeze(dim=1)
        # print('gat1', x.shape)
        x = F.elu(x)
        x = self.gat2(g, x)
        return x.squeeze(dim=1)

