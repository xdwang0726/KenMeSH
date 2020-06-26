import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
import dgl.function as fn
#from torch_geometric.nn import GCNConv


class Embeddings_OOV(torch.nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, dim)
        self.embedding.weight.requires_grad = False
        # vector for oov
        self.oov = torch.nn.Parameter(data=torch.rand(1, dim))
        self.oov_index = -1
        self.dim = dim

    def forward(self, arr):
        N = arr.shape[0]
        mask = (arr == self.oov_index).long()
        mask_ = mask.unsqueeze(dim=1).float()
        embed = (1-mask_) * self.embedding((1 - mask) * arr) + mask_ * (self.oov.expand((N, self.dim)))
        return embed


class ContentsExtractor(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, embedding_dim=200):
        super(ContentsExtractor, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])

    def forward(self, input_seq):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)

        embedded_seq = embedded_seq.unsqueeze(1)
        x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
        x_maxpool = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x_conv]  # len(Ks) * (bs, kernel_sz)

        x_concat = torch.cat(x_maxpool, 1)

        return x_concat


# Using PyTorch Geometric
# class LabelNet(nn.Module):
#     def __init__(self, node_features, hiddern_gcn, num_classes):
#         super(LabelNet, self).__init__()
#         self.gcn1 = GCNConv(node_features, hiddern_gcn)
#         self.gcn2 = GCNConv(hiddern_gcn, num_classes)
#
#     def forward(self, data):
#         nodes, edge_index = data.x, data.edge_index
#
#         x = self.gcn1(nodes, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.gcn2(x, edge_index)
#         return x

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        """
        inputs: g,       object of Graph
                feature, node features
        """
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class LabelNet(nn.Module):
    def __init__(self, node_features, hidden_gcn_size, num_classes):
        super(LabelNet, self).__init__()
        self.gcn1 = GCNLayer(node_features, hidden_gcn_size)
        self.gcn2 = GCNLayer(hidden_gcn_size, num_classes)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        return x


class MeSH_GCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, node_features, hidden_gcn_size, num_classes, dropout_rate,
                 embedding_dim=200):
        super(MeSH_GCN, self).__init__()
        self.cnn = ContentsExtractor(vocab_size, nKernel, ksz, embedding_dim)
        self.gcn = LabelNet(node_features, hidden_gcn_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_seq, data):
        x_feature = self.cnn(input_seq)
        label_feature = self.gcn(data)
        concat = torch.cat((x_feature, label_feature), dim=0)
        concat = self.dropout(concat)
        x = F.log_softmax(concat, dim=1)
        return x

