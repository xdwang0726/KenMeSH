from functools import partial

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
from dgl.nn.pytorch.conv import SAGEConv

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
    def __init__(self, vocab_size, nKernel, ksz, num_class, embedding_dim=200):
        super(ContentsExtractor, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(len(self.ksz) * self.nKernel, num_class)

    def forward(self, input_seq):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)

        embedded_seq = embedded_seq.unsqueeze(1)
        x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
        x_maxpool = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x_conv]  # len(Ks) * (bs, kernel_sz)
        x_concat = torch.cat(x_maxpool, 1)

        x = self.dropout(x_concat)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class attenCNN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, embedding_dim=200):
        super(attenCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])

        self.transform = nn.Linear(nKernel, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

        self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim * 2)
        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

    def forward(self, input_seq, g_node_feat):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
        # x_conv = [F.relu(conv(embedded_seq)) for conv in self.convs]

        # label-wise attention (mapping different parts of the document representation to different labels)
        x_doc = [torch.tanh(self.transform(line.transpose(1, 2))) for line in
                 x_conv]  # [bs, (n_words-ks+1), embedding_sz]

        atten = [torch.softmax(torch.matmul(x, g_node_feat.transpose(0, 1)), dim=1) for x in
                 x_doc]  # []bs, (n_words-ks+1), n_labels]
        x_content = [torch.matmul(x_conv[i], att) for i, att in enumerate(atten)]
        x_concat = torch.cat(x_content, dim=1)

        x_feature = nn.functional.relu(self.content_final(x_concat.transpose(1, 2)))
        return x_feature


class CorNetBlock(nn.Module):
    def __init__(self, context_size, output_size):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)
        self.act_fn = torch.sigmoid

    def forward(self, output_dstrbtn):
        identity_logits = output_dstrbtn
        output_dstrbtn = self.act_fn(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn


class CorNet(nn.Module):
    def __init__(self, output_size, cornet_dim=1000, n_cornet_blocks=2):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [CorNetBlock(cornet_dim, output_size) for _ in range(n_cornet_blocks)])
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits


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
    def __init__(self, hidden_gcn_size, num_classes, in_node_features=200):
        super(LabelNet, self).__init__()
        self.gcn1 = GCNLayer(in_node_features, hidden_gcn_size)
        self.gcn2 = GCNLayer(hidden_gcn_size, num_classes)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        # print('gcn_shape', x.shape)
        # print('embedding_shape', g.ndata['feat'].shape)

        # concat MeSH embeddings together with GCN result
        x = torch.cat([x, g.ndata['feat']], dim=1)
        return x


class MeSH_GCN_Old(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, embedding_dim=200):
        super(MeSH_GCN_Old, self).__init__()
        # gcn_out = len(ksz) * nKernel

        self.cnn = ContentsExtractor(vocab_size, nKernel, ksz, embedding_dim)
        self.gcn = LabelNet(hidden_gcn_size, embedding_dim * 2, embedding_dim)

    def forward(self, input_seq, g, features):
        x_feature = self.cnn(input_seq)
        label_feature = self.gcn(g, features)
        label_feature = torch.transpose(label_feature, 0, 1)
        x = torch.matmul(x_feature, label_feature)
        x = torch.sigmoid(x)
        return x


class MeSH_GCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, embedding_dim=200):
        super(MeSH_GCN, self).__init__()
        # gcn_out = len(ksz) * nKernel

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])
        # self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, nKernel, k) for k in ksz)

        self.transform = nn.Linear(nKernel, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

        self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim * 2)
        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

        self.gcn = LabelNet(hidden_gcn_size, embedding_dim, embedding_dim)

    def forward(self, input_seq, g, features):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        # print('embedding', embedded_seq.shape)
        embedded_seq = embedded_seq.unsqueeze(1)
        # print('embedding2', embedded_seq.shape)
        x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
        # x_conv = [F.relu(conv(embedded_seq)) for conv in self.convs]
        # print(x_conv[0].shape, x_conv[1].shape, x_conv[2].shape)
        # label-wise attention (mapping different parts of the document representation to different labels)
        # print('w', self.transform.weight.shape)
        # print('b', self.transform.bias.shape)
        x_doc = [torch.tanh(self.transform(line.transpose(1, 2))) for line in
                 x_conv]  # [bs, (n_words-ks+1), embedding_sz]
        # print("x", x_doc[0].shape, x_doc[1].shape, x_doc[2].shape)

        atten = [torch.softmax(torch.matmul(x, g.ndata['feat'].transpose(0, 1)), dim=1) for x in
                 x_doc]  # []bs, (n_words-ks+1), n_labels]
        # print('atten', atten[0].shape, atten[1].shape, atten[2].shape)

        x_content = [torch.matmul(x_conv[i], att) for i, att in enumerate(atten)]
        # print('x_content', x_content[0].shape, x_content[1].shape, x_content[2].shape)

        x_concat = torch.cat(x_content, dim=1)
        # print('x_concat', x_concat.shape)

        x_feature = nn.functional.relu(self.content_final(x_concat.transpose(1, 2)))
        # print('x_feature', x_feature.shape)

        label_feature = self.gcn(g, features)
        # print('label', label_feature.shape)
        # label_feature = torch.transpose(label_feature, 0, 1)

        # print('label2', label_feature.shape)

        # def element_wise_mul(m1, m2):
        #     m1.to('cpu')
        #     m2.to('cpu')
        #     result = torch.zeros(0).to('cpu')
        #     for i in range(m1.shape[1]):
        #         v1 = m1[:, i, :]
        #         v2 = m2[:, i]
        #         v = torch.matmul(v1, v2).unsqueeze(1).to('cpu')
        #         print('v', v.device)
        #         print('result', result.device)
        #         result = torch.cat((result, v), dim=1)
        #         #print('result', result.device)
        #     result.to('cuda')
        #     return result

        # x = element_wise_mul(x_feature, label_feature)
        # x = torch.diagonal(torch.matmul(x_feature, label_feature), offset=0).transpose(0, 1)
        x = torch.sum(x_feature * label_feature, dim=2)
        # print('x_final', x.shape)
        x = torch.sigmoid(x)
        return x


class CorGCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, output_size, embedding_dim=200, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(CorGCN, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.hidden_gcn_size = hidden_gcn_size
        self.output_size = output_size

        self.content_feature = attenCNN(vocab_size, nKernel, ksz, embedding_dim=200)
        self.gcn = LabelNet(hidden_gcn_size, embedding_dim, embedding_dim)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, g_node_feature, g):
        x_feature = self.content_feature(input_seq, g_node_feature)
        label_feature = self.gcn(g, g_node_feature)
        x = torch.sum(x_feature * label_feature, dim=2)
        cor_logit = self.cornet(x)
        cor_logit = torch.sigmoid(cor_logit)
        return cor_logit


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                            self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                edges_src = edges.src['id'].to('cuda')
                print('edge_data', len(edges.data['rel_type']))
                index = edges.data['rel_type'] * self.in_feat + edges_src
                return {'msg': embed[index] * edges.data['norm']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels=2, num_bases=0,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=True):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        # h is node feature
        # r is edge type
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class EntityClassify(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.num_nodes, self.num_rels, "basis",
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop)


class MeSH_RGCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, num_nodes, embedding_dim=200):
        super(MeSH_RGCN, self).__init__()

        self.content_feature = attenCNN(vocab_size, nKernel, ksz, embedding_dim)

        self.rgcn = EntityClassify(num_nodes, hidden_gcn_size)

    def forward(self, input_seq, g_node_feature, g):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.rgcn(g, g.ndata['feat'], g.edata['rel_type'], g.edata['norm'])
        print('label', label_feature.shape)

        x = torch.sum(x_feature * label_feature, dim=2)
        print('x_final', x.shape)
        x = torch.sigmoid(x)
        return x


class CorGraphSage(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, output_size, embedding_dim=200, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(CorGraphSage, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.output_size = output_size

        self.content_feature = attenCNN(vocab_size, nKernel, ksz, embedding_dim)
        self.graphsage = SAGEConv(embedding_dim, embedding_dim * 2, aggregator_type='pool')
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, g_node_feature, g):
        x_feature = self.content_feature(input_seq, g_node_feature)
        # print('feat', g_node_feature.shape)
        label_feature = self.graphsage(g, g_node_feature)
        # print('x_feature', x_feature.shape)
        # print('label', label_feature.shape)
        x = torch.sum(x_feature * label_feature, dim=2)
        cor_logit = self.cornet(x)
        cor_logit = torch.sigmoid(cor_logit)
        return cor_logit
