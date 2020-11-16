from functools import partial

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import SAGEConv, RelGraphConv
import dgl.nn as dglnn

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
    def __init__(self, vocab_size, nKernel, ksz, add_original_embedding, atten_dropout=0.5, embedding_dim=200):
        super(attenCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz
        self.add_original_embedding = add_original_embedding
        self.dropout = nn.Dropout(atten_dropout)

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])

        self.transform = nn.Linear(nKernel, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

        # just graph embedding
        # if self.add_original_embedding:
        #     self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim * 2)
        # # concatenate graph embedding together with original MeSH embeddings
        # else:
        #     self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim)

        self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim * 2)

        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

    def forward(self, input_seq, g_node_feat):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        embedded_seq = self.dropout(embedded_seq)

        x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
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
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, add_original_embedding, atten_dropout,
                 embedding_dim=200):
        super(MeSH_GCN, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.hidden_gcn_size = hidden_gcn_size
        self.add_original_embedding = add_original_embedding
        self.atten_dropout = atten_dropout

        self.content_feature = attenCNN(self.vocab_size, self.nKernel, self.ksz, self.add_original_embedding,
                                        self.atten_dropout, embedding_dim=200)
        self.gcn = LabelNet(hidden_gcn_size, embedding_dim, embedding_dim)

    def forward(self, input_seq, g, g_node_feature):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.gcn(g, g_node_feature)
        # if self.add_original_embedding:
        #     label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        # print('concat', label_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        x = torch.sum(x_feature * label_feature, dim=2)
        x = torch.sigmoid(x)
        return x


class CorGCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, output_size, add_original_embedding, atten_dropout,
                 embedding_dim=200, cornet_dim=1000, n_cornet_blocks=2):
        super(CorGCN, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.hidden_gcn_size = hidden_gcn_size
        self.output_size = output_size
        self.add_original_embedding = add_original_embedding
        self.atten_dropout = atten_dropout

        self.content_feature = attenCNN(self.vocab_size, self.nKernel, self.ksz, self.add_original_embedding,
                                        self.atten_dropout, embedding_dim=200)
        self.gcn = LabelNet(hidden_gcn_size, embedding_dim, embedding_dim)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, g_node_feature, g):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        # if self.add_original_embedding:
        #     label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        #     print('concat', label_feature.shape)
        # else:
        #     None

        x = torch.sum(x_feature * label_feature, dim=2)
        cor_logit = self.cornet(x)
        cor_logit = torch.sigmoid(cor_logit)
        return cor_logit


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels=2, num_bases=-1,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=True, low_mem=True):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.low_mem = low_mem

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
                            dropout=self.dropout, low_mem=self.low_mem)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                            dropout=self.dropout, low_mem=self.low_mem)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                            self.num_bases, activation=None,
                            self_loop=self.use_self_loop, low_mem=self.low_mem)


class MeSH_RGCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_rgcn_size, add_original_embedding, atten_dropout,
                 embedding_dim=200):
        super(MeSH_RGCN, self).__init__()
        self.add_original_embedding = add_original_embedding
        self.embedding_dim = embedding_dim
        self.atten_dropout = atten_dropout

        self.content_feature = attenCNN(vocab_size, nKernel, ksz, self.add_original_embedding, self.atten_dropout,
                                        embedding_dim=self.embedding_dim)

        self.rgcn = EntityClassify(embedding_dim, hidden_rgcn_size, embedding_dim, num_rels=2, num_bases=-1,
                                   dropout=0, use_self_loop=False, use_cuda=True, low_mem=True)

    def forward(self, input_seq, g, g_node_feature, edge_type, edge_norm):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.rgcn(g, g_node_feature, edge_type, edge_norm)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        # if self.add_original_embedding:
        #     label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])

        x = torch.sum(x_feature * label_feature, dim=2)
        x = torch.sigmoid(x)
        return x


class CorRGCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_rgcn_size, output_size, add_original_embedding, atten_dropout,
                 embedding_dim=200, cornet_dim=1000, n_cornet_blocks=2):
        super(CorRGCN, self).__init__()
        self.add_original_embedding = add_original_embedding
        self.atten_dropout = atten_dropout

        self.content_feature = attenCNN(vocab_size, nKernel, ksz, self.add_original_embedding, self.atten_dropout,
                                        embedding_dim)

        self.rgcn = EntityClassify(embedding_dim, hidden_rgcn_size, embedding_dim, num_rels=2, num_bases=-1,
                                   dropout=0, use_self_loop=False, use_cuda=True, low_mem=True)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, g, g_node_feature, edge_type, edge_norm):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.rgcn(g, g_node_feature, edge_type, edge_norm)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        # if self.add_original_embedding:
        #     label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])

        x = torch.sum(x_feature * label_feature, dim=2)

        cor_logit = self.cornet(x)
        cor_logit = torch.sigmoid(cor_logit)
        return cor_logit


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers=1,
                 activation=F.relu,
                 dropout=0.5,
                 aggregator_type='lstm'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))  # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class CorGraphSage(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_graphsage_size, output_size, embedding_dim=200, cornet_dim=1000,
                 n_cornet_blocks=2, model='GCN'):
        super(CorGraphSage, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.output_size = output_size
        self.model = model

        self.content_feature = attenCNN(vocab_size, nKernel, ksz, embedding_dim, model=self.model)
        self.graphsage = GraphSAGE(embedding_dim, hidden_graphsage_size * 2, embedding_dim * 2)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, g_node_feature, g):
        x_feature = self.content_feature(input_seq, g_node_feature)
        # print('feat', g_node_feature.shape)
        label_feature = self.graphsage(g, g_node_feature)
        # print('x_feature', x_feature, x_feature.shape)
        # print('label', label_feature, label_feature.shape)
        x = torch.sum(x_feature * label_feature, dim=2)
        # print('x', x, x.shape)
        cor_logit = self.cornet(x)
        cor_logit = torch.sigmoid(cor_logit)
        return cor_logit
