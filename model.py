import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv, RelGraphConv
from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel


class Embedding(nn.Module):
    """
    """

    def __init__(self, vocab_size=None, emb_size=None, emb_init=None, emb_trainable=True, padding_idx=0, dropout=0.2):
        super(Embedding, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def forward(self, inputs):
        emb_out = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != self.padding_idx).sum(dim=-1), inputs != self.padding_idx
        print('masks', type(masks), masks, masks.shape)
        return emb_out[:, :lengths.max()], lengths, masks[:, :lengths.max()]


class CNN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, num_class, embedding_dim=200):
        super(CNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz

        # self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.embedding_layer = Embedding(vocab_size=vocab_size, emb_size=embedding_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(len(self.ksz) * self.nKernel, num_class)

    def forward(self, input_seq):
        embedded_seq, _, masks = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        print('mask1', masks.shape, masks)
        embedded_seq = embedded_seq.unsqueeze(1)
        x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
        x_maxpool = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x_conv]  # len(Ks) * (bs, kernel_sz)
        x_concat = torch.cat(x_maxpool, 1)

        x = self.dropout(x_concat)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


# attention CNN with one kernel size
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

        self.conv = nn.Conv2d(1, nKernel, (ksz, embedding_dim))

        # self.transform = nn.Linear(nKernel, embedding_dim)
        # nn.init.xavier_uniform_(self.transform.weight)
        # nn.init.zeros_(self.transform.bias)

        self.content_final = nn.Linear(self.nKernel, embedding_dim * 2)

        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

    def forward(self, input_seq, g_node_feat):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        embedded_seq = self.dropout(embedded_seq)

        abstract_conv = F.relu(self.conv(embedded_seq)).squeeze(3)  # len(Ks) * (bs, kernel_sz, seq_len)
        print('x_conv', abstract_conv.shape)
        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract = torch.tanh(abstract_conv.transpose(1, 2))  # [bs, (n_words-ks+1), embedding_sz]
        print('abstract', abstract.shape)
        abstract_atten = torch.softmax(torch.matmul(abstract, g_node_feat.transpose(0, 1)), dim=1)
        print('atten', abstract_atten.shape)
        abstract_content = torch.matmul(abstract_conv, abstract_atten)
        print('abstract_after_atten', abstract_content.shape)
        x_feature = nn.functional.relu(self.content_final(abstract_content.transpose(1, 2)))
        print('x_feature', x_feature.shape)
        return x_feature


# attention CNN with multiple kernel size
# class attenCNN(nn.Module):
#     def __init__(self, vocab_size, nKernel, ksz, add_original_embedding, atten_dropout=0.2, embedding_dim=200):
#         super(attenCNN, self).__init__()
#
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.nKernel = nKernel
#         self.ksz = ksz
#         self.add_original_embedding = add_original_embedding
#         self.dropout = nn.Dropout(atten_dropout)
#
#         self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
#
#         self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])
#
#         self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim * 2)
#
#         nn.init.xavier_normal_(self.content_final.weight)
#         nn.init.zeros_(self.content_final.bias)
#
#     def forward(self, input_seq, g_node_feat):
#         embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
#         embedded_seq = embedded_seq.unsqueeze(1)
#         embedded_seq = self.dropout(embedded_seq)
#
#         x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
#         # label-wise attention (mapping different parts of the document representation to different labels)
#         x_doc = [torch.tanh(line.transpose(1, 2)) for line in x_conv] # [bs, (n_words-ks+1), embedding_sz]
#
#         atten = [torch.softmax(torch.matmul(x, g_node_feat.transpose(0, 1)), dim=1) for x in
#                  x_doc]  # []bs, (n_words-ks+1), n_labels]
#         x_content = [torch.matmul(x_conv[i], att) for i, att in enumerate(atten)]
#         x_concat = torch.cat(x_content, dim=1)
#
#         x_feature = nn.functional.relu(self.content_final(x_concat.transpose(1, 2)))
#         return x_feature

# multichannel attention CNN with multiple kernel size
# class multichannel_attenCNN(nn.Module):
#     def __init__(self, vocab_size, nKernel, ksz, add_original_embedding, atten_dropout=0.5, embedding_dim=200):
#         super(multichannel_attenCNN, self).__init__()
#
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.nKernel = nKernel
#         self.ksz = ksz
#         self.add_original_embedding = add_original_embedding
#         self.dropout = nn.Dropout(atten_dropout)
#
#         self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
#
#         self.convs = nn.ModuleList([nn.Conv2d(1, nKernel, (k, embedding_dim)) for k in ksz])
#
#         # just graph embedding
#         # if self.add_original_embedding:
#         #     self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim * 2)
#         # # concatenate graph embedding together with original MeSH embeddings
#         # else:
#         #     self.content_final = nn.Linear(len(self.ksz) * self.nKernel, embedding_dim)
#
#         self.content_final = nn.Linear(len(self.ksz) * self.nKernel * 2, embedding_dim * 2)
#
#         nn.init.xavier_normal_(self.content_final.weight)
#         nn.init.zeros_(self.content_final.bias)
#
#     def forward(self, input_seq, input_title, g_node_feat):
#         embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
#         embedded_seq = embedded_seq.unsqueeze(1)
#         embedded_seq = self.dropout(embedded_seq)
#
#         embedded_title = self.embedding_layer(input_title)  # size: (bs, seq_len, embed_dim)
#         embedded_title = embedded_title.unsqueeze(1)
#         embedded_title = self.dropout(embedded_title)
#
#         abstract_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in
#                          self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
#         title_conv = [F.relu(conv(embedded_title)).squeeze(3) for conv in self.convs]
#         print('conv', abstract_conv[0].shape, title_conv[0].shape)
#         # label-wise attention (mapping different parts of the document representation to different labels)
#         abstract = [torch.tanh(line.transpose(1, 2)) for line in
#                     abstract_conv]  # [bs, (n_words-ks+1), embedding_sz]
#         title = [torch.tanh(line.transpose(1, 2)) for line in
#                  title_conv]
#         print('content', abstract[0].shape, title[0].shape)
#
#         abstract_atten = [torch.softmax(torch.matmul(x, g_node_feat.transpose(0, 1)), dim=1) for x in
#                           abstract]  # []bs, (n_words-ks+1), n_labels]
#         title_atten = [torch.softmax(torch.matmul(x, g_node_feat.transpose(0, 1)), dim=1) for x in
#                        title]
#         print('atten', abstract_atten[0].shape, title_atten[0].shape)
#
#         abstract_content = [torch.matmul(abstract_conv[i], att) for i, att in enumerate(abstract_atten)]
#         title_content = [torch.matmul(title_conv[i], att) for i, att in enumerate(title_atten)]
#         print('content_feature', abstract_content[0].shape, title_content[0].shape)
#
#         ab_title_concat = [torch.cat((ab, title_content[i]), dim=1) for i, ab in enumerate(abstract_content)]
#         content_concat = torch.cat(ab_title_concat, dim=1)
#         print('concat', content_concat.shape)
#
#         x_feature = nn.functional.relu(self.content_final(content_concat.transpose(1, 2)))
#         print('x_feature', x_feature.shape)
#
#         return x_feature

# multichannel attention CNN with one kernel size
class multichannel_attenCNN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, add_original_embedding, atten_dropout=0.5, embedding_dim=200):
        super(multichannel_attenCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz
        self.add_original_embedding = add_original_embedding
        self.dropout = nn.Dropout(atten_dropout)

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.conv = nn.Conv2d(1, nKernel, (ksz, embedding_dim))

        self.content_final = nn.Linear(self.nKernel * 2, embedding_dim * 2)

        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

    def forward(self, input_seq, input_title, g_node_feat):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        embedded_seq = self.dropout(embedded_seq)

        embedded_title = self.embedding_layer(input_title)  # size: (bs, seq_len, embed_dim)
        embedded_title = embedded_title.unsqueeze(1)
        embedded_title = self.dropout(embedded_title)

        abstract_conv = F.relu(self.conv(embedded_seq)).squeeze(3)  # len(Ks) * (bs, kernel_sz, seq_len)
        title_conv = F.relu(self.conv(embedded_title)).squeeze(3)

        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract = torch.tanh(abstract_conv.transpose(1, 2))  # [bs, (n_words-ks+1), embedding_sz]
        title = torch.tanh(title_conv.transpose(1, 2))

        abstract_atten = torch.softmax(torch.matmul(abstract, g_node_feat.transpose(0, 1)), dim=1)
        title_atten = torch.softmax(torch.matmul(title, g_node_feat.transpose(0, 1)), dim=1)

        abstract_content = torch.matmul(abstract_conv, abstract_atten)
        title_content = torch.matmul(title_conv, title_atten)

        content_concat = torch.cat((abstract_content, title_content), dim=1)

        x_feature = nn.functional.relu(self.content_final(content_concat.transpose(1, 2)))

        return x_feature


class MLAttention(nn.Module):
    """
    """

    def __init__(self, labels_num, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, labels_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        attention = self.attention(inputs).transpose(1, 2).masked_fill(1.0 - masks, -(np.inf))  # N, labels_num, L
        attention = F.softmax(attention, -1)
        return attention


class Bert(BertPreTrainedModel):
    def __init__(self, config, num_label):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.atten = MLAttention(num_label, config.hidden_size)

        # self.fc1 = nn.Linear(config.hidden_size, 512)
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)
        #
        # self.fc2 = nn.Linear(512, 256)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)
        #
        # self.fc3 = nn.Linear(256, 1)
        # nn.init.xavier_normal_(self.fc3.weight)
        # nn.init.zeros_(self.fc3.bias)

    def forward(self, src_input_ids, src_attention_mask):
        output, _ = self.bert(src_input_ids, src_attention_mask)
        output = self.dropout(output)
        # print('output', output.shape)
        # output_transform = torch.relu(self.transform(output))
        # print('output_transform', output_transform.shape)  # [8, 512, 200]

        # atten = torch.softmax(torch.matmul(output, g_node_feat.transpose(0, 1)), dim=1)
        # content = torch.matmul(output.transpose(1, 2), atten)
        #
        # x_feature = nn.functional.tanh(self.fc1(content.transpose(1, 2)))
        # x_feature = nn.functional.tanh(self.fc2(x_feature))
        # x_feature = self.fc3(x_feature).squeeze(2)
        atten_out = self.atten(output, src_attention_mask)
        print('atten_out', atten_out.shape)

        x = torch.sigmoid(atten_out)
        return x


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
    def __init__(self, hidden_gcn_size, num_classes, in_node_features):
        super(LabelNet, self).__init__()
        self.gcn1 = GCNLayer(in_node_features, hidden_gcn_size)
        self.gcn2 = GCNLayer(hidden_gcn_size, num_classes)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        return x


class Baseline(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, atten_dropout=0.5, embedding_dim=200):
        super(Baseline, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz

        self.dropout = nn.Dropout(atten_dropout)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.conv = nn.Conv2d(1, nKernel, (ksz, embedding_dim))

        self.fc1 = nn.Linear(self.nKernel, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, input_seq, g_node_feat):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        embedded_seq = self.dropout(embedded_seq)

        abstract_conv = self.conv(embedded_seq).squeeze(3)  # [bs, (n_words-ks+1), embedding_sz]

        # label-wise attention (mapping different parts of the document representation to different labels)
        # abstract = torch.tanh(abstract_conv.transpose(1, 2))  # [bs, (n_words-ks+1), embedding_sz] with/without tanh

        abstract_atten = torch.softmax(torch.matmul(abstract_conv.transpose(1, 2), g_node_feat.transpose(0, 1)), dim=1)

        abstract_content = torch.matmul(abstract_conv, abstract_atten)
        print('abstract_content', abstract_content.shape)

        x_feature = nn.functional.tanh(self.fc1(abstract_content.transpose(1, 2)))
        print('fc1', x_feature.shape)
        x_feature = self.fc2(x_feature).squeeze(2)
        print('fc2', x_feature.shape)
        x = torch.sigmoid(x_feature)
        return x


class MeSH_GCN_Old(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, embedding_dim=200):
        super(MeSH_GCN_Old, self).__init__()
        # gcn_out = len(ksz) * nKernel

        self.cnn = CNN(vocab_size, nKernel, ksz, embedding_dim)
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

        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        x = torch.sum(x_feature * label_feature, dim=2)
        x = torch.sigmoid(x)
        return x


class MeSH_GCN_Multi(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, add_original_embedding, atten_dropout, output_size,
                 embedding_dim=200, cornet_dim=1000, n_cornet_blocks=2):
        super(MeSH_GCN_Multi, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.hidden_gcn_size = hidden_gcn_size
        self.add_original_embedding = add_original_embedding
        self.atten_dropout = atten_dropout

        self.content_feature = multichannel_attenCNN(self.vocab_size, self.nKernel, self.ksz,
                                                     self.add_original_embedding,
                                                     self.atten_dropout, embedding_dim=200)
        self.gcn = LabelNet(hidden_gcn_size, embedding_dim, embedding_dim)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, input_title, g_node_feature, g):
        x_feature = self.content_feature(input_seq, input_title, g_node_feature)

        label_feature = self.gcn(g, g_node_feature)

        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])
        x = torch.sum(x_feature * label_feature, dim=2)
        x = self.cornet(x)
        x = torch.sigmoid(x)
        return x


class Bert_GCN(nn.Module):
    def __init__(self, config):
        super(Bert_GCN, self).__init__()

        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.gcn = LabelNet(config.hidden_size, config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, g, g_node_feature):
        _, pooled_output = self.bert(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        x_feature = nn.functional.tanh(self.linear(pooled_output.squeeze(1)))  # [bz, bert_hidden_sz * 2] (8, 768 *2)

        label_feature = self.gcn(g, g_node_feature)  # [num_labels, hidden_sz] (29468, 768)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)
        # label_feature = self.linear(label_feature)
        # print('label2', label_feature.shape)

        x = torch.matmul(x_feature, label_feature.transpose(0, 1))
        x = torch.sigmoid(x)
        return x


class Bert_atten_GCN(nn.Module):
    def __init__(self, config, num_labels, gcn_hidden_gcn_size, embedding_dim=200):
        super(Bert_atten_GCN, self).__init__()

        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.transform = nn.Linear(config.hidden_size, embedding_dim)
        # nn.init.xavier_uniform_(self.transform.weight)
        # nn.init.zeros_(self.transform.bias)

        self.content_final = nn.Linear(config.hidden_size, embedding_dim * 2)
        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

        self.gcn = LabelNet(gcn_hidden_gcn_size, embedding_dim, embedding_dim)

    def forward(self, input_ids, attention_mask, g, g_node_feature):
        output, _ = self.bert(input_ids, attention_mask)
        output = self.dropout(output)
        # print('pooled', output.shape)

        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract = torch.tanh(self.transform(output))
        #print('output', output.shape)
        abstract_atten = torch.softmax(torch.matmul(abstract, g_node_feature.transpose(0, 1)), dim=1)
        #print('atten', abstract_atten.shape)
        abstract_content = torch.matmul(output.transpose(1, 2), abstract_atten)
        #print('abstract', abstract_content.shape)

        # match bert output size with graph output
        x_feature = self.content_final(abstract_content.transpose(1, 2))
        #print('x_feature', x_feature.shape)

        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)
        #print('label1', label_feature.shape)
        x = torch.sum(x_feature * label_feature, dim=2)
        #print('x', x.shape)
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


class Multi_RGCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_rgcn_size, output_size, add_original_embedding, atten_dropout,
                 embedding_dim=200, cornet_dim=1000, n_cornet_blocks=2):
        super(Multi_RGCN, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.add_original_embedding = add_original_embedding
        self.atten_dropout = atten_dropout

        self.content_feature = multichannel_attenCNN(self.vocab_size, self.nKernel, self.ksz,
                                                     self.add_original_embedding,
                                                     self.atten_dropout, embedding_dim=200)

        self.rgcn = EntityClassify(embedding_dim, hidden_rgcn_size, embedding_dim, num_rels=2, num_bases=-1,
                                   dropout=0, use_self_loop=False, use_cuda=True, low_mem=True)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, input_title, g, g_node_feature, edge_type, edge_norm):
        x_feature = self.content_feature(input_seq, input_title, g_node_feature)

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
