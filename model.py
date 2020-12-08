import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv, RelGraphConv

from transformers.modeling_bert import BertPreTrainedModel
from transformers import BertModel


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


class CNN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, num_class, embedding_dim=200):
        super(CNN, self).__init__()

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

        self.transform = nn.Linear(nKernel, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

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
        abstract = torch.tanh(self.transform(abstract_conv.transpose(1, 2)))  # [bs, (n_words-ks+1), embedding_sz]
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
#         self.transform = nn.Linear(nKernel, embedding_dim)
#         nn.init.xavier_uniform_(self.transform.weight)
#         nn.init.zeros_(self.transform.bias)
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
#         x_doc = [torch.tanh(self.transform(line.transpose(1, 2))) for line in
#                  x_conv]  # [bs, (n_words-ks+1), embedding_sz]
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
#         self.transform = nn.Linear(nKernel, embedding_dim)
#         nn.init.xavier_uniform_(self.transform.weight)
#         nn.init.zeros_(self.transform.bias)
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
#         abstract = [torch.tanh(self.transform(line.transpose(1, 2))) for line in
#                     abstract_conv]  # [bs, (n_words-ks+1), embedding_sz]
#         title = [torch.tanh(self.transform(line.transpose(1, 2))) for line in
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

        self.transform = nn.Linear(nKernel, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

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
        abstract = torch.tanh(self.transform(abstract_conv.transpose(1, 2)))  # [bs, (n_words-ks+1), embedding_sz]
        title = torch.tanh(self.transform(title_conv.transpose(1, 2)))

        abstract_atten = torch.softmax(torch.matmul(abstract, g_node_feat.transpose(0, 1)), dim=1)
        title_atten = torch.softmax(torch.matmul(title, g_node_feat.transpose(0, 1)), dim=1)

        abstract_content = torch.matmul(abstract_conv, abstract_atten)
        title_content = torch.matmul(title_conv, title_atten)

        content_concat = torch.cat((abstract_content, title_content), dim=1)

        x_feature = nn.functional.relu(self.content_final(content_concat.transpose(1, 2)))

        return x_feature


class Bert(BertPreTrainedModel):
    def __init__(self, config, d_model=768, num_d_heads=8, num_d_layer=6):
        super(Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_d_heads)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_d_layer, encoder_norm)

    def forward(self, src_input_ids, src_token_type_ids, src_attention_mask):
        _, pooled_output = self.bert(src_input_ids, src_token_type_ids, src_attention_mask)
        pooled_output = self.dropout(pooled_output)
        encoder_output = self.encoder(pooled_output.unsqueeze(1))
        print('encoder', encoder_output.shape)
        return encoder_output


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
        self.transform = nn.Linear(nKernel, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

        self.content_final = nn.Linear(self.nKernel, 1)

        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

    def forward(self, input_seq, g_node_feat):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        embedded_seq = self.dropout(embedded_seq)

        abstract_conv = F.relu(self.conv(embedded_seq)).squeeze(3)  # len(Ks) * (bs, kernel_sz, seq_len)

        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract = torch.tanh(self.transform(abstract_conv.transpose(1, 2)))  # [bs, (n_words-ks+1), embedding_sz]

        abstract_atten = torch.softmax(torch.matmul(abstract, g_node_feat.transpose(0, 1)), dim=1)

        abstract_content = torch.matmul(abstract_conv, abstract_atten)
        print('abstract_content', abstract_content.shape)

        x_feature = nn.functional.relu(self.content_final(abstract_content.transpose(1, 2))).squeeze(2)
        print('x_feature', x_feature.shape)
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
    def __init__(self, config, num_labels, gcn_hidden_gcn_size, embedding_dim=200):
        super(Bert_GCN, self).__init__()

        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.linear = nn.Linear(config.hidden_size, embedding_dim * 2)
        self.gcn = LabelNet(gcn_hidden_gcn_size, embedding_dim, embedding_dim)

        self.classifier = nn.Linear(config.hidden_size + embedding_dim * 2, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.linear = nn.Linear(embedding_dim * 2, 768)
    def forward(self, input_ids, attention_mask, g, g_node_feature):
        _, pooled_output = self.bert(input_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        print('pooled', pooled_output.shape)
        # x_feature = nn.functional.relu(self.linear(pooled_output.squeeze(1)))

        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)
        print('label1', label_feature.shape)
        # label_feature = self.linear(label_feature)
        # print('label2', label_feature.shape)

        x = torch.matmul(pooled_output, label_feature.transpose(0, 1))
        #print('final_feature', x.shape)
        # x = self.classifier(pooled_output)
        # x = torch.cat((pooled_output, label_feature.transpose(0, 1)), dim=1)
        # x = nn.functional.relu(self.linear(x))
        x = torch.sigmoid(x)
        return x


class Bert_atten_GCN(nn.Module):
    def __init__(self, config, num_labels, gcn_hidden_gcn_size, embedding_dim=200):
        super(Bert_atten_GCN, self).__init__()

        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.transform = nn.Linear(config.hidden_size, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)

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
