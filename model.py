import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv, RelGraphConv
from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gat import GAT

########## Embedding ##########
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
        # lengths, masks = (inputs != self.padding_idx).sum(dim=-1), inputs != self.padding_idx
        return emb_out  # emb_out[:, :lengths.max()], lengths, masks[:, :lengths.max()]


########## CNN ###########
class CNN_Baseline(nn.Module):
    """
    CNN baseline with 3 kernel size
    """
    def __init__(self, vocab_size, nKernel, ksz, num_class, embedding_dim=200):
        super(CNN_Baseline, self).__init__()

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
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        x_conv = [F.relu(conv(embedded_seq)).squeeze(3) for conv in self.convs]  # len(Ks) * (bs, kernel_sz, seq_len)
        x_maxpool = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x_conv]  # len(Ks) * (bs, kernel_sz)
        x_concat = torch.cat(x_maxpool, 1)

        x = self.dropout(x_concat)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class Baseline(nn.Module):
    """
    cnn with label-wise attention
    """

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
        print('embed', embedded_seq.shape)
        embedded_seq = embedded_seq.unsqueeze(1)
        print('embed_unsqueeze', embedded_seq.shape)
        embedded_seq = self.dropout(embedded_seq)

        abstract_conv = self.conv(embedded_seq).squeeze(3)  # [bs, (n_words-ks+1), embedding_sz]
        print('cov', abstract_conv.shape)

        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract_atten = torch.softmax(torch.matmul(abstract_conv.transpose(1, 2), g_node_feat.transpose(0, 1)), dim=1)
        abstract_content = torch.matmul(abstract_conv, abstract_atten)  # [bs, embed]
        # print('abstract_content', abstract_content.shape)

        x_feature = nn.functional.tanh(self.fc1(abstract_content.transpose(1, 2)))
        # print('x_feature', x_feature.shape)
        x_feature = self.fc2(x_feature).squeeze(2)
        # print('x', x_feature.shape)
        x = torch.sigmoid(x_feature)
        return x


class attenCNN(nn.Module):
    """
    label-wise attention CNN with one kernel size
    """

    def __init__(self, vocab_size, nKernel, ksz, atten_dropout=0.5, embedding_dim=200):
        super(attenCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.nKernel = nKernel
        self.ksz = ksz
        self.dropout = nn.Dropout(atten_dropout)

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.conv = nn.Conv2d(1, nKernel, (ksz, embedding_dim))

        self.content_final = nn.Linear(self.nKernel, embedding_dim * 2)

        nn.init.xavier_normal_(self.content_final.weight)
        nn.init.zeros_(self.content_final.bias)

    def forward(self, input_seq, g_node_feat):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)
        embedded_seq = embedded_seq.unsqueeze(1)
        embedded_seq = self.dropout(embedded_seq)

        abstract_conv = F.relu(self.conv(embedded_seq)).squeeze(3)  # [bs, (n_words-ks+1), embedding_sz]

        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract_atten = torch.softmax(torch.matmul(abstract_conv.transpose(1, 2), g_node_feat.transpose(0, 1)), dim=1)
        abstract_content = torch.matmul(abstract_conv, abstract_atten) # (bs, emb_sz, num_label)

        x_feature = nn.functional.tanh(self.content_final(abstract_content.transpose(1, 2))) # (bs, num_label, emb_sz*2)
        return x_feature


class multichannel_attenCNN(nn.Module):
    """
    multichannel attention CNN with one kernel size
    """

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

        abstract_conv = F.relu(self.conv(embedded_seq)).squeeze(3)  # [bs, (n_words-ks+1), embedding_sz]
        title_conv = F.relu(self.conv(embedded_title)).squeeze(3)

        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract_atten = torch.softmax(torch.matmul(abstract_conv.transpose(1, 2), g_node_feat.transpose(0, 1)), dim=1)
        title_atten = torch.softmax(torch.matmul(title_conv.transpose(1, 2), g_node_feat.transpose(0, 1)), dim=1)

        abstract_content = torch.matmul(abstract_conv, abstract_atten)
        title_content = torch.matmul(title_conv, title_atten)

        content_concat = torch.cat((abstract_content, title_content), dim=1)

        x_feature = nn.functional.tanh(self.content_final(content_concat.transpose(1, 2)))
        return x_feature


class single_channel_dilatedCNN(nn.Module):
    def __init__(self, vocab_size, dropout, ksz, output_size, embedding_dim=200, rnn_num_layers=2, cornet_dim=1000, n_cornet_blocks=2):
        super(single_channel_dilatedCNN, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=rnn_num_layers,
                           dropout=self.dropout, bidirectional=True, batch_first=True)

        self.dconv = nn.Sequential(nn.Conv1d(self.embedding_dim * 2, self.embedding_dim * 2, kernel_size=self.ksz, padding=0, dilation=1),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim * 2, self.embedding_dim * 2, kernel_size=self.ksz, padding=0, dilation=2),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim * 2, self.embedding_dim * 2, kernel_size=self.ksz, padding=0, dilation=3),
                                   nn.SELU(), nn.AlphaDropout(p=0.05))

        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)
        # corNet
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, input_length, mask, g, g_node_feature):
        embedded_seq = self.embedding_layer(input_seq)  # size: (bs, seq_len, embed_dim)

        packed_seq = pack_padded_sequence(embedded_seq, input_length, batch_first=True, enforce_sorted=False)
        packed_output, (_,_) = self.rnn(packed_seq)
        outputs, _ = pad_packed_sequence(packed_output, batch_first=True)  # (bs, seq_len, emb_dim*2)

        outputs = outputs.permute(0, 2, 1) # (bs, emb_dim*2, seq_length)
        outputs = self.dconv(outputs)  # (bs, embed_dim*2, seq_len-ksz+1)

        # get label features
        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 200*2])
        atten_mask = label_feature.transpose(0, 1) * mask.unsqueeze(1)

        # label-wise attention (mapping different parts of the document representation to different labels)
        abstract_atten = torch.softmax(torch.matmul(outputs.transpose(1, 2), atten_mask), dim=1)
        x_feature = torch.matmul(outputs, abstract_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        x_feature = torch.sum(x_feature * label_feature, dim=2)

        # CorNet
        x_feature = self.cornet(x_feature)
        return x_feature


class multichannel_dilatedCNN(nn.Module):
    def __init__(self, vocab_size, dropout, ksz, output_size, G, device, embedding_dim=200, rnn_num_layers=2, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(multichannel_dilatedCNN, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=rnn_num_layers,
                           dropout=self.dropout, bidirectional=True, batch_first=True)

        self.dconv = nn.Sequential(nn.Conv1d(self.embedding_dim*2, self.embedding_dim*2, kernel_size=self.ksz, padding=0, dilation=1),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim*2, self.embedding_dim*2, kernel_size=self.ksz, padding=0, dilation=2),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim*2, self.embedding_dim*2, kernel_size=self.ksz, padding=0, dilation=3),
                                   nn.SELU(), nn.AlphaDropout(p=0.05))

        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)

        # corNet
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_abstract, input_title, ab_length, title_length, g, g_node_feature): #g_c, g_node_feature_c):
        # get label features
        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)

        # get title content features
        title = self.embedding_layer(input_title.long())
        title = pack_padded_sequence(title, title_length, batch_first=True, enforce_sorted=False)

        title, (_,_) = self.rnn(title)
        title, _ = pad_packed_sequence(title, batch_first=True)  # (bs, seq_len, emb_dim*2)

        title_atten = torch.softmax(torch.matmul(title, label_feature.transpose(0, 1)), dim=1)
        title_feature = torch.matmul(title.transpose(1, 2), title_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get abstract content features
        abstract = self.embedding_layer(input_abstract)  # size: (bs, seq_len, embed_dim)
        abstract = pack_padded_sequence(abstract, ab_length, batch_first=True, enforce_sorted=False)
        abstract, (_,_) = self.rnn(abstract)
        abstract, _ = pad_packed_sequence(abstract, batch_first=True)  # (bs, seq_len, emb_dim*2)

        abstract = abstract.permute(0, 2, 1) # (bs, emb_dim*2, seq_length)
        abstract_conv = self.dconv(abstract)  # (bs, embed_dim*2, seq_len-ksz+1)
        abstract_atten = torch.softmax(torch.matmul(abstract_conv.transpose(1, 2), label_feature.transpose(0, 1)), dim=1)  # size: (bs, seq_len-ksz+1, 29368)
        abstract_feature = torch.matmul(abstract_conv, abstract_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get document feature
        x_feature = title_feature + abstract_feature  # size: (bs, 29368, embed_dim*2)
        x_feature = torch.sum(x_feature * label_feature, dim=2)

        # add CorNet
        x_feature = self.cornet(x_feature)
        return x_feature


class multichannel_dilatedCNN_with_MeSH_mask(nn.Module):
    def __init__(self, vocab_size, dropout, ksz, output_size, G, device, embedding_dim=200, rnn_num_layers=2, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(multichannel_dilatedCNN_with_MeSH_mask, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=rnn_num_layers,
                           dropout=self.dropout, bidirectional=True, batch_first=True)

        self.dconv = nn.Sequential(nn.Conv1d(self.embedding_dim * 2, self.embedding_dim * 2, kernel_size=self.ksz, padding=0, dilation=1),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim * 2, self.embedding_dim * 2, kernel_size=self.ksz, padding=0, dilation=2),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim * 2, self.embedding_dim * 2, kernel_size=self.ksz, padding=0, dilation=3),
                                   nn.SELU(), nn.AlphaDropout(p=0.05))

        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)
        # self.gat = GAT(embedding_dim, embedding_dim, embedding_dim)
        # heads = ([gat_num_heads] * gat_num_layers) + [gat_num_out_heads]
        # heads = ([4] * 2) + [1]
        # self.gat = GAT(device, G, num_layers=2, in_node_feats=embedding_dim, hidden_gat_size=embedding_dim,
        #                num_classes=embedding_dim, heads=heads)
        # linear
        # self.linear = nn.Linear(self.embedding_dim * 2, 1)

        # corNet
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, abstract, title, mask, ab_length, title_length, g, g_node_feature): #g_c, g_node_feature_c):
        # get label features
        label_feature = self.gcn(g, g_node_feature)
        # label_feature = self.gat(g_node_feature)
        # label_cooccurence_feature = self.gcn(g_c, g_node_feature_c)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1) # torch.Size([29368, 200*2])
        # print('label_feature', label_feature.shape)
        # label_feature = torch.cat((label_feature, label_cooccurence_feature), dim=1)  # torch.Size([29368, 200*2])

        # get title content features
        atten_mask = label_feature.transpose(0, 1) * mask.unsqueeze(1)

        title = self.embedding_layer(title.long())
        title = pack_padded_sequence(title, title_length, batch_first=True, enforce_sorted=False) # packed input title

        output_title, (_,_) = self.rnn(title) # packed rnn output title
        output_title, _ = pad_packed_sequence(output_title, batch_first=True)  # unpacked rnn output title with size: (bs, seq_len, emb_dim*2)
        alpha_title = torch.softmax(torch.matmul(output_title, atten_mask), dim=1)
        title_features = torch.matmul(output_title.transpose(1, 2), alpha_title).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get abstract content features
        abstract = self.embedding_layer(abstract)  # size: (bs, seq_len, embed_dim)
        abstract = pack_padded_sequence(abstract, ab_length, batch_first=True, enforce_sorted=False)
        output_abstract, (_,_) = self.rnn(abstract)
        output_abstract, _ = pad_packed_sequence(output_abstract, batch_first=True)  # (bs, seq_len, emb_dim*2)

        output_abstract = self.dconv(output_abstract.permute(0, 2, 1))  # (bs, embed_dim*2, seq_len-ksz+1)
        alpha_abstract = torch.softmax(torch.matmul(output_abstract.transpose(1, 2), atten_mask), dim=1)  # size: (bs, seq_len-ksz+1, 29368)
        abstract_features = torch.matmul(output_abstract, alpha_abstract).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get document feature
        x_feature = title_features + abstract_features  # size: (bs, 29368, embed_dim*2)
        x_feature = torch.sum(x_feature * label_feature, dim=2)
        # x_feature = torch.sum(x_feature * (atten_mask.transpose(1, 2)), dim=2)

        # add CorNet
        x_feature = self.cornet(x_feature)
        return x_feature


class multichannel_with_MeSH_mask(nn.Module):
    def __init__(self, vocab_size, dropout, ksz, output_size, G, device, embedding_dim=200, rnn_num_layers=2, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(multichannel_with_MeSH_mask, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=rnn_num_layers,
                           dropout=self.dropout, bidirectional=True, batch_first=True)

        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)

        # corNet
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, abstract, title, mask, ab_length, title_length, g, g_node_feature):
        # get label features
        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1) # torch.Size([29368, 200*2])

        # get title content features
        atten_mask = label_feature.transpose(0, 1) * mask.unsqueeze(1)
        title = self.embedding_layer(title.long())
        title = pack_padded_sequence(title, title_length, batch_first=True, enforce_sorted=False) # packed input title

        output_title, (_,_) = self.rnn(title) # packed rnn output title
        output_title, _ = pad_packed_sequence(output_title, batch_first=True)  # unpacked rnn output title with size: (bs, seq_len, emb_dim*2)

        alpha_title = torch.softmax(torch.matmul(output_title, atten_mask), dim=1)
        title_features = torch.matmul(output_title.transpose(1, 2), alpha_title).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get abstract content features
        abstract = self.embedding_layer(abstract)  # size: (bs, seq_len, embed_dim)
        abstract = pack_padded_sequence(abstract, ab_length, batch_first=True, enforce_sorted=False)
        output_abstract, (_,_) = self.rnn(abstract)
        output_abstract, _ = pad_packed_sequence(output_abstract, batch_first=True)  # (bs, seq_len, emb_dim*2)

        alpha_abstract = torch.softmax(torch.matmul(output_abstract, atten_mask), dim=1)  # size: (bs, seq_len-ksz+1, 29368)
        abstract_features = torch.matmul(output_abstract.transpose(1, 2), alpha_abstract).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get document feature
        x_feature = title_features + abstract_features  # size: (bs, 29368, embed_dim*2)
        x_feature = torch.sum(x_feature * label_feature, dim=2)

        # add CorNet
        x_feature = self.cornet(x_feature)
        return x_feature


class multichannel_dilatedCNN_without_graph(nn.Module):
    def __init__(self, vocab_size, dropout, ksz, output_size, embedding_dim=200, rnn_num_layers=2, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(multichannel_dilatedCNN_without_graph, self).__init__()

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=rnn_num_layers,
                           dropout=self.dropout, bidirectional=True, batch_first=True)

        self.dconv = nn.Sequential(nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=self.ksz, padding=0, dilation=1),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=self.ksz, padding=0, dilation=2),
                                   nn.SELU(), nn.AlphaDropout(p=0.05),
                                   nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=self.ksz, padding=0, dilation=3),
                                   nn.SELU(), nn.AlphaDropout(p=0.05))

        self.fc1 = nn.Linear(self.embedding_dim, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # self.dropout = nn.Dropout(0.5)

        # corNet
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_abstract, input_title, mask, ab_length, title_length, g_node_feature):

        # get title content features
        atten_mask = g_node_feature.transpose(0, 1) * mask.unsqueeze(1)
        embedded_title = self.embedding_layer(input_title.long())
        packed_title = pack_padded_sequence(embedded_title, title_length, batch_first=True, enforce_sorted=False)

        output_title, (_,_) = self.rnn(packed_title)
        output_title, _ = pad_packed_sequence(output_title, batch_first=True)  # (bs, seq_len, emb_dim*2)
        output_title = output_title[:, :, :self.embedding_dim] + output_title[:, :, self.embedding_dim:]  # (bs, seq_len, emb_dim)

        title_atten = torch.softmax(torch.matmul(output_title, atten_mask), dim=1)
        title_feature = torch.matmul(output_title.transpose(1, 2), title_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim)

        # get abstract content features
        embedded_abstract = self.embedding_layer(input_abstract)  # size: (bs, seq_len, embed_dim)
        packed_abstract = pack_padded_sequence(embedded_abstract, ab_length, batch_first=True, enforce_sorted=False)
        output_abstract, (_,_) = self.rnn(packed_abstract)
        output_abstract, _ = pad_packed_sequence(output_abstract, batch_first=True)  # (bs, seq_len, emb_dim*2)
        output_abstract = output_abstract[:, :, :self.embedding_dim] + output_abstract[:, :, self.embedding_dim:]

        outputs_abstract = output_abstract.permute(0, 2, 1) # (bs, emb_dim, seq_length)
        abstract_conv = self.dconv(outputs_abstract)  # (bs, embed_dim, seq_len-ksz+1)

        abstract_atten = torch.softmax(torch.matmul(abstract_conv.transpose(1, 2), atten_mask), dim=1)  # size: (bs, seq_len-ksz+1, 29368)
        abstract_feature = torch.matmul(abstract_conv, abstract_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim)

        # get document feature
        x_feature = title_feature + abstract_feature  # size: (bs, 29368, embed_dim)

        x_feature = torch.tanh(self.fc1(x_feature))
        # x_feature = self.fc_drop(x_feature)
        # add CorNet
        cor_logit = self.cornet(x_feature.squeeze(2))
        return cor_logit


class HGCN4MeSH(nn.Module):

    def __init__(self, vocab_size, dropout, ksz, embedding_dim=200, rnn_num_layers=2):
        super(HGCN4MeSH, self).__init__()
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ksz = ksz
        self.embedding_dim = embedding_dim

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embedding_dim)
        self.emb_drop = nn.Dropout(0.2)

        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=rnn_num_layers,
                          dropout=self.dropout, bidirectional=True, batch_first=True)

        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)

        self.fc1 = nn.Linear(self.embedding_dim*2, embedding_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(embedding_dim, 1)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        #self.fc_drop = nn.Dropout(0.2)

    def forward(self, input_abstract, input_title, ab_length, title_length, g, g_node_feature):
        # get label features
        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1) # torch.Size([29368, 200*2])

        # get title content features
        embedded_title = self.embedding_layer(input_title.long())
        embedded_title = self.emb_drop(embedded_title)
        packed_title = pack_padded_sequence(embedded_title, title_length, batch_first=True, enforce_sorted=False)

        packed_output_title, _ = self.rnn(packed_title)
        output_unpacked_title, _ = pad_packed_sequence(packed_output_title, batch_first=True)  # (bs, seq_len, emb_dim*2)

        title_atten = torch.softmax(torch.matmul(output_unpacked_title, label_feature.transpose(0, 1)), dim=1)
        title_feature = torch.matmul(output_unpacked_title.transpose(1, 2), title_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get abstract content features
        embedded_abstract = self.embedding_layer(input_abstract)  # size: (bs, seq_len, embed_dim)
        embedded_abstract = self.emb_drop(embedded_abstract)
        packed_abstract = pack_padded_sequence(embedded_abstract, ab_length, batch_first=True, enforce_sorted=False)
        packed_output_abstract, _ = self.rnn(packed_abstract)
        output_unpacked_abstract, _ = pad_packed_sequence(packed_output_abstract, batch_first=True)  # (bs, seq_len, emb_dim*2)

        abstract_atten = torch.softmax(torch.matmul(output_unpacked_abstract, label_feature.transpose(0, 1)), dim=1)  # size: (bs, seq_len-ksz+1, 29368)
        abstract_feature = torch.matmul(output_unpacked_abstract.transpose(1, 2), abstract_atten).transpose(1, 2)  # size: (bs, 29368, embed_dim*2)

        # get document feature
        x_feature = title_feature + abstract_feature  # size: (bs, 29368, embed_dim*2)
        x_feature = nn.functional.leaky_relu(self.fc1(x_feature), negative_slope=0.2)
        x_feature = nn.functional.leaky_relu(self.fc2(x_feature), negative_slope=0.2)
        # x_feature = self.fc_drop(x_feature)

        return x_feature.squeeze(2)



class MLAttention(nn.Module):
    """
    """

    def __init__(self, labels_num, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, labels_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        masks = 1 - masks
        attention = self.attention(inputs).transpose(1, 2).masked_fill_(masks.bool(), -np.inf)  # [bz,num_label,seq_len]
        attention = F.softmax(attention, -1)
        x = torch.matmul(attention, inputs)  # [bz, num_label, hidden_sz]
        return x


class Bert_Baseline(BertPreTrainedModel):
    def __init__(self, config, num_label):
        super(Bert_Baseline, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.atten = MLAttention(num_label, config.hidden_size)

        self.fc1 = nn.Linear(config.hidden_size, 512)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.fc3 = nn.Linear(256, 1)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, src_input_ids, src_attention_mask):
        output, _ = self.bert(src_input_ids, src_attention_mask)
        output = self.dropout(output)

        atten_out = self.atten(output, src_attention_mask)  # [bz, num_label, hidden_sz]

        x_feature = nn.functional.tanh(self.fc1(atten_out))
        x_feature = nn.functional.tanh(self.fc2(x_feature))
        x_feature = self.fc3(x_feature).squeeze(2)

        x = torch.sigmoid(x_feature)
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


class MeSH_GCN(nn.Module):
    """
    attenCNN + GCN

    """
    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, embedding_dim=200):
        super(MeSH_GCN, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.hidden_gcn_size = hidden_gcn_size

        self.content_feature = attenCNN(self.vocab_size, self.nKernel, self.ksz)
        self.gcn = LabelNet(hidden_gcn_size, embedding_dim, embedding_dim)

    def forward(self, input_seq, g, g_node_feature):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])

        x = torch.sum(x_feature * label_feature, dim=2)
        x = torch.sigmoid(x)
        return x


class CorGCN(nn.Module):
    """
    attenCNN + GCN + CorNet
    """

    def __init__(self, vocab_size, nKernel, ksz, hidden_gcn_size, output_size, embedding_dim=200, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(CorGCN, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz
        self.hidden_gcn_size = hidden_gcn_size
        self.output_size = output_size

        self.content_feature = attenCNN(self.vocab_size, self.nKernel, self.ksz, atten_dropout=0.5, embedding_dim=200)
        self.gcn = LabelNet(hidden_gcn_size, embedding_dim, embedding_dim)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, g_node_feature, g):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.gcn(g, g_node_feature)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])

        x = torch.sum(x_feature * label_feature, dim=2)
        cor_logit = self.cornet(x)
        cor_logit = torch.sigmoid(cor_logit)
        return cor_logit


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
    def __init__(self, config, num_label):
        super(Bert_GCN, self).__init__()

        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self-attention
        self.atten = MLAttention(num_label, config.hidden_size)

        # label-wise attention
        self.linear = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.gcn = LabelNet(config.hidden_size, config.hidden_size, config.hidden_size)

        # weight adaptive layer
        self.linear_weight1 = torch.nn.Linear(config.hidden_size, 1)
        self.linear_weight2 = torch.nn.Linear(config.hidden_size, 1)

        # shared for all attention component
        self.linear_final = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ab, attention_ab, g, g_node_feature):
        # input_ab, input_title, attention_ab, attention_title, g, g_node_feature
        # self-attention output
        output, _ = self.bert(input_ab, attention_ab)
        output = self.dropout(output)  # [bz, seq_length, hidden_sz]

        # title_output, _ = self.bert(input_title, attention_title)
        # title_output = self.dropout(title_output)

        # output = torch.cat((title_output, ab_output), dim=1)
        # attention_mask = torch.cat((attention_title, attention_ab), dim=0)
        # self_atten_out = self.atten(output, attention_mask) # [bz, num_label, hidden_sz] [8, 29368, 768]
        self_atten_out = self.atten(output, attention_ab)

        # label-wise attention output mapping different parts of the document representation to different labels
        label_feature = self.gcn(g, g_node_feature)  # [num_labels, hidden_sz] (29468, 768)
        # label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # [29468, 768*2]
        # output_trans = self.linear(output)
        # label_atten = torch.softmax(torch.matmul(output_trans, label_feature.transpose(0, 1)),
        #                             dim=1)
        label_atten = torch.softmax(torch.matmul(output, label_feature.transpose(0, 1)), dim=1)
        label_atten_out = torch.matmul(output.transpose(1, 2), label_atten)  # [bz, hidden_sz, number_label]

        # attention fusion output
        factor1 = torch.sigmoid(self.linear_weight1(self_atten_out))
        factor2 = torch.sigmoid(self.linear_weight2(label_atten_out.transpose(1, 2)))
        factor1 = factor1 / (factor1 + factor2)
        factor2 = 1 - factor1

        out = factor1 * self_atten_out + factor2 * (label_atten_out.transpose(1, 2))
        out = F.relu(self.linear_final(out))
        out = torch.sigmoid(self.output_layer(out).squeeze(-1))

        return out


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
    def __init__(self, vocab_size, nKernel, ksz, hidden_rgcn_size, embedding_dim=200):
        super(MeSH_RGCN, self).__init__()
        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz = ksz

        self.content_feature = attenCNN(self.vocab_size, self.nKernel, self.ksz)

        self.rgcn = EntityClassify(embedding_dim, hidden_rgcn_size, embedding_dim, num_rels=2, num_bases=-1,
                                   dropout=0, use_self_loop=False, use_cuda=True, low_mem=True)

    def forward(self, input_seq, g, g_node_feature, edge_type, edge_norm):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.rgcn(g, g_node_feature, edge_type, edge_norm)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])

        x = torch.sum(x_feature * label_feature, dim=2)
        x = torch.sigmoid(x)
        return x


class CorRGCN(nn.Module):
    def __init__(self, vocab_size, nKernel, ksz, hidden_rgcn_size, output_size, embedding_dim=200, cornet_dim=1000,
                 n_cornet_blocks=2):
        super(CorRGCN, self).__init__()

        self.vocab_size = vocab_size
        self.nKernel = nKernel
        self.ksz =ksz

        self.content_feature = attenCNN(self.vocab_size, self.nKernel, self.ksz, atten_dropout=0.5, embedding_dim=200)

        self.rgcn = EntityClassify(embedding_dim, hidden_rgcn_size, embedding_dim, num_rels=2, num_bases=-1,
                                   dropout=0, use_self_loop=False, use_cuda=True, low_mem=True)
        self.cornet = CorNet(output_size, cornet_dim, n_cornet_blocks)

    def forward(self, input_seq, g, g_node_feature, edge_type, edge_norm):
        x_feature = self.content_feature(input_seq, g_node_feature)

        label_feature = self.rgcn(g, g_node_feature, edge_type, edge_norm)
        label_feature = torch.cat((label_feature, g_node_feature), dim=1)  # torch.Size([29368, 400])

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
