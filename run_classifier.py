import os

import ijson
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm

from utils import tokenize, TextMultiLabelDataset
from build_graph import get_edge_and_node_fatures, build_MeSH_graph
from model import MeSH_GCN


def prepare_dataset(train_data_path, test_data_path, mesh_id_list_path, word2vec_path, MeSH_id_pair_path,
                    parent_children_path, using_gpu):
    """ Load Dataset and Preprocessing """
    # load training data
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    all_text = []
    label = []
    label_id = []

    print("Loading training data")
    for obj in tqdm(objects):
        try:
            ids = obj["pmid"].strip()
            text = obj["abstractText"].strip()
            original_label = obj["meshMajor"]
            mesh_id = obj['meshId']
            pmid.append(ids)
            all_text.append(text)
            label.append(original_label)
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())
    print("Finish loading training data")

    # load test data
    f_t = open(test_data_path, encoding="utf8")
    test_objects = ijson.items(f_t, 'articles.item')

    test_pmid = []
    test_text = []

    print("Loading test data")
    for obj in tqdm(test_objects):
        try:
            ids = obj["pmid"].strip()
            text = obj["abstract"].strip()
            test_pmid.append(ids)
            test_text.append(text)
        except AttributeError:
            print(obj["pmid"].strip())
    print("Finish loading test data")

    # read full MeSH ID list
    with open(mesh_id_list_path, "r") as ml:
        meshIDs = ml.readlines()

    meshIDs = [ids.strip() for ids in meshIDs]
    print('Number of labels: ', len(meshIDs))
    mlb = MultiLabelBinarizer(classes=meshIDs)

    label_vectors = mlb.fit_transform(label_id)

    # write training data to dataframe
    df_train = pd.DataFrame(pmid, columns=['PMID'])
    df_train['text'] = all_text
    df_train = pd.concat([df_train, pd.DataFrame(label_vectors, columns=meshIDs)], axis=1)

    # write test data to dataframe
    df_test = pd.DataFrame(test_pmid, columns=['PMID'])
    df_test['text'] = test_text

    text_col = 'text'
    label_col = meshIDs

    TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, truncate_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.FloatTensor)

    train = TextMultiLabelDataset(df_train, text_field=TEXT, label_field=LABEL, txt_col=text_col, lbl_cols=label_col)
    test = TextMultiLabelDataset(df_test, test_text=TEXT, text_col=text_col, label_field=None, test=True)

    # build vocab
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)
    vectors.unk_init = init.xavier_uniform
    TEXT.build_vocab(train, vectors=vectors)

    # using the training corpus to create the vocabulary
    train_iter = data.Iterator(dataset=train, batch_size=32, train=True, repeat=False, device=0 if using_gpu else -1)
    test_iter = data.Iterator(dataset=test, batch_size=64, train=False, sort=False, device=0 if using_gpu else -1)

    vocab_size = len(TEXT.vocab.itos)
    num_classes = len(meshIDs)

    # Prepare label features
    edges, node_count, label_embedding = get_edge_and_node_fatures(MeSH_id_pair_path, parent_children_path, vocab_size,
                                                                   TEXT)
    G = build_MeSH_graph(edges, node_count, label_embedding)

    # Prepare model
    net = MeSH_GCN(vocab_size, nKernel, ksz, num_classes, hidden_gcn_size, dropout_rate, embedding_dim)
    net.embedding_layer.weight.data.copy_(TEXT.vocab.vectors)
    return train_iter, test_iter, G
