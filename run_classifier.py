import argparse
import os

import numpy as np
import ijson
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm

from build_graph import get_edge_and_node_fatures, build_MeSH_graph
from model import MeSH_GCN
from utils import tokenize, TextMultiLabelDataset


def prepare_dataset(train_data_path, test_data_path, mesh_id_list_path, word2vec_path, MeSH_id_pair_path,
                    parent_children_path, using_gpu=True, nKernel=128, ksz=[3, 4, 5], hidden_gcn_size=512,
                    embedding_dim=200):
    """ Load Dataset and Preprocessing """
    # load training data
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    all_text = []
    label = []
    label_id = []

    print("Loading training data")
    for i, obj in enumerate(tqdm(objects)):
        if i <= 500:
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
        else:
            break

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

    # # write training data to dataframe
    # print('write training data to dataframe')
    # df_train = pd.DataFrame(pmid, columns=['PMID'])
    # df_train['text'] = all_text
    # df_train = pd.concat([df_train, pd.DataFrame(label_vectors)], axis=1)

    # # write test data to dataframe
    # print('write test data to dataframe')
    # df_test = pd.DataFrame(test_pmid, columns=['PMID'])
    # df_test['text'] = test_text

    # text_col = 'text'
    # label_col = list(range(0, len(df_train.columns) - 2))

    print('use torchtext to prepare training and test data')
    TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, truncate_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.FloatTensor)

    train = TextMultiLabelDataset(all_text, text_field=TEXT, label_field=LABEL, lbls=label_vectors)
    test = TextMultiLabelDataset(test_text, text_field=TEXT, label_field=None, lbls=None)

    # build vocab
    print('Starting loading vocab')
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)
    vectors.unk_init = init.xavier_uniform
    TEXT.build_vocab(train, vectors=vectors)
    print('Finished loading vocab')

    # using the training corpus to create the vocabulary
    train_iter = data.Iterator(dataset=train, batch_size=128, train=True, repeat=False, device=0 if using_gpu else -1)
    test_iter = data.Iterator(dataset=test, batch_size=64, train=False, sort=False, device=0 if using_gpu else -1)

    vocab_size = len(TEXT.vocab.itos)
    num_classes = len(meshIDs)

    # Prepare label features
    edges, node_count, label_embedding = get_edge_and_node_fatures(MeSH_id_pair_path, parent_children_path, vocab_size,
                                                                   TEXT)
    G = build_MeSH_graph(edges, node_count, label_embedding)

    # Prepare model
    net = MeSH_GCN(vocab_size, nKernel, ksz, hidden_gcn_size, embedding_dim)
    net.cnn.embedding_layer.weight.data.copy_(TEXT.vocab.vectors)
    return mlb, train_iter, test_iter, G, net


# predicted binary labels
# find the top k labels in the predicted label set
def top_k_predicted(predictions, k):
    predicted_label = np.zeros(predictions.shape)
    for i in range(len(predictions)):
        top_k_index = (predictions[i].argsort()[-k:][::-1]).tolist()
        for j in top_k_index:
            predicted_label[i][j] = 1
    predicted_label = predicted_label.astype(np.int64)
    return predicted_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--test_path')
    parser.add_argument('--mesh_id_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--mesh_parent_children_path')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)

    mlb, train_iter, test_iter, G, model = prepare_dataset(args.train_path, args.test_path, args.mesh_id_path,
                                                           args.word2vec_path, args.meSH_pair_path,
                                                           args.mesh_parent_children_path)
    model.to(device)
    G.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=.99)
    criterion = nn.BCELoss()

    # start training:
    for epoch in range(3):
        for i, batch in enumerate(tqdm(train_iter)):
            model.train()
            # load data
            xs = batch.text.to(device)
            ys = batch.label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            lr_scheduler.step()

            # forward + backward + optimize
            outputs = model(xs, G, G.ndata['feat'])
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished training!')

    # start testing
    model.eval()
    pred = []
    for batch in tqdm(test_iter):
        xs = batch.text.to(device)
        logits = model(xs, xs, G, G.ndata['feat'])
        pred.append(logits)

    pred = pred.data.cpu().numpy()
    top_5_pred = top_k_predicted(pred, 5)
    # convert binary label back to orginal ones
    top_5_mesh = mlb.inverse_transform(top_5_pred)
    top_5_mesh = [list(item) for item in top_5_mesh]

    pred_label_5 = open('TextCNN_pred_label_5.txt', 'w')
    for meshs in top_5_mesh:
        mesh = ' '.join(meshs)
        pred_label_5.writelines(mesh.strip() + "\r")
    pred_label_5.close()

if __name__ == "__main__":
    main()
