import argparse
import logging
import os
import sys

import ijson
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors
from tqdm import tqdm

from build_graph import get_edge_and_node_fatures, build_MeSH_graph
from model import MeSH_GCN
from utils import MeSH_indexing


def prepare_dataset(train_data_path, test_data_path, mesh_id_list_path, word2vec_path, MeSH_id_pair_path,
                    parent_children_path):
    """ Load Dataset and Preprocessing """
    # load training data
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    all_text = []
    label = []
    label_id = []

    logging.info("Start loading training data")
    for i, obj in enumerate(tqdm(objects)):
        if i <= 1000:
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

    logging.info("Finish loading training data")

    # load test data
    f_t = open(test_data_path, encoding="utf8")
    test_objects = ijson.items(f_t, 'documents.item')

    test_pmid = []
    test_text = []

    logging.info("Start loading test data")
    for obj in tqdm(test_objects):
        try:
            ids = obj["pmid"]
            text = obj["abstract"].strip()
            test_pmid.append(ids)
            test_text.append(text)
        except AttributeError:
            print(obj["pmid"].strip())
    logging.info("Finish loading test data")

    # read full MeSH ID list
    with open(mesh_id_list_path, "r") as ml:
        meshIDs = ml.readlines()

    meshIDs = [ids.strip() for ids in meshIDs]
    logging.info('Total number of labels:'.format(len(meshIDs)))
    mlb = MultiLabelBinarizer(classes=meshIDs)

    # Preparing training and test datasets
    logging.info('Prepare training and test sets')
    train_dataset, test_dataset = MeSH_indexing(all_text, label_id, test_text)

    # build vocab
    logging.info('Build vocab')
    vocab = train_dataset.get_vocab()

    # create Vector object map tokens to vectors
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    # Prepare label features
    edges, node_count, label_embedding = get_edge_and_node_fatures(MeSH_id_pair_path, parent_children_path, vectors)
    G = build_MeSH_graph(edges, node_count, label_embedding)

    return mlb, vocab, train_dataset, test_dataset, vectors, G


def weight_matrix(vocab, vectors, dim=200):
    weight_matrix = np.zeros([len(vocab.itos), dim])
    for i, token in enumerate(vocab.stoi):
        try:
            weight_matrix[i] = vectors.__getitem__(token)
        except KeyError:
            weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
    return torch.from_numpy(weight_matrix)


def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """

    label = [entry[0] for entry in batch]

    # padding according to the maximum sequence length in batch
    text = [entry[1] for entry in batch]
    text = pad_sequence(text, batch_first=True)
    return text, label


def train(train_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, num_workers, optimizer, lr_scheduler):
    train_data = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch,
                            num_workers=num_workers)

    num_lines = num_epochs * len(train_data)

    for epoch in range(num_epochs):
        for i, (text, label) in enumerate(train_data):
            optimizer.zero_grad()
            label = mlb.fit_transform(label)
            text, label = text.to(device), label.to(device)
            output = model(text, G, G.ndata['feat'])
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
                        progress * 100, lr_scheduler.get_lr()[0], loss))
        # Adjust the learning rate
        lr_scheduler.step()


def test(test, model, batch_sz, device):
    data = DataLoader(test, batch_size=batch_sz, collate_fn=generate_batch)

    total_accuracy = []
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            accuracy = (output.argmax(1) == cls).float().mean().item()
            total_accuracy.append(accuracy)

    if not total_accuracy:
        return 0.0

    return sum(total_accuracy) / len(total_accuracy)


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

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=128)
    parser.add_argument('--ksz', type=list, default=[3, 4, 5])
    parser.add_argument('--hidden_gcn_size', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=200)

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_step_sz', type=int, default=1)
    parser.add_argument('--lr_gamma', type=float, default=0.8)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info('Device:'.format(device))

    # Get dataset and label graph & Load pre-trained embeddings
    mlb, vocab, train_dataset, test_dataset, vectors, G = prepare_dataset(args.train_path,
                                                                          args.test_path, args.mesh_id_path,
                                                                          args.word2vec_path, args.meSH_pair_path,
                                                                          args.mesh_parent_children_path)

    vocab_size = len(vocab)
    model = MeSH_GCN(vocab_size, args.nKernel, args.ksz, args.hidden_gcn_size, args.embedding_dim)
    model.cnn.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))

    model.to(device)
    G.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCELoss()

    # training
    train(train, model, mlb, G, args.batch_sz, args.num_epochs, criterion, device, args.num_workers, optimizer,
          lr_scheduler)


    # start testing
    # model.eval()
    # pred = []
    # for batch in tqdm(test_iter):
    #     xs = batch.text.to(device)
    #     logits = model(xs, xs, G, G.ndata['feat'])
    #     pred.append(logits)
    #
    # pred = pred.data.cpu().numpy()
    # top_5_pred = top_k_predicted(pred, 5)
    # # convert binary label back to orginal ones
    # top_5_mesh = mlb.inverse_transform(top_5_pred)
    # top_5_mesh = [list(item) for item in top_5_mesh]
    #
    # pred_label_5 = open('TextCNN_pred_label_5.txt', 'w')
    # for meshs in top_5_mesh:
    #     mesh = ' '.join(meshs)
    #     pred_label_5.writelines(mesh.strip() + "\r")
    # pred_label_5.close()

if __name__ == "__main__":
    main()
