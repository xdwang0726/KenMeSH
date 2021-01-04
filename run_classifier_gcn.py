import argparse
import logging
import os
import pickle
import sys

import ijson
import numpy as np
import torch
import torch.nn as nn
from dgl.data.utils import load_graphs
from sklearn.preprocessing import MultiLabelBinarizer
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors
from tqdm import tqdm

from model import MeSH_GCN
from utils import MeSH_indexing, pad_sequence
from eval_helper import precision_at_ks, example_based_evaluation, micro_macro_eval
from threshold_opt import eval
import torch.distributed as dist
import torch.utils.data.distributed


def prepare_dataset(train_data_path, test_data_path, MeSH_id_pair_file, word2vec_path, graph_file):
    """ Load Dataset and Preprocessing """
    # load training data
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    all_text = []
    label = []
    label_id = []

    print('Start loading training data')
    logging.info("Start loading training data")
    for i, obj in enumerate(tqdm(objects)):
        if i <= 1000:
            try:
                ids = obj["pmid"]
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
    logging.info("Finish loading training data")

    # load test data
    f_t = open(test_data_path, encoding="utf8")
    test_objects = ijson.items(f_t, 'documents.item')

    test_pmid = []
    test_text = []
    test_label = []

    print('Start loading test data')
    logging.info("Start loading test data")
    for obj in tqdm(test_objects):
        ids = obj["pmid"]
        text = obj["abstract"].strip()
        label = obj['meshId']
        test_pmid.append(ids)
        test_text.append(text)
        test_label.append(label)

    logging.info("Finish loading test data")

    print('load and prepare Mesh')
    # read full MeSH ID list
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    meshIDs = list(mapping_id.values())
    print('Total number of labels:', len(meshIDs))
    logging.info('Total number of labels:'.format(len(meshIDs)))
    mlb = MultiLabelBinarizer(classes=meshIDs)

    # Preparing training and test datasets
    print('prepare training and test sets')
    logging.info('Prepare training and test sets')
    train_dataset, test_dataset = MeSH_indexing(all_text, label_id, test_text, test_label)

    # build vocab
    print('building vocab')
    logging.info('Build vocab')
    vocab = train_dataset.get_vocab()

    # create Vector object map tokens to vectors
    print('load pre-trained BioWord2Vec')
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]

    print('graph', G.ndata['feat'].shape)

    # edges, node_count, label_embedding = get_edge_and_node_fatures(MeSH_id_pair_path, parent_children_path, vectors)
    # G = build_MeSH_graph(edges, node_count, label_embedding)

    print('prepare dataset and labels graph done!')
    return len(meshIDs), mlb, vocab, train_dataset, test_dataset, vectors, G


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
    # check if the dataset if train or test
    if len(batch[0]) == 2:
        label = [entry[0] for entry in batch]

        # padding according to the maximum sequence length in batch
        text = [entry[1] for entry in batch]
        text = pad_sequence(text, ksz=10, batch_first=True)
        return text, label
    else:
        text = [entry for entry in batch]
        text = pad_sequence(text, ksz=10, batch_first=True)
        return text


def train(train_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, num_workers, optimizer, lr_scheduler):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data = DataLoader(train_dataset, batch_size=batch_sz, collate_fn=generate_batch,
                            num_workers=num_workers, sampler=train_sampler)

    num_lines = num_epochs * len(train_data)

    print("Training....")
    for epoch in range(num_epochs):
        for i, (text, label) in enumerate(train_data):
            # ('batch:', i)
            # print('train_original', i, label, '\n')
            # test_label = mlb.fit_transform(label)

            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            text, label, G = text.to(device), label.to(device), G.to(device)
            output = model(text, G, G.ndata['feat'])
            # print('Allocated1:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')

            # print train output
            # pred = output.data.cpu().numpy()
            # print('pred_index', pred.argsort()[::-1][:, :10])
            # top_10_pred = top_k_predicted(test_label, pred, 10)
            # top_10_mesh = mlb.inverse_transform(top_10_pred)
            # print('predicted train', i, top_10_mesh, '\n')

            optimizer.zero_grad()
            loss = criterion(output, label)
            # print('loss', loss)
            loss.backward()
            optimizer.step()
            # print('Allocated2:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.8f} loss: {:3.8f}\n".format(
                        progress * 100, lr_scheduler.get_last_lr()[0], loss))
        # Adjust the learning rate
        lr_scheduler.step()
        # print('Allocated3:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')


def test(test_dataset, model, G, batch_sz, device):
    test_data = DataLoader(test_dataset, batch_size=batch_sz, collate_fn=generate_batch)
    pred = torch.zeros(0).to(device)
    ori_label = []
    print('Testing....')
    for text, label in test_data:
        text = text.to(device)
        print('test_orig', label, '\n')
        ori_label.append(label)
        flattened = [val for sublist in ori_label for val in sublist]
        with torch.no_grad():
            output = model(text, G, G.ndata['feat'])

            # results = output.data.cpu().numpy()
            # print(type(results), results.shape)
            # idx = results.argsort()[::-1][:, :10]
            # print(idx)
            # prob = [results[0][i] for i in idx]
            # print('probability:', prob)
            # top_10_pred = top_k_predicted(flattened, results, 10)
            # top_10_mesh = mlb.inverse_transform(top_10_pred)
            # print('predicted_test', top_10_mesh, '\n')

            pred = torch.cat((pred, output), dim=0)
    print('###################DONE#########################')
    return pred, flattened


# predicted binary labels
# find the top k labels in the predicted label set
# def top_k_predicted(predictions, k):
#     predicted_label = np.zeros(predictions.shape)
#     for i in range(len(predictions)):
#         top_k_index = (predictions[i].argsort()[-k:][::-1]).tolist()
#         for j in top_k_index:
#             predicted_label[i][j] = 1
#     predicted_label = predicted_label.astype(np.int64)
#     return predicted_label

def top_k_predicted(goldenTruth, predictions, k):
    predicted_label = np.zeros(predictions.shape)
    for i in range(len(predictions)):
        goldenK = len(goldenTruth[i])
        if goldenK <= k:
            top_k_index = (predictions[i].argsort()[-goldenK:][::-1]).tolist()
        else:
            top_k_index = (predictions[i].argsort()[-k:][::-1]).tolist()
        for j in top_k_index:
            predicted_label[i][j] = 1
    predicted_label = predicted_label.astype(np.int64)
    return predicted_label


def getLabelIndex(labels):
    label_index = np.zeros((len(labels), len(labels[1])))
    for i in range(0, len(labels)):
        index = np.where(labels[i] == 1)
        index = np.asarray(index)
        N = len(labels[1]) - index.size
        index = np.pad(index, [(0, 0), (0, N)], 'constant')
        label_index[i] = index

    label_index = np.array(label_index, dtype=int)
    label_index = label_index.astype(np.int32)
    return label_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path')
    parser.add_argument('--test_path')
    parser.add_argument('----meSH_pair_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--mesh_parent_children_path')
    parser.add_argument('--graph')
    parser.add_argument('--results')
    parser.add_argument('--save-model-path')

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=200)
    parser.add_argument('--ksz', default=10)
    parser.add_argument('--hidden_gcn_size', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--add_original_embedding', type=bool, default=True)
    parser.add_argument('--atten_dropout', type=float, default=0.5)

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.1)

    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://172.16.1.186:2222', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int, help='rank of distributed processes')

    # parser.add_argument('--fp16', default=True, type=bool)
    # parser.add_argument('--fp16_opt_level', type=str, default='O0')

    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()  # check if it is multiple gpu
    device = torch.device(args.device if torch.cuda.is_available() else "cpu", args.local_rank)
    # device = torch.device(args.device)
    logging.info('Device:'.format(device))

    # initialize the distributed training
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                            rank=args.dist_rank)

    # Get dataset and label graph & Load pre-trained embeddings
    num_nodes, mlb, vocab, train_dataset, test_dataset, vectors, G = prepare_dataset(args.train_path,
                                                                                     args.test_path,
                                                                                     args.meSH_pair_path,
                                                                                     args.word2vec_path, args.graph)

    vocab_size = len(vocab)
    model = MeSH_GCN(vocab_size, args.nKernel, args.ksz, args.hidden_gcn_size, args.add_original_embedding,
                     args.atten_dropout, embedding_dim=args.embedding_dim)

    # if torch.cuda.device_count() > 1:
    #     print("num of GPUs:", torch.cuda.device_count())
    #     model = nn.DataParallel(model)

    # model.cnn.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))
    model.content_feature.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    G.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCELoss()

    # training
    print("Start training!")
    train(train_dataset, model, mlb, G, args.batch_sz, args.num_epochs, criterion, device, args.num_workers, optimizer,
          lr_scheduler)
    print('Finish training!')
    # testing
    results, test_labels = test(test_dataset, model, G, args.batch_sz, device)
    # print('predicted:', results, '\n')

    test_label_transform = mlb.fit_transform(test_labels)
    # print('test_golden_truth', test_labels)

    pred = results.data.cpu().numpy()

    top_5_pred = top_k_predicted(test_labels, pred, 10)

    # convert binary label back to orginal ones
    top_5_mesh = mlb.inverse_transform(top_5_pred)
    # print('test_top_10:', top_5_mesh, '\n')
    top_5_mesh = [list(item) for item in top_5_mesh]

    pickle.dump(pred, open(args.results, "wb"))

    print("\rSaving model to {}".format(args.save_model_path))
    torch.save(model.to('cpu'), args.save_model_path)

    # precision @k
    test_labelsIndex = getLabelIndex(test_label_transform)
    precision = precision_at_ks(pred, test_labelsIndex, ks=[1, 3, 5])

    for k, p in zip([1, 3, 5], precision):
        print('p@{}: {:.5f}'.format(k, p))

    # example based evaluation
    example_based_measure_5 = example_based_evaluation(test_labels, top_5_mesh)
    print("EMP@5, EMR@5, EMF@5")
    for em in example_based_measure_5:
        print(em, ",")

    # label based evaluation
    label_measure_5 = micro_macro_eval(test_label_transform, top_5_pred)
    print("MaP@5, MiP@5, MaF@5, MiF@5: ")
    for measure in label_measure_5:
        print(measure, ",")

    micro_precision, micro_recall, micro_f_score = eval(test_label_transform, pred, num_nodes, len(pred))
    print(micro_precision, micro_recall, micro_f_score)
if __name__ == "__main__":
    main()
