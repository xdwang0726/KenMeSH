import argparse
import logging
import os
import pickle
import sys

# import EarlyStopping
# from pytorchtools import EarlyStopping
import dgl
import ijson
import numpy as np
from dgl.data.utils import load_graphs
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors, Vocab
from tqdm import tqdm

from eval_helper import precision_at_ks, example_based_evaluation, micro_macro_eval
from losses import *
from model import multichannel_dilatedCNN
from utils_multi import MeSH_indexing, pad_sequence


def prepare_dataset(train_data_path, test_data_path, MeSH_id_pair_file, word2vec_path, graph_file, num_example): #graph_cooccurence_file
    """ Load Dataset and Preprocessing """
    # load training data
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    train_title = []
    all_text = []
    label = []
    label_id = []

    print('Start loading training data')
    logging.info("Start loading training data")
    for i, obj in enumerate(tqdm(objects)):
        if i <= num_example:
            try:
                ids = obj["pmid"]
                heading = obj['title'].strip()
                heading = heading.translate(str.maketrans('', '', '[]'))
                # print('heading', type(heading), heading)
                if len(heading) == 0:
                    print('paper ', ids, ' does not have title!')
                else:
                    if heading == 'In process':
                        continue
                    else:
                        text = obj["abstractText"].strip()
                        text = text.translate(str.maketrans('', '', '[]'))
                        original_label = obj["meshMajor"]
                        mesh_id = obj['meshId']
                        pmid.append(ids)
                        train_title.append(heading)
                        all_text.append(text)
                        label.append(original_label)
                        label_id.append(mesh_id)
            except AttributeError:
                print(obj["pmid"].strip())
        else:
            break


    # for i, obj in enumerate(tqdm(objects)):
    #     try:
    #         ids = obj["pmid"]
    #         heading = obj['title'].strip()
    #         heading = heading.translate(str.maketrans('', '', '[]'))
    #         abstract = obj["abstractText"].strip()
    #         clean_abstract = abstract.translate(str.maketrans('', '', '[]'))
    #         if len(heading) == 0 or heading == 'In process':
    #             print('paper ', ids, ' does not have title!')
    #             continue
    #         elif len(clean_abstract) == 0:
    #             print('paper ', ids, ' does not have abstract!')
    #             continue
    #         else:
    #             try:
    #                 original_label = obj["meshMajor"]
    #                 mesh_id = obj['meshId']
    #                 journal = obj['journal']
    #                 pmid.append(ids)
    #                 title.append(heading)
    #                 all_text.append(abstract)
    #                 label.append(original_label)
    #                 label_id.appen(mesh_id)
    #                 journals.append(journal)
    #             except KeyError:
    #                 print('tfidf error', ids)
    #     except AttributeError:
    #         print(obj["pmid"].strip())

    print('check if title and abstract are coresponded')
    if len(all_text) == len(train_title):
        print('True')
    else:
        print(len(all_text), len(train_title))
    print("Finish loading training data")
    logging.info("Finish loading training data")
    print("number of training data", len(pmid))

    # load test data
    f_t = open(test_data_path, encoding="utf8")
    test_objects = ijson.items(f_t, 'documents.item')

    # test_pmid = []
    test_title = []
    test_text = []
    # test_label = []

    print('Start loading test data')
    logging.info("Start loading test data")
    for obj in tqdm(test_objects):
        # ids = obj["pmid"]
        heading = obj["title"].strip()
        text = obj["abstract"].strip()
        # label = obj['meshId']
        # test_pmid.append(ids)
        test_title.append(heading)
        test_text.append(text)
        # test_label.append(label)

    # for i, obj in enumerate(tqdm(objects)):
    #     if 100000 < i <= 120000:
    #         try:
    #             ids = obj["pmid"]
    #             heading = obj['title'].strip()
    #             heading = heading.translate(str.maketrans('', '', '[]'))
    #             # print('heading', type(heading), heading)
    #             if len(heading) == 0:
    #                 print('paper ', ids, ' does not have title!')
    #             else:
    #                 if heading == 'In process':
    #                     continue
    #                 else:
    #                     text = obj["abstractText"].strip()
    #                     text = text.translate(str.maketrans('', '', '[]'))
    #                     mesh_id = obj['meshId']
    #                     test_pmid.append(ids)
    #                     test_title.append(heading)
    #                     test_text.append(text)
    #                     test_label.append(mesh_id)
    #         except AttributeError:
    #             print(obj["pmid"].strip())
    #     else:
    #         break
    logging.info("Finish loading test data")
    print("number of test data", len(test_title))

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

    # create Vector object map tokens to vectors
    print('load pre-trained BioWord2Vec')
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    # build vocab
    # print('building vocab')
    # vocab = Vocab(vectors.stoi, specials=[])
    # print('vocab', len(vocab.itos))

    # Preparing training and test datasets
    print('prepare training and test sets')
    logging.info('Prepare training and test sets')
    train_dataset, test_dataset = MeSH_indexing(all_text, label_id, test_text, None, train_title, test_title, ngrams=1, vocab=None,
                  include_unk=False, is_test=True, is_multichannel=True)

    # build vocab
    print('building vocab')
    logging.info('Build vocab')
    vocab = train_dataset.get_vocab()



    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]
    # G_c = load_graphs(graph_cooccurence_file)[0][0]

    print('graph', G.ndata['feat'].shape)

    # edges, node_count, label_embedding = get_edge_and_node_fatures(MeSH_id_pair_path, parent_children_path, vectors)
    # G = build_MeSH_graph(edges, node_count, label_embedding)

    print('prepare dataset and labels graph done!')
    return len(meshIDs), mlb, vocab, train_dataset, test_dataset, vectors, G # G_c


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
    if len(batch[0]) == 3:
        label = [entry[0] for entry in batch]

        # padding according to the maximum sequence length in batch
        abstract = [entry[1] for entry in batch]
        abstract_length = [len(seq) for seq in abstract]
        abstract = pad_sequence(abstract, ksz=10, batch_first=True)

        title = [entry[2] for entry in batch]
        title_length = []
        for i, seq in enumerate(title):
            if len(seq) == 0:
                length = len(seq) + 1
            else:
                length = len(seq)
            title_length.append(length)
        title = pad_sequence(title, ksz=10, batch_first=True)
        return label, abstract, title, abstract_length, title_length

    else:
        abstract = [entry[0] for entry in batch]
        abstract_length = [len(seq) for seq in abstract]
        abstract = pad_sequence(abstract, ksz=10, batch_first=True)

        title = [entry[1] for entry in batch]
        title_length = []
        for i, seq in enumerate(title):
            if len(seq) == 0:
                length = len(seq) + 1
            else:
                length = len(seq)
            title_length.append(length)
        title = pad_sequence(title, ksz=10, batch_first=True)
        return abstract, title, abstract_length, title_length


def train(train_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, num_workers, optimizer, lr_scheduler):
    train_data = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch,
                            num_workers=num_workers)

    num_lines = num_epochs * len(train_data)

    #    early_stopping = EarlyStopping(patience=patience, verbose=True)
    print("Training....")
    for epoch in range(num_epochs):
        for i, (label, abstract, title, abstract_length, title_length) in enumerate(train_data):
            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            abstract_length = torch.Tensor(abstract_length)
            title_length = torch.Tensor(title_length)
            abstract, title, label, abstract_length, title_length = abstract.to(device), title.to(device), label.to(device), abstract_length.to(device), title_length.to(device)
            G = G.to(device)
            G.ndata['feat'] = G.ndata['feat'].to(device)
            # G_c = G_c.to(device)
            output = model(abstract, title, abstract_length, title_length, G, G.ndata['feat']) #, G_c, G_c.ndata['feat'])
            # output = model(abstract, title, G.ndata['feat'])

            optimizer.zero_grad()
            loss = criterion(output, label)
            # print('loss', loss)
            loss.backward()
            optimizer.step()
            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.8f} loss: {:3.8f}\n".format(
                        progress * 100, lr_scheduler.get_last_lr()[0], loss))
        # Adjust the learning rate
        lr_scheduler.step()


def test(test_dataset, model, G, batch_sz, device):
    test_data = DataLoader(test_dataset, batch_size=batch_sz, collate_fn=generate_batch)
    pred = torch.zeros(0).to(device)
    ori_label = []
    print('Testing....')
    #for label, abstract, title, abstract_length, title_length in test_data:
    for abstract, title, abstract_length, title_length in test_data:
        abstract_length = torch.Tensor(abstract_length)
        title_length = torch.Tensor(title_length)
        abstract, title, abstract_length, title_length = abstract.to(device), title.to(device), abstract_length.to(device), title_length.to(device)
        G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)
        # G_c, G_c.ndata['feat'] = G_c.to(device), G_c.ndata['feat'].to(device)
        # ori_label.append(label)
        # flattened = [val for sublist in ori_label for val in sublist]
        with torch.no_grad():
            output = model(abstract, title, abstract_length, title_length, G, G.ndata['feat']) #, G_c, G_c.ndata['feat'])
            # output = model(abstract, title, G.ndata['feat'])
            pred = torch.cat((pred, output), dim=0)
    print('###################DONE#########################')
    return pred #, flattened


# def top_k_predicted(goldenTruth, predictions, k):
#     predicted_label = np.zeros(predictions.shape)
#     for i in range(len(predictions)):
#         goldenK = len(goldenTruth[i])
#         if goldenK <= k:
#             top_k_index = (predictions[i].argsort()[-goldenK:][::-1]).tolist()
#         else:
#             top_k_index = (predictions[i].argsort()[-k:][::-1]).tolist()
#         for j in top_k_index:
#             predicted_label[i][j] = 1
#     predicted_label = predicted_label.astype(np.int64)
#     return predicted_label


def top_k_predicted(predictions, k):
    predicted_label = np.zeros(predictions.shape)
    for i in range(len(predictions)):
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
    parser.add_argument('--graph_cooccurence')
    parser.add_argument('--results')
    parser.add_argument('--save-model-path')
    parser.add_argument('--model-path')

    parser.add_argument('--num_example', type=int, default=10000)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=200)
    parser.add_argument('--ksz', default=5)
    parser.add_argument('--hidden_gcn_size', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--add_original_embedding', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--atten_dropout', type=float, default=0.5)

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.98)

    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()  # check if it is multiple gpu
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device(args.device)
    logging.info('Device:'.format(device))

    # Get dataset and label graph & Load pre-trained embeddings
    num_nodes, mlb, vocab, train_dataset, test_dataset, vectors, G = prepare_dataset(args.train_path,
                                                                                     args.test_path,
                                                                                     args.meSH_pair_path,
                                                                                     args.word2vec_path,
                                                                                     args.graph,
                                                                                     args.num_example) # args. graph_cooccurence,

    vocab_size = len(vocab)

    # model = multichannel_dilatedCNN(vocab_size, args.dropout, args.ksz, num_nodes, G, device,
    #                                 embedding_dim=200, rnn_num_layers=2, cornet_dim=1000, n_cornet_blocks=2,
    #                                 gat_num_heads=8, gat_num_layers=2, gat_num_out_heads=1)
    #
    # model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)
    # model.embedding_layer.weight.data.copy_(vectors.vectors).to(device)
    # model = multichannle_attenCNN(vocab_size, args.nKernel, args.ksz, args.add_original_embedding,
    #                        args.atten_dropout, embedding_dim=args.embedding_dim)
    #
    # model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))

    # model.to(device)
    # G = G.to(device)
    # G = dgl.add_self_loop(G)
    # # G_c.to(device)
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    # criterion = nn.BCELoss()
    # criterion = FocalLoss()
    # criterion = AsymmetricLossOptimized()

    # training
    # print("Start training!")
    # train(train_dataset, model, mlb, G, args.batch_sz, args.num_epochs, criterion, device, args.num_workers, optimizer,
    #       lr_scheduler)
    # print('Finish training!')
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, args.save_parameter_path)
    # print('save model')
    # torch.save(model, args.save_model_path)

    # load model
    model = torch.load(args.model_path)

    # testing
    # results, test_labels = test(test_dataset, model, G, args.batch_sz, device)
    results = test(test_dataset, model, G, args.batch_sz, device)

    # test_label_transform = mlb.fit_transform(test_labels)
    # print('test_golden_truth', test_labels)

    pred = results.data.cpu().numpy()

    # top_5_pred = top_k_predicted(test_labels, pred, 10)
    top_10_pred = top_k_predicted(pred, 10)

    # convert binary label back to orginal ones
    top_10_mesh = mlb.inverse_transform(top_10_pred)
    # print('test_top_10:', top_5_mesh, '\n')
    top_10_mesh = [list(item) for item in top_10_mesh]

    pickle.dump(top_10_mesh, open(args.results, "wb"))

    # print("\rSaving model to {}".format(args.save_model_path))
    # torch.save(model.to('cpu'), args.save_model_path)

    # # precision @k
    # test_labelsIndex = getLabelIndex(test_label_transform)
    # precision = precision_at_ks(pred, test_labelsIndex, ks=[1, 3, 5])
    #
    # for k, p in zip([1, 3, 5], precision):
    #     print('p@{}: {:.5f}'.format(k, p))
    #
    # # example based evaluation
    # example_based_measure_5 = example_based_evaluation(test_labels, top_5_mesh)
    # print("EMP@5, EMR@5, EMF@5")
    # for em in example_based_measure_5:
    #     print(em, ",")
    #
    # # label based evaluation
    # label_measure_5 = micro_macro_eval(test_label_transform, top_5_pred)
    # print("MaP@5, MiP@5, MaF@5, MiF@5: ")
    # for measure in label_measure_5:
    #     print(measure, ",")


if __name__ == "__main__":
    main()
