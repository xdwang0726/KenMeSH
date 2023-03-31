import argparse
from ctypes import sizeof
import json
import os
import sys
import logging 
import tracemalloc

import dgl
import ijson

import pickle
import psutil
import random
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, random_split
from torchtext.vocab import Vectors
from tqdm import tqdm

from eval_helper import precision_at_ks, example_based_evaluation, micro_macro_eval, zero_division
from losses import *
from model import *
from pytorchtools import EarlyStopping
from utils import MeSH_indexing, pad_sequence


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn


def flatten(l):
    flat = [i for item in l for i in item]
    return flat


# def prepare_dataset(title_path, abstract_path, label_path, mask_path, MeSH_id_pair_file, word2vec_path, graph_file, is_multichannel):
def prepare_dataset(dataset_path, MeSH_id_pair_file, word2vec_path, graph_file, is_multichannel):
    """ Load Dataset and Preprocessing """
    # load training data
    print('Start loading training data')
    # mesh_mask = pickle.load(open(mask_path, 'rb'))

    # all_title = pickle.load(open(title_path, 'rb'))
    # all_text = pickle.load(open(abstract_path, 'rb'))
    # label_id = pickle.load(open(label_path, 'rb'))
    tracemalloc.start()

    mesh_mask = []
    all_title = []
    all_text = []
    label_id = []
    
    f = open(dataset_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item') 

    for i, obj in enumerate(tqdm(objects)):
        title = obj["title"]
        abstractText = obj["abstractText"]

        if len(title) < 1 or len(abstractText) < 1:
            continue      

        titles = title.split(" ")
        if len(titles) < 2:
            continue

        abstractTexts = abstractText.split(" ")
        if len(abstractTexts) < 2:
            continue
        
        mesh_mask.append(obj["meshMask"])
        all_title.append(obj["title"])
        all_text.append(obj["abstractText"])
        label_id.append(list(obj["meshID"].keys()))

        # print(all_text, all_title, label_id, mesh_mask)
        # break
    

    assert len(all_text) == len(all_title), 'title and abstract in the training set are not matching'
    print('Finish loading training data')
    f.close()

    print("Load Dataset Memory: ", tracemalloc.get_traced_memory())
    tracemalloc.stop()

    print('number of training data %d' % len(all_title))

    tracemalloc.start()
    print('load and prepare Mesh')
    # read full MeSH ID list
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()
    meshIDs = list(mapping_id.values())
    print('Total number of labels %d' % len(meshIDs))
    # mesh_ids_str = [str(x) for x in meshIDs]
    # index_dic = {k: v for k, v in enumerate(meshIDs)}
    # mesh_index = list(index_dic.values())
    
    mlb = MultiLabelBinarizer(classes=meshIDs)
    mlb.fit(meshIDs)

    print("Mesh ID, MLB: ", tracemalloc.get_traced_memory())
    tracemalloc.stop()

    tracemalloc.start()

    # create Vector object map tokens to vectors
    print('load pre-trained BioWord2Vec')
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    print("Loading Vectors: ", tracemalloc.get_traced_memory())
    tracemalloc.stop()

    tracemalloc.start()
    # Preparing training and test datasets
    print('prepare training and test sets')
    # Converting texts in Tokens in a Tensor List
    dataset = MeSH_indexing(all_text, all_title, all_text[20:], all_title[20:], label_id[20:], mesh_mask[20:], all_text[:20],
                            all_title[:20], label_id[:20], mesh_mask[20:], is_test=False, is_multichannel=is_multichannel)

    # build vocab
    print('building vocab')
    vocab = dataset.get_vocab()
    valid_size = 0.02
    split = int(np.floor(valid_size * len(all_title)))
    train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[len(all_title[:-20]) - split, split])
    print("Prepare Dataset Mesh Indexing: ", tracemalloc.get_traced_memory())
    tracemalloc.stop()

    tracemalloc.start()
    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]
    print('graph', G.ndata['feat'].shape)
    
    print("Graph Load: ", tracemalloc.get_traced_memory())
    tracemalloc.stop()

    print('prepare dataset and labels graph done!')
    return len(meshIDs), mlb, vocab, train_dataset, valid_dataset, vectors, G


def weight_matrix(vocab, vectors, dim=200):
    weight_matrix = np.zeros([len(vocab.get_itos()), dim])
    for i, token in enumerate(vocab.get_stoi()):
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
    # check if the dataset is multi-channel or not
    if len(batch[0]) == 4:
        label = [entry[0] for entry in batch]
        mesh_mask = [entry[1] for entry in batch]

        # padding according to the maximum sequence length in batch
        abstract = [entry[2] for entry in batch]
        abstract_length = [len(seq) for seq in abstract]
        abstract = pad_sequence(abstract, ksz=3, batch_first=True)

        title = [entry[3] for entry in batch]
        title_length = []
        for i, seq in enumerate(title):
            if len(seq) == 0:
                length = len(seq) + 1
            else:
                length = len(seq)
            title_length.append(length)
        title = pad_sequence(title, ksz=3, batch_first=True)
        return label, mesh_mask, abstract, title, abstract_length, title_length

    else:
        label = [entry[0] for entry in batch]
        mesh_mask = [entry[1] for entry in batch]

        text = [entry[2] for entry in batch]
        text_length = [len(seq) for seq in text]
        text = pad_sequence(text, ksz=3, batch_first=True)

        return label, mesh_mask, text, text_length


def train(train_dataset, valid_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, num_workers, optimizer,
          lr_scheduler, model_name):

    train_data = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch, num_workers=num_workers, pin_memory=True)

    valid_data = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=True, collate_fn=generate_batch, num_workers=num_workers, pin_memory=True)

    print('train', len(train_data.dataset))

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=3, verbose=True)

    print("Training....")
    for epoch in range(num_epochs):
        model.train()  # prep model for training
        if model_name == 'ablation1':
            for i, (label, mesh_mask, text, text_length) in enumerate(train_data):
                label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)



                # mesh_mask = torch.from_numpy(mlb.fit_transform(mesh_mask)).type(torch.float)
                text_length = torch.Tensor(text_length)
                text, label, mesh_mask, text_length = text.to(device), label.to(device), mesh_mask.to(
                    device), text_length.to(device)
                G = G.to(device)
                G.ndata['feat'] = G.ndata['feat'].to(device)
                output = model(text, text_length, mesh_mask, G, G.ndata['feat'])
                loss = criterion(output, label)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                for param in model.parameters():
                    param.grad = None
                train_losses.append(loss.item())  # record training loss

            # Adjust the learning rate
            lr_scheduler.step()

            with torch.no_grad():
                model.eval()
                for i, (label, mesh_mask, text, text_length) in enumerate(valid_data):
                    label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
                    # mesh_mask = torch.from_numpy(mlb.fit_transform(mesh_mask)).type(torch.float)
                    text_length = torch.Tensor(text_length)
                    text, label, mesh_mask, text_length = text.to(device), label.to(device), mesh_mask.to(
                        device), text_length.to(device)
                    G = G.to(device)
                    G.ndata['feat'] = G.ndata['feat'].to(device)

                    output = model(text, text_length, mesh_mask, G, G.ndata['feat'])

                    loss = criterion(output, label)
                    valid_losses.append(loss.item())
        else:
            for i, (label, mask, abstract, title, abstract_length, title_length) in enumerate(train_data):
                # print("----------------------------------------------------")
                # print(f'label: ${label}, \n mask: ${mask}, \n abstract: ${abstract}, \n title: ${title}, \n abstract_length: ${abstract_length}, \n title_length: ${title_length}', "\n-----------Train Data-----------")
                
                try:
                    label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
                    # print("Label: \n", label)
                    # print(print("Mask: ", np.bincount(label[0])))

                    # mask = torch.from_numpy(mlb.fit_transform(mask)).type(torch.float)
                    abstract_length = torch.Tensor(abstract_length)
                    title_length = torch.Tensor(title_length)

                    abstract = torch.from_numpy(np.asarray(abstract))
                    title = torch.from_numpy(np.asarray(title))
                    label = torch.from_numpy(np.asarray(label))
                    mask = torch.from_numpy(np.asarray(mask))
                    abstract_length = torch.Tensor(np.asarray(abstract_length))
                    title_length = torch.Tensor(np.asarray(title_length))

                    abstract, title, label, mask, abstract_length, title_length = abstract.to(device), title.to(device), label.to(device), mask.to(device), abstract_length, title_length
                    G = G.to(device)
                    G.ndata['feat'] = G.ndata['feat'].to(device)
                    if model_name == "Full":
                        output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat'])
                    elif model_name == "ablation2":
                        output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat'])
                    elif model_name == "ablation3":
                        output = model(abstract, title, mask, abstract_length, title_length, G.ndata['feat'])
                    elif model_name == "HGCN4MeSH":
                        output = model(abstract, title, abstract_length, title_length, G, G.ndata['feat'])

                    loss = criterion(output, label)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    for param in model.parameters():
                        param.grad = None
                    train_losses.append(loss.item())  # record training loss
                except BaseException as exception:
                    logging.warning(f"Exception Name: {type(exception).__name__}")
                    logging.warning(f"Exception Desc: {exception}")
                    # print("Training Data: ", label, mask, abstract, title, abstract_length, title_length)
            # Adjust the learning rate
            lr_scheduler.step()

            with torch.no_grad():
                model.eval()
                for i, (label, mask, abstract, title, abstract_length, title_length) in enumerate(valid_data):
                    label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
                    # mask = torch.from_numpy(mlb.fit_transform(mask)).type(torch.float)
                    abstract_length = torch.Tensor(abstract_length)
                    title_length = torch.Tensor(title_length)

                    abstract = torch.from_numpy(np.asarray(abstract))
                    title = torch.from_numpy(np.asarray(title))
                    label = torch.from_numpy(np.asarray(label))
                    mask = torch.from_numpy(np.asarray(mask))
                    abstract_length = torch.from_numpy(np.asarray(abstract_length))
                    title_length = torch.from_numpy(np.asarray(title_length))

                    abstract, title, label, mask, abstract_length, title_length = abstract.to(device), title.to(device), label.to(device), mask.to(device), abstract_length, title_length
                    G = G.to(device)
                    G.ndata['feat'] = G.ndata['feat'].to(device)

                    if model_name == "Full":
                        output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat'])
                    elif model_name == "ablation2":
                        output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat'])
                    elif model_name == "ablation3":
                        output = model(abstract, title, mask, abstract_length, title_length, G.ndata['feat'])
                    elif model_name == "HGCN4MeSH":
                        output = model(abstract, title, abstract_length, title_length, G, G.ndata['feat'])

                    loss = criterion(output, label)
                    valid_losses.append(loss.item())
                    # loss.detach().item()

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model, avg_train_losses, avg_valid_losses


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


def binarize_probs(probs, thresholds):
    nb_classes = probs.shape[-1]
    binarized_output = np.zeros_like(probs)

    for k in range(nb_classes):
        binarized_output[:, k] = (np.sign(probs[:, k] - thresholds[k]) + 1) // 2

    return binarized_output


def plot_loss(train_loss, valid_loss, save_path):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--title_path')
    # parser.add_argument('--abstract_path')
    # parser.add_argument('--label_path')
    # parser.add_argument('--mask_path')
    parser.add_argument('--dataset_path')
    
    parser.add_argument('----meSH_pair_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--graph')
    parser.add_argument('--save-model-path')
    parser.add_argument('--model_name', default='Full', type=str)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=200)
    parser.add_argument('--ksz', default=3)
    parser.add_argument('--hidden_gcn_size', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--atten_dropout', type=float, default=0.5)
    
    # lr -> 0.001, 0.0001, 0.0003, 0.0005
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=2)
    parser.add_argument('--lr_gamma', type=float, default=0.9)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    n_gpu = torch.cuda.device_count()  # check if it is multiple gpu
    print('{} gpu is avaliable'.format(n_gpu))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    # Get dataset and label graph & Load pre-trained embeddings
    if args.model_name == 'Full':
        # num_nodes, mlb, vocab, test_dataset, vectors, G = prepare_dataset(args.title_path, args.abstract_path,
        #                                                                   args.label_path, args.mask_path, args.meSH_pair_path,
        #                                                                   args.word2vec_path, args.graph, is_multichannel=True)
        num_nodes, mlb, vocab, train_dataset, valid_dataset, vectors, G = prepare_dataset(args.dataset_path, args.meSH_pair_path,
                                                                          args.word2vec_path, args.graph, is_multichannel=True)
        vocab_size = len(vocab)
        # Checking the Model
        # print("Debugging the model: ")
        model = multichannel_dilatedCNN_with_MeSH_mask(vocab_size, args.dropout, args.ksz, num_nodes, G, device,
                                                       args.embedding_dim, rnn_num_layers=2, cornet_dim=1000,
                                                       n_cornet_blocks=2)
        model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)
    elif args.model_name == 'ablation1':
        num_nodes, mlb, vocab, test_dataset, vectors, G = prepare_dataset(args.title_path, args.abstract_path,
                                                                          args.label_path, args.mask_path,
                                                                          args.meSH_pair_path,
                                                                          args.word2vec_path, args.graph,
                                                                          is_multichannel=False)
        vocab_size = len(vocab)
        model = single_channel_dilatedCNN(vocab_size, args.dropout, args.ksz, num_nodes, args.embedding_dim,
                                          rnn_num_layers=2, cornet_dim=1000, n_cornet_blocks=2)
        model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)
    elif args.model_name == 'ablation2':
        num_nodes, mlb, vocab, test_dataset, vectors, G = prepare_dataset(args.title_path, args.abstract_path,
                                                                          args.label_path, args.mask_path,
                                                                          args.meSH_pair_path,
                                                                          args.word2vec_path, args.graph,
                                                                          is_multichannel=True)
        vocab_size = len(vocab)
        model = multichannel_with_MeSH_mask(vocab_size, args.dropout, args.ksz, num_nodes, G, device, args.embedding_dim,
                                            rnn_num_layers=2, cornet_dim=1000, n_cornet_blocks=2)
        model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)
    elif args.model_name == 'ablation3':
        num_nodes, mlb, vocab, test_dataset, vectors, G = prepare_dataset(args.title_path, args.abstract_path,
                                                                          args.label_path, args.mask_path,
                                                                          args.meSH_pair_path,
                                                                          args.word2vec_path, args.graph,
                                                                          is_multichannel=True)
        vocab_size = len(vocab)
        model = multichannel_dilatedCNN_without_graph(vocab_size, args.dropout, args.ksz, num_nodes, args.embedding_dim,
                                                      rnn_num_layers=2, cornet_dim=1000, n_cornet_blocks=2)
        model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)
    elif args.model_name == 'ablation4':
        num_nodes, mlb, vocab, test_dataset, vectors, G = prepare_dataset(args.title_path, args.abstract_path,
                                                                          args.label_path, args.mask_path,
                                                                          args.meSH_pair_path,
                                                                          args.word2vec_path, args.graph,
                                                                          is_multichannel=True)
        vocab_size = len(vocab)
        model = multichannel_dilatedCNN(vocab_size, args.dropout, args.ksz, num_nodes, G, device, args.embedding_dim,
                                        rnn_num_layers=2, cornet_dim=1000, n_cornet_blocks=2)
        model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)
    elif args.model_name == 'HGCN4MeSH':
        num_nodes, mlb, vocab, train_dataset, valid_dataset, vectors, G = \
            prepare_dataset(args.title_path, args.abstract_path, args.label_path, args.mask_path, args.meSH_pair_path,
                            args.word2vec_path, args.graph, is_multichannel=True)
        vocab_size = len(vocab)

        model = HGCN4MeSH(vocab_size, args.dropout, args.ksz, args.embedding_dim, rnn_num_layers=2)
        model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)

    model.to(device)
    G = G.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCEWithLogitsLoss()

    # training
    print("Start training!")
    model, train_loss, valid_loss = train(train_dataset, valid_dataset, model, mlb, G, args.batch_sz,
                                          args.num_epochs, criterion, device, args.num_workers, optimizer, lr_scheduler,
                                          args.model_name)
    print('Finish training!')

    print('save model for inference')
    torch.save(model.state_dict(), args.save_model_path)


if __name__ == "__main__":
    main()
