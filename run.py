import argparse
import os
import random
import socket
import sys
import pickle

import dgl
import ijson
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.utils.data.distributed
from dgl.data.utils import load_graphs
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext.vocab import Vectors
from tqdm import tqdm

from eval_helper import precision_at_ks, example_based_evaluation, micro_macro_eval, zero_division
from losses import *
from model import multichannel_dilatedCNN_with_MeSH_mask
from pytorchtools import EarlyStopping
from utils_multi import MeSH_indexing, pad_sequence, DistributedSamplerWrapper


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


def prepare_dataset(title_path, abstract_path, label_path, mask_path, MeSH_id_pair_file, word2vec_path, graph_file, num_example): #graph_cooccurence_file
    """ Load Dataset and Preprocessing """
    # load training data
    # f = open(train_data_path, encoding="utf8")
    # objects = ijson.items(f, 'articles.item')
    print('Start loading training data')
    mesh_mask = pickle.load(open(mask_path, 'rb'))

    all_title = pickle.load(open(title_path, 'rb'))
    all_text = pickle.load(open(abstract_path, 'rb'))
    label_id = pickle.load(open(label_path, 'rb'))

    # train_title = train_title[:num_example]
    # all_text = all_text[:num_example]
    # label_id = label_id[:num_example]
    # for i, obj in enumerate(tqdm(objects)):
    #     if i <= num_example:
    #         try:
    #             # ids = obj['pmid']
    #             heading = obj['title'].strip()
    #             heading = heading.translate(str.maketrans('', '', '[]'))
    #             text = obj['abstractText'].strip()
    #             text = text.translate(str.maketrans('', '', '[]'))
    #             mesh_id = obj['meshId']
    #             train_title.append(heading)
    #             all_text.append(text)
    #             label_id.append(mesh_id)
    #         except AttributeError:
    #             print(obj['pmid'].strip())
    #     else:
    #         break

    assert len(all_text[:num_example]) == len(all_title[:num_example]) #'title and abstract in the training set are not matching'
    print('Finish loading training data')
    print('number of training data %d' % len(all_text))

    # load test data
    print('Start loading test data')
    # f_t = open(test_data_path, encoding="utf8")
    # test_objects = ijson.items(f_t, 'articles.item')

    # test_title = train_title[-20000:]
    # test_text = all_text[-20000:]
    # test_label_id = label_id[-20000:]
    # test_mesh_mask = mesh_mask[-20000:]

    # for i, obj in enumerate(tqdm(test_objects)):
    #     if 610000 < i <= 620000:
    #         ids = obj['pmid']
    #         heading = obj['title'].strip()
    #         text = obj['abstractText'].strip()
    #         mesh_id = obj['meshID']
    #         journal = obj['journal'].split(',')
    #         neigh = obj['neighbors'].split(',')
    #         mesh = set(journal + neigh)
    #         test_pmid.append(ids)
    #         test_title.append(heading)
    #         test_text.append(text)
    #         test_label_id.append(mesh_id)
    #         test_mesh_mask.append(mesh)
    #     elif i > 620000:
    #         break
    print('number of test data %d' % len(all_title[-20000:]))

    print('load and prepare Mesh')
    # read full MeSH ID list
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    meshIDs = list(mapping_id.values())
    index_dic = {k: v for v, k in enumerate(meshIDs)}
    mesh_index = list(index_dic.values())
    print('Total number of labels %d' % len(meshIDs))

    mlb = MultiLabelBinarizer(classes=mesh_index)
    mlb.fit(mesh_index)

    # create Vector object map tokens to vectors
    print('load pre-trained BioWord2Vec')
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    # Preparing training and test datasets
    print('prepare training and test sets')
    dataset, test_dataset = MeSH_indexing(all_text[:num_example], label_id[:num_example], label_id[-20000:],
                                          mesh_mask[:num_example], mesh_mask[-20000:], label_id[-20000:],
                                          all_title[:num_example], all_title[-20000:], ngrams=1, vocab=None,
                                          include_unk=False, is_test=False, is_multichannel=True)

    # get validation set
    valid_size = 0.1
    indices = list(range(len(all_title[:num_example])))
    split = int(np.floor(valid_size * len(all_title[:num_example])))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # build vocab
    print('building vocab')
    vocab = dataset.get_vocab()

    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]
    # G_c = load_graphs(graph_cooccurence_file)[0][0]
    print('graph', G.ndata['feat'].shape)

    print('prepare dataset and labels graph done!')
    return len(meshIDs), mlb, vocab, dataset, test_dataset, vectors, G, train_sampler, valid_sampler # G_c


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
        mesh_mask = [entry[0] for entry in batch]

        abstract = [entry[1] for entry in batch]
        abstract_length = [len(seq) for seq in abstract]
        abstract = pad_sequence(abstract, ksz=3, batch_first=True)

        title = [entry[2] for entry in batch]
        title_length = []
        for i, seq in enumerate(title):
            if len(seq) == 0:
                length = len(seq) + 1
            else:
                length = len(seq)
            title_length.append(length)
        title = pad_sequence(title, ksz=3, batch_first=True)
        return mesh_mask, abstract, title, abstract_length, title_length


def train(train_dataset, train_sampler, valid_sampler, model, mlb, G, batch_sz, num_epochs, criterion, device,
          num_workers, optimizer, lr_scheduler, world_size, rank):
    _train_sampler = DistributedSamplerWrapper(train_sampler, num_replicas=world_size, rank=rank)

    train_data = DataLoader(train_dataset, batch_size=batch_sz, sampler=_train_sampler, collate_fn=generate_batch,
                            num_workers=num_workers)

    _valid_sampler = DistributedSamplerWrapper(valid_sampler, num_replicas=world_size, rank=rank)
    valid_data = DataLoader(train_dataset, batch_size=batch_sz, sampler=_valid_sampler,
                            collate_fn=generate_batch, num_workers=num_workers)

    num_lines = num_epochs * len(train_data)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=3, verbose=True)

    print("Training....")
    for epoch in range(num_epochs):
        model.train()  # prep model for training
        for i, (label, mask, abstract, title, abstract_length, title_length) in enumerate(train_data):
            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            mask = torch.from_numpy(mlb.fit_transform(mask)).type(torch.float)
            abstract_length = torch.Tensor(abstract_length)
            title_length = torch.Tensor(title_length)
            abstract, title, label, mask, abstract_length, title_length = abstract.to(device), title.to(device), label.to(device), mask.to(device), abstract_length.to(device), title_length.to(device)
            G = G.to(device)
            G.ndata['feat'] = G.ndata['feat'].to(device)
            # G_c = G_c.to(device)
            output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat']) #, G_c, G_c.ndata['feat'])
            # output = model(abstract, title, G.ndata['feat'])

            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())  # record training loss

            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 3000 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.8f} loss: {:3.8f}\n".format(
                        progress * 100, lr_scheduler.get_last_lr()[0], loss))
        # Adjust the learning rate
        lr_scheduler.step()

        model.eval()
        for i, (label, mask, abstract, title, abstract_length, title_length) in enumerate(valid_data):
            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            mask = torch.from_numpy(mlb.fit_transform(mask)).type(torch.float)
            abstract_length = torch.Tensor(abstract_length)
            title_length = torch.Tensor(title_length)
            abstract, title, label, mask, abstract_length, title_length = abstract.to(device), title.to(device), label.to(device), mask.to(device), abstract_length.to(device), title_length.to(device)
            G = G.to(device)
            G.ndata['feat'] = G.ndata['feat'].to(device)
            # G_c = G_c.to(device)
            output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat']) #, G_c, G_c.ndata['feat'])

            loss = criterion(output, label)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # print('[{} / {}] Train Loss: {.5f}, Valid Loss: {.5f}'.format(epoch+1, num_epochs, train_loss, valid_loss))
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


def test(test_dataset, model, mlb, G, batch_sz, device):
    test_data = DataLoader(test_dataset, batch_size=batch_sz, collate_fn=generate_batch, shuffle=False)
    # pred = torch.zeros(0).to(device)
    top_k_precisions = []
    sum_ebp = 0.
    sum_ebr = 0.
    sum_ebf = 0.
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    print('Testing....')
    model.eval()
    for label, mask, abstract, title, abstract_length, title_length in test_data:
        mask = torch.from_numpy(mlb.fit_transform(mask)).type(torch.float)
        abstract_length = torch.Tensor(abstract_length)
        title_length = torch.Tensor(title_length)
        mask, abstract, title, abstract_length, title_length = mask.to(device), abstract.to(device), title.to(device), abstract_length.to(device), title_length.to(device)
        G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)
        # G_c, G_c.ndata['feat'] = G_c.to(device), G_c.ndata['feat'].to(device)
        label = mlb.fit_transform(label)

        with torch.no_grad():
            output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat']) #, G_c, G_c.ndata['feat'])
            # output = model(abstract, title, G.ndata['feat'])
            # pred = torch.cat((pred, output), dim=0)

        # calculate precision at k
        pred = output.data.cpu().numpy()
        test_labelsIndex = getLabelIndex(label)
        precisions = precision_at_ks(pred, test_labelsIndex, ks=[1, 3, 5])
        top_k_precisions.append(precisions)
        # calculate example-based evaluation
        sums = example_based_evaluation(pred, label, threshold=0.5)
        sum_ebp += sums[0]
        sum_ebr += sums[1]
        sum_ebf += sums[2]
        # calculate label-based evaluation
        confusion = micro_macro_eval(pred, label, threshold=0.5)
        tp += confusion[0]
        tn += confusion[1]
        fp += confusion[2]
        fn += confusion[3]

    # Evaluations
    print('Calculate Precision at K...')
    p_at_1 = np.mean(flatten(p_at_k[0] for p_at_k in top_k_precisions))
    p_at_3 = np.mean(flatten(p_at_k[1] for p_at_k in top_k_precisions))
    p_at_5 = np.mean(flatten(p_at_k[2] for p_at_k in top_k_precisions))
    for k, p in zip([1, 3, 5], [p_at_1, p_at_3, p_at_5]):
        print('p@{}: {:.5f}'.format(k, p))

    print('Calculate Example-based Evaluation')
    ebp = sum_ebp / len(test_dataset)
    ebr = sum_ebp / len(test_dataset)
    ebf = sum_ebp / len(test_dataset)
    for n, m in zip(['EBP', 'EBR', 'EBF'], [ebp, ebr, ebf]):
        print('{}: {:.5f}'.format(n, m))

    print('Calculate Label-based Evaluation')
    mip = zero_division(np.sum(tp), (np.sum(tp) + np.sum(fp)))
    mir = zero_division(np.sum(tp), (np.sum(tp) + np.sum(fn)))
    mif = zero_division(2 * mir * mip, (mir + mip))
    for n, m in zip(['MiP', 'MiR', 'MiF'], [mip, mir, mif]):
        print('{}: {:.5f}'.format(n, m))

    print('###################DONE#########################')


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
    # parser.add_argument('--train_path')
    # parser.add_argument('--test_path')
    parser.add_argument('--title_path')
    parser.add_argument('--abstract_path')
    parser.add_argument('--label_path')
    parser.add_argument('--mask_path')
    parser.add_argument('----meSH_pair_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--graph')
    parser.add_argument('--graph_cooccurence')
    parser.add_argument('--results')
    parser.add_argument('--save-model-path')
    parser.add_argument('--loss')

    parser.add_argument('--num_example', type=int, default=10000)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=200)
    parser.add_argument('--ksz', default=3)
    parser.add_argument('--hidden_gcn_size', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--atten_dropout', type=float, default=0.5)

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=2)
    parser.add_argument('--lr_gamma', type=float, default=0.9)

    parser.add_argument('--init_method', type=str, default='tcp://127.0.0.1:3456')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

    args = parser.parse_args()
    set_seed(0)
    ngpus_per_node = torch.cuda.device_count()
    print('number of gpus per node: %d' % ngpus_per_node)
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + int(local_rank)

    available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',',""))  # check if it is multiple gpu
    print('available gpus: ', available_gpus)
    current_device = int(available_gpus[local_rank])
    torch.cuda.set_device(current_device)

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    # init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=world_size, rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))
    # Get dataset and label graph & Load pre-trained embeddings
    num_nodes, mlb, vocab, train_dataset, test_dataset, vectors, G, train_sampler, valid_sampler = prepare_dataset(
        args.title_path, args.abstract_path, args.label_path, args.mask_path, args.meSH_pair_path, args.word2vec_path,
        args.graph, args.num_example) # args. graph_cooccurence,

    vocab_size = len(vocab)
    model = multichannel_dilatedCNN_with_MeSH_mask(vocab_size, args.dropout, args.ksz, num_nodes, G, current_device,
                                    embedding_dim=200, rnn_num_layers=2, cornet_dim=1000, n_cornet_blocks=2)
                                    #gat_num_heads=8, gat_num_layers=2, gat_num_out_heads=1)
    model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).cuda()
    # model.embedding_layer.weight.data.copy_(vectors.vectors).to(device)
    # model = multichannle_attenCNN(vocab_size, args.nKernel, args.ksz, args.add_original_embedding,
    #                        args.atten_dropout, embedding_dim=args.embedding_dim)
    #
    # model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device], output_device=current_device)
    print('From Rank: {}, ==> Preparing data..'.format(rank))
    # G = G.to(device)
    # G = dgl.add_self_loop(G)
    # G_c.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion = FocalLoss()
    # criterion = AsymmetricLossOptimized()

    # training
    print("Start training!")
    model, train_loss, valid_loss = train(train_dataset, train_sampler, valid_sampler, model, mlb, G, args.batch_sz,
                                          args.num_epochs, criterion, current_device, args.num_workers, optimizer,
                                          lr_scheduler, world_size, rank)
    print('Finish training!')

    # visualize the loss as the network trained
    plot_loss(train_loss, valid_loss, args.loss)

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    # }, args.save_parameter_path)
    # print('save model')
    # torch.save(model, args.save_model_path)

    # load model
    # model = torch.load(args.model_path)
    #
    # testing
    test(test_dataset, model, mlb, G, args.batch_sz, current_device)


if __name__ == "__main__":
    main()
