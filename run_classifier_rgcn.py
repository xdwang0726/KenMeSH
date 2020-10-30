import argparse
import logging
import os
import pickle
import sys
import time

import dgl
import ijson
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors
from tqdm import tqdm

from eval_helper import precision_at_ks, example_based_evaluation, perf_measure
from model import MeSH_RGCN, RelGraphEmbedLayer
from utils import MeSH_indexing
import torch.multiprocessing as mp


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
        if i <= 20000:
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
    G = dgl.load_graphs(graph_file)[0][0]

    print('graph', G.ndata['feat'].shape)

    print('prepare dataset and labels graph done!')
    return len(meshIDs), mlb, vocab, train_dataset, test_dataset, vectors, G


class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    g : DGLHeterograph
        Full graph
    target_idx : tensor
        The target training node IDs in g
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """

    def __init__(self, g, target_idx, fanouts):
        self.g = g
        self.target_idx = target_idx
        self.fanouts = fanouts

    """Do neighbor sample
    Parameters
    ----------
    seeds :
        Seed nodes
    Returns
    -------
    tensor
        Seed nodes, also known as target nodes
    blocks
        Sampled subgraphs
    """

    def sample_blocks(self, seeds):
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = torch.tensor(seeds).long()
        cur = self.target_idx[seeds]
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            etypes = self.g.edata[dgl.ETYPE][frontier.edata[dgl.EID]]
            block = dgl.to_block(frontier, cur)
            block.srcdata[dgl.NTYPE] = self.g.ndata[dgl.NTYPE][block.srcdata[dgl.NID]]
            block.srcdata['type_id'] = self.g.ndata[dgl.NID][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks


def weight_matrix(vocab, vectors, dim=200):
    weight_matrix = np.zeros([len(vocab.itos), dim])
    for i, token in enumerate(vocab.stoi):
        try:
            weight_matrix[i] = vectors.__getitem__(token)
        except KeyError:
            weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
    return torch.from_numpy(weight_matrix)


def train(model, embed_layer, data_loader, node_feats):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []

    with torch.no_grad():
        for sample_data in tqdm(data_loader):
            torch.cuda.empty_cache()
            seeds, blocks = sample_data
            feats = feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                        blocks[0].srcdata[dgl.NTYPE],
                                        blocks[0].srcdata['type_id'],
                                        node_feats)
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())
    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    return eval_logits, eval_seeds


def run(proc_id, n_gpus, args, devices, queue=None):
    dev_id = devices[proc_id]

    num_nodes, mlb, vocab, train_dataset, test_dataset, vectors, G = prepare_dataset(args.train_path,
                                                                                     args.test_path,
                                                                                     args.meSH_pair_path,
                                                                                     args.word2vec_path, args.graph)
    vocab_size = len(vocab)
    node_feats = G.ndata['feat']
    num_of_ntype = 1
    num_rels = 2

    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    category_id = 0
    node_ids = torch.arange(G.number_of_nodes())
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    train_sampler = NeighborSampler(G, target_idx, fanouts)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_sampler.sample_blocks,
                            shuffle=True,
                            num_workers=args.num_workers)

    test_sampler = NeighborSampler(G, target_idx, [None] * args.n_layers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_sampler.sample_blocks,
                             shuffle=False,
                             num_workers=args.num_workers)

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        backend = 'nccl'

        # using sparse embedding or usig mix_cpu_gpu model (embedding model can not be stored in GPU)
        if args.sparse_embedding or args.mix_cpu_gpu:
            backend = 'gloo'
        torch.distributed.init_process_group(backend=backend,
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=dev_id)

        # node features
        # None for one-hot feature, if not none, it should be the feature tensor.
        #
    embed_layer = RelGraphEmbedLayer(dev_id,
                                     g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     args.n_hidden,
                                     sparse_emb=args.sparse_embedding)

    # create model
    # all model params are in device.
    model = MeSH_RGCN(vocab_size, args.nKernel, args.ksz, args.hidden_rgcn_size, num_nodes, dev_id, embedding_dim=200)

    if dev_id >= 0 and n_gpus == 1:
        torch.cuda.set_device(dev_id)
        model.cuda(dev_id)
        # embedding layer may not fit into GPU, then use mix_cpu_gpu
        if args.mix_cpu_gpu is False:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        model.cuda(dev_id)
        if args.mix_cpu_gpu:
            embed_layer = DistributedDataParallel(embed_layer, device_ids=None, output_device=None)
        else:
            embed_layer.cuda(dev_id)
            embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    if dev_id >= 0 and n_gpus == 1:
        torch.cuda.set_device(dev_id)
        model.cuda(dev_id)
        # embedding layer may not fit into GPU, then use mix_cpu_gpu
        if args.mix_cpu_gpu is False:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        model.cuda(dev_id)
        if args.mix_cpu_gpu:
            embed_layer = DistributedDataParallel(embed_layer, device_ids=None, output_device=None)
        else:
            embed_layer.cuda(dev_id)
            embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    # optimizer
    if args.sparse_embedding:
        dense_params = list(model.parameters())
        if args.node_feats:
            if n_gpus > 1:
                dense_params += list(embed_layer.module.embeds.parameters())
            else:
                dense_params += list(embed_layer.embeds.parameters())
        optimizer = torch.optim.Adam(dense_params, lr=args.lr, weight_decay=args.l2norm)
        if n_gpus > 1:
            emb_optimizer = torch.optim.SparseAdam(embed_layer.module.node_embeds.parameters(), lr=args.lr)
        else:
            emb_optimizer = torch.optim.SparseAdam(embed_layer.node_embeds.parameters(), lr=args.lr)
    else:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    num_lines = args.num_epochs * len(train_data)
    for epoch in range(args.num_epochs):
        model.train()
        embed_layer.train()

        for i, sample_data in enumerate(train_data):
            seeds, blocks = sample_data
            t0 = time.time()
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model(blocks, feats)
            loss = nn.BCELoss(logits, labels[seeds])
            t1 = time.time()
            optimizer.zero_grad()
            if args.sparse_embedding:
                emb_optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            if args.sparse_embedding:
                emb_optimizer.step()
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)

            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.8f} loss: {:3.8f}\n".format(
                        progress * 100, loss))

        if n_gpus > 1:
            torch.distributed.barrier()

        # only process 0 will do the evaluation
        if (queue is not None) or (proc_id == 0):
            test_logits, test_seeds = evaluate(model, embed_layer, test_loader, node_feats)
            if queue is not None:
                queue.put((test_logits, test_seeds))

            # gather evaluation result from multiple processes
            if proc_id == 0:
                if queue is not None:
                    test_logits = []
                    test_seeds = []
                    for i in range(n_gpus):
                        log = queue.get()
                        test_l, test_s = log
                        test_logits.append(test_l)
                        test_seeds.append(test_s)
                    test_logits = torch.cat(test_logits)
                    test_seeds = torch.cat(test_seeds)
                test_loss = nn.BCELoss(test_logits, labels[test_seeds].cpu()).item()
                test_acc = torch.sum(test_logits.argmax(dim=1) == labels[test_seeds].cpu()).item() / len(test_seeds)
                print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))
                print()
        # sync for test
        if n_gpus > 1:
            torch.distributed.barrier()



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

    parser.add_argument('--gpu', type=str, default=['0', '1'], help="Comma separated list of GPU device IDs.")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    parser.add_argument('--nKernel', type=int, default=200)
    parser.add_argument('--ksz', type=list, default=[3, 4, 5])
    parser.add_argument('--hidden_rgcn_size', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_sz', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.1)

    args = parser.parse_args()

    device = list(map(int, args.gpu.split(',')))

    # unpack data
    num_nodes, mlb, vocab, train_dataset, test_dataset, vectors, G = prepare_dataset(args.train_path,
                                                                                     args.test_path, args.meSH_pair_pat
    args.word2vec_path, args.graph)
    vocab_size = len(vocab)
    print('vocab_size:', vocab_size)

    node_feats = G.ndata['feat']
    num_of_ntype = 1
    num_rels = 2

    # calculate norm
    if args.global_norm is False:
        for canonical_etype in G.canonical_etypes:
            u, v, eid = G.all_edges(form='all', etype=canonical_etype)
            _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = torch.ones(eid.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            G.edges[canonical_etype].data['norm'] = norm

    g = dgl.to_homogeneous(G, edata=['norm'])

    g.ndata[dgl.NTYPE].share_memory_()
    g.edata[dgl.ETYPE].share_memory_()
    g.edata['norm'].share_memory_()
    node_ids = torch.arange(g.number_of_nodes())

    g.create_formats_()

    n_gpus = len(device)
    if n_gpus == 1:
        run(0, n_gpus, args, device, (g,))
    else:
        queue = mp.Queue(n_gpus)
        procs = []
        num_train_examples =
        num_test_examples =
        train_seeds = torch.randperm(num_train_examples)
        test_seeds = torch.randperm(num_test_examples)
        train_seeds_per_proc = num_train_examples // n_gpus
        test_seeds_per_proc = num_test_examples // n_gpus
        for proc_id in range(n_gpus):
            # we have multi-gpu for training and testing
            # so split trian set, valid set and test set into num-of-gpu parts.
            proc_train_seeds = train_seeds[proc_id * train_seeds_per_proc:
                                           (proc_id + 1) * train_seeds_per_proc \
                                               if (proc_id + 1) * train_seeds_per_proc < num_train_examples \
                                               else num_train_examples]
            proc_test_seeds = test_seeds[proc_id * test_seeds_per_proc:
                                         (proc_id + 1) * test_seeds_per_proc \
                                             if (proc_id + 1) * test_seeds_per_proc < num_test_examples \
                                             else num_test_examples]
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices,
                                             (g, node_feats, num_of_ntype, num_nodes, num_rels, target_idx,
                                              train_idx, val_idx, test_idx, labels),
                                             (proc_train_seeds, proc_test_seeds),
                                             queue))

            p.start()
            procs.append(p)
        for p in procs:
            p.join()



    model = MeSH_RGCN(vocab_size, args.nKernel, args.ksz, args.hidden_gcn_size, num_nodes, args.embedding_dim)
    model.content_feature.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))
    model = model.to(device)
    G.to(device)

    # if n_gpus > 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    criterion = nn.BCELoss()

    # training
    print("Start training!")
    train(train_dataset, model, mlb, G, args.batch_sz, args.num_epochs, criterion, device, args.num_workers,
          optimizer, lr_scheduler)
    print('Finish training!')
    # testing
    results, test_labels = test(test_dataset, model, G, args.batch_sz, device)

    # if n_gpus == 1:
    #     run(0, n_gpus, args, devices, data)
    # else:
    #     procs = []
    #     for proc_id in range(n_gpus):
    #         p = mp.Process(target=thread_wrapped_func(run),
    #                        args=(proc_id, n_gpus, args, devices, data))
    #         p.start()
    #         procs.append(p)
    #     for p in procs:
    #         p.join()










    # dist.init_process_group(backend='nccl', init_method='file:///mnt/nfs/sharedfile', world_size=1, rank=0)
    # dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip=args.master_ip, master_port=args.master_port)
    # dist.init_process_group(backend='nccl', world_size=args.world_size, rank=0)
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda")
    # dev0 = torch.device('cuda:0')
    # dev1 = torch.device('cuda:1')

    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=1)
    # device_ids = [0, 1]
    # print('num of gpus per node:', os.environ['CUDA_VISIBLE_DEVICES'])
    #
    # # device = torch.device(args.device if torch.cuda.is_available() else "cpu", args.local_rank)
    # # device = torch.device(args.device)
    # logging.info('Device:'.format(device))
    #
    # # Get dataset and label graph & Load pre-trained embeddings
    # num_nodes, mlb, vocab, train_dataset, test_dataset, vectors, G = prepare_dataset(args.train_path,
    #                                                                                  args.test_path, args.meSH_pair_path,
    #                                                                                  args.word2vec_path, args.graph)
    #
    # vocab_size = len(vocab)
    # print('vocab_size:', vocab_size)
    # # model = MeSH_GCN(vocab_size, args.nKernel, args.ksz, args.hidden_gcn_size, args.embedding_dim)
    # model = MeSH_RGCN(vocab_size, args.nKernel, args.ksz, args.hidden_gcn_size, num_nodes, dev0, dev1,
    #                   args.embedding_dim)
    # # model = ContentsExtractor(vocab_size, args.nKernel, args.ksz, 29368, 200)
    # # torch.distributed.init_process_group(backend="nccl")
    #
    #
    # # model.cnn.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))
    # model.content_feature.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors))
    #
    # model.cuda()
    # # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # G.create_formats_()
    # G.to(devices)
    #
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.parallel.DistributedDataParallel(model)
    #     # device_ids will include all GPU devices by default
    #     print('model parallel done!')
    #
    # # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    # criterion = nn.BCELoss()
    #
    # # training
    # print("Start training!")
    # train(train_dataset, model, mlb, G, args.batch_sz, args.num_epochs, criterion, dev0, dev1, args.num_workers,
    #       optimizer,
    #       lr_scheduler)
    # print('Finish training!')
    # # testing
    # results, test_labels = test(test_dataset, model, G, args.batch_sz, dev0, dev1)
    # # print('predicted:', results, '\n')

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
    label_measure_5 = perf_measure(test_label_transform, top_5_pred)
    print("MaP@5, MiP@5, MaF@5, MiF@5: ")
    for measure in label_measure_5:
        print(measure, ",")


if __name__ == "__main__":
    main()
