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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from model import Bert_GCN, Bert_Baseline
from utils import bert_MeSH
from eval_helper import precision_at_ks, example_based_evaluation, micro_macro_eval
from transformers import AutoTokenizer, AutoConfig
import transformers
from threshold_opt import eval
import socket
import torch.distributed as dist
import torch.multiprocessing as mp


def prepare_dataset(train_data_path, test_data_path, MeSH_id_pair_file, graph_file, tokenizer):
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
        if i <= 10000:
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
    train_dataset, test_dataset = bert_MeSH(all_text, label_id, test_text, test_label, tokenizer=tokenizer, max_len=512)

    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]

    print('graph', G.ndata['feat'].shape)

    print('prepare dataset and labels graph done!')
    return len(meshIDs), mlb, train_dataset, test_dataset, G


def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    input_ids = [torch.tensor(entry['input_ids']) for entry in batch]
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = [torch.tensor(entry['attention_mask']) for entry in batch]
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    label = [entry['label'] for entry in batch]
    return input_ids, attention_mask, label


def train(train_dataset, model, mlb, G, batch_sz, num_epochs, criterion, device, optimizer, lr_scheduler):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data = DataLoader(train_dataset, batch_size=batch_sz, collate_fn=generate_batch)

    num_lines = num_epochs * len(train_data)

    print("Training....")
    for epoch in range(num_epochs):
        for i, data in enumerate(train_data):
            input_ids, attention_mask, label = data
            label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
            input_ids, attention_mask, label = input_ids.cuda(), attention_mask.cuda(), label.cuda()
            # output = model(input_ids, attention_mask, G, G.ndata['feat'])
            # output = model(input_ids, attention_mask, G.ndata['feat'])
            output = model(input_ids, attention_mask)

            # training precision@k
            # original_label = mlb.fit_transform(label).cpu()
            # pred = output.data.cpu().numpy()
            # labelsIndex = getLabelIndex(original_label)
            # precision = precision_at_ks(pred, labelsIndex, ks=[1, 3, 5])

            optimizer.zero_grad()
            loss = criterion(output, label)
            # print('loss', loss)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters())
            optimizer.step()
            # print('Allocated2:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            processed_lines = i + len(train_data) * epoch
            progress = processed_lines / float(num_lines)
            if processed_lines % 128 == 0:
                sys.stderr.write(
                    "\rProgress: {:3.0f}% lr: {:3.8f} loss: {:3.8f}\n".format(
                        progress * 100, lr_scheduler.get_last_lr()[0], loss))
            # print(optimizer.param_groups[0]['lr'])
            # for k, p in zip([1, 3, 5], precision):
            #     print('p@{}: {:.5f}'.format(k, p))
        # Adjust the learning rate
        lr_scheduler.step()
        # print('Allocated3:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')


def test(test_dataset, model, G, batch_sz, device):
    test_data = DataLoader(test_dataset, batch_size=batch_sz, collate_fn=generate_batch)
    pred = torch.zeros(0).cuda()
    ori_label = []
    print('Testing....')
    for data in test_data:
        input_ids, attention_mask, label = data
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
        print('test_orig', label, '\n')
        ori_label.append(label)
        flattened = [val for sublist in ori_label for val in sublist]
        with torch.no_grad():
            # output = model(input_ids, attention_mask, G, G.ndata['feat'])
            # output = model(input_ids, attention_mask, G.ndata['feat'])
            output = model(input_ids, attention_mask)

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


def run(dev_id, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # initialize the distributed training
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    dist.init_process_group(backend='nccl', init_method='tcp://{}:{}'.format(ip_address, args.port),
                            world_size=args.world_size, rank=dev_id)
    # delete world_size / rank
    # random port number
    gpu_rank = dist.get_rank()
    assert gpu_rank == dev_id
    main(dev_id, args)


def main(dev_id, args):
    device = torch.device('cuda:{}'.format(dev_id))
    # set current device
    torch.cuda.set_device(device)

    # prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.biobert)
    bert_config = AutoConfig.from_pretrained(args.biobert)
    # Get dataset and label graph & Load pre-trained embeddings
    num_nodes, mlb, train_dataset, test_dataset, G = prepare_dataset(args.train_path,
                                                                     args.test_path,
                                                                     args.meSH_pair_path,
                                                                     args.graph, tokenizer)

    # create model
    model = Bert_Baseline(bert_config, num_nodes)
    model.to(device)
    G.to(device)
    # wrap the model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], find_unused_parameters=True)

    # loss function
    criterion = nn.BCELoss().cuda(dev_id)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)

    # training
    print("Start training!")
    train(train_dataset, model, mlb, G, args.batch_sz, args.num_epochs, criterion, device, optimizer, lr_scheduler)
    print('Finish training!')
    # testing
    results, test_labels = test(test_dataset, model, G, args.batch_sz, device)

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
    print("MaP@5, MiP@5, MaR@5, MiR@5, MaF@5, MiF@5: ")
    for measure in label_measure_5:
        print(measure, ",")

    micro_precision, micro_recall, micro_f_score = eval(test_label_transform, pred, num_nodes, len(pred))
    print(micro_precision, micro_recall, micro_f_score)

    # def main():
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--train_path')
    #     parser.add_argument('--test_path')
    #     parser.add_argument('----meSH_pair_path')
    #     parser.add_argument('--word2vec_path')
    #     parser.add_argument('--meSH_pair_path')
    #     parser.add_argument('--mesh_parent_children_path')
    #     parser.add_argument('--graph')
    #     parser.add_argument('--results')
    #     parser.add_argument('--save-model-path')
    #
    #     parser.add_argument('--device', default='cuda', type=str)
    #     parser.add_argument('--hidden_gcn_size', type=int, default=768)
    #     parser.add_argument('--embedding_dim', type=int, default=200)
    #     parser.add_argument('--biobert', type=str)
    #
    #     parser.add_argument('--num_epochs', type=int, default=3)
    #     parser.add_argument('--batch_sz', type=int, default=32)
    #     parser.add_argument('--num_workers', type=int, default=1)
    #     parser.add_argument('--bert_lr', type=float, default=2e-5)
    #     parser.add_argument('--lr', type=float, default=5e-5)
    #     parser.add_argument('--momentum', type=float, default=0.9)
    #     parser.add_argument('--weight_decay', type=float, default=0.01)
    #     parser.add_argument('--scheduler_step_sz', type=int, default=5)
    #     parser.add_argument('--lr_gamma', type=float, default=0.1)
    #
    #     parser.add_argument('--port', type=str, default='20000')
    #     parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    #     parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    #     parser.add_argument('--local_rank', default=0, type=int, help='rank of distributed processes')
    #     # parser.add_argument('--fp16', default=True, type=bool)
    #     # parser.add_argument('--fp16_opt_level', type=str, default='O0')
    #
    #     args = parser.parse_args()
    #
    #     # n_gpu = torch.cuda.device_count()  # check if it is multiple gpu
    #     # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #     # # device = torch.device(args.device)
    #     # logging.info('Device:'.format(device))
    #     #
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    #     # # initialize the distributed training
    #     # hostname = socket.gethostname()
    #     # ip_address = socket.gethostbyname(hostname)
    #     # dist.init_process_group(backend=args.dist_backend, init_method='tcp://{}:{}'.format(ip_address, args.port),
    #     #                         world_size=args.world_size, rank=args.local_rank)
    #     # gpu_rank = torch.distributed.get_rank()
    #     #
    #     # tokenizer = AutoTokenizer.from_pretrained(args.biobert)
    #     # bert_config = AutoConfig.from_pretrained(args.biobert)
    #     # # Get dataset and label graph & Load pre-trained embeddings
    #     # num_nodes, mlb, train_dataset, test_dataset, G = prepare_dataset(args.train_path,
    #     #                                                                  args.test_path,
    #     #                                                                  args.meSH_pair_path,
    #     #                                                                  args.graph, tokenizer)
    #     #
    #     #
    #     # # model = Bert_GCN(bert_config, num_nodes)
    #     #
    #     # model = Bert_Baseline(bert_config, num_nodes)
    #     # # model = Bert_GCN(bert_config, num_nodes)
    #     # # model = Bert(bert_config, embedding_dim=args.embedding_dim)
    #     # # model = nn.DataParallel(model.cuda(), device_ids=[0, 1, 2, 3])
    #     # model.to(device)
    #     # G.to(device)
    #     # # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
    #     # #                                                   output_device=args.local_rank)
    #     # model = nn.DataParallel(model.cuda(), device_ids=[0, 1, 2, 3])
    #     #
    #     # # bert_params = list(map(id, model.bert.parameters()))
    #     # # base_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    #     # # layer_list = ['bert.weight', 'bert.bias']
    #     # # bert_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
    #     # # base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
    #     # # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #     # # optimizer = torch.optim.AdamW([{'params': bert_params, 'lr': args.bert_lr}, {'params': base_params, 'lr': args.lr}],
    #     # #                               lr=args.lr, weight_decay=args.weight_decay)
    #     #
    #     # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_sz, gamma=args.lr_gamma)
    #     # criterion = nn.BCELoss()
    #     #
    #     # # training
    #     # print("Start training!")
    #     # train(train_dataset, model, mlb, G, args.batch_sz, args.num_epochs, criterion, device, optimizer, lr_scheduler)
    #     # print('Finish training!')
    #     # # testing
    #     # results, test_labels = test(test_dataset, model, G, args.batch_sz, device)
    #     # # print('predicted:', results, '\n')

    mp.spawn(main, nprocs=args.gpus, args=(args,))



if __name__ == "__main__":
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
    parser.add_argument('--hidden_gcn_size', type=int, default=768)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--biobert', type=str)

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler_step_sz', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.1)

    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--port', type=str, default='20000')
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int, help='rank of distributed processes')
    # parser.add_argument('--fp16', default=True, type=bool)
    # parser.add_argument('--fp16_opt_level', type=str, default='O0')
    args = parser.parse_args()

    devices = list(map(int, args.gpus.split(',')))
    args.ngpu = len(devices)
    mp = torch.multiprocessing.get_context('spawn')
    procs = []
    for dev_id in devices:
        procs.append(mp.Process(target=run, args=(dev_id, args),
                                daemon=True))
        procs[-1].start()
    for p in procs:
        p.join()
