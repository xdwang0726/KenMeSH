import argparse
import pickle

import ijson
import numpy as np
import torch
from tqdm import tqdm


def label_count(train_data_path, MeSH_id_pair_file):
    # get MeSH in each example
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []

    print('Start loading training data')
    for i, obj in enumerate(tqdm(objects)):
        try:
            mesh_id = obj['meshId']
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    # count number of nodes and get parent and children edges
    print('count number of nodes and get edges of the graph')
    node_count = len(mapping_id)
    values = list(mapping_id.values())

    class_freq = {}
    for doc in label_id:
        for label in doc:
            # label_id = mapping_id.get(label)
            if label in class_freq:
                class_freq[label] = class_freq[label] + 1
            else:
                class_freq[label] = 1

    train_labels = list(class_freq.keys())
    all_meshIDs = list(mapping_id.values())

    missing_mesh = list(set(all_meshIDs) - set(train_labels))

    neg_class_freq = {k: len(label_id) - v for k, v in class_freq.items()}
    save_data = dict(class_freq=class_freq, neg_class_freq=neg_class_freq)

    return save_data


def new_label_mapping(train_data_path, MeSH_id_pair_file, new_mesh_id_path):
    # get MeSH in each example
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []

    print('Start loading training data')
    for i, obj in enumerate(tqdm(objects)):
        try:
            mesh_id = obj['meshId']
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    flat_label = list(set([m for meshs in label_id for m in meshs]))
    print('len of mesh', len(flat_label))
    # get descriptor and MeSH mapped
    new_mapping = []
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            if value.strip() in flat_label:
                new_mapping.append(line)

    # count number of nodes and get parent and children edges
    print('count number of nodes and get edges of the graph %s' % len(new_mapping))
    with open(new_mesh_id_path, 'w') as f:
        for item in new_mapping:
            f.write("%s" % item)

    class_freq = {}
    for doc in label_id:
        for label in doc:
            if label in class_freq:
                class_freq[label] = class_freq[label] + 1
            else:
                class_freq[label] = 1

    # train_labels = list(class_freq.keys())
    # all_meshIDs = list(new_mapping)
    #
    # missing_mesh = list(set(all_meshIDs) - set(train_labels))
    # print('missing_mesh', missing_mesh)

    neg_class_freq = {k: len(label_id) - v for k, v in class_freq.items()}
    save_data = dict(class_freq=class_freq, neg_class_freq=neg_class_freq)

    return save_data


def get_tail_labels(train_data_path):
    # get MeSH in each example
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []

    print('Start loading training data')
    for i, obj in enumerate(tqdm(objects)):
        try:
            mesh_id = obj['meshId']
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    label_sample = {}
    for i, doc in enumerate(label_id):
        for label in doc:
            if label in label_sample:
                label_sample[label].append(i)
            else:
                label_sample[label] = []
                label_sample[label].append(i)

    label_set = list(label_sample.keys())
    num_labels = len(label_set)
    irpl = np.array([len(docs) for docs in list(label_sample.values())])
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    for i, label in enumerate(label_set):
        if irpl[i] > mir:
            tail_label.append(label)

    print('There are total %d tail labels' % len(tail_label))

    return tail_label


def get_label_negative_positive_ratio(train_data_path, MeSH_id_pair_file):

    # get MeSH in each example
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []

    print('Start loading training data')
    for i, obj in enumerate(tqdm(objects)):
        try:
            mesh_id = obj['meshId']
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    meshIDs = list(mapping_id.values())
    print('There are %d Meshs' % len(meshIDs))

    label_freq = {}
    for doc in label_id:
        for label in doc:
            if label in label_freq:
                label_freq[label] = label_freq[label] + 1
            else:
                label_freq[label] = 1
    pos = []
    for ids in meshIDs:
        if ids in list(label_freq.keys()):
            pos.append(list(label_freq.values())[list(label_freq.keys()).index(ids)])
        else:
            pos.append(0)

    num_examples = len(label_id)
    pos = np.array(pos)
    print('There are %d lables in total' % np.count_nonzero(pos))
    neg = num_examples - pos
    neg_pos_ratio = neg / pos
    neg_pos_ratio = torch.from_numpy(neg_pos_ratio).type(torch.float)
    return neg_pos_ratio


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--new_meSH_pair')
    parser.add_argument('--class_freq')

    args = parser.parse_args()

    save_data = label_count(args.train, args.meSH_pair_path)
    # save_data = new_label_mapping(args.train, args.meSH_pair_path, args.new_meSH_pair)
    with open(args.class_freq, 'wb') as f:
        pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
    # tail_labels = get_tail_labels(args.train)
    # pickle.dump(tail_labels, open(args.class_freq, 'wb'))
    # neg_pos_ratio = get_label_negative_positive_ratio(args.train, args.meSH_pair_path)
    # pickle.dump(neg_pos_ratio, open(args.class_freq, 'wb'))



if __name__ == "__main__":
    main()
