import argparse
import ijson
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import MultiLabelBinarizer


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--meSH_pair_path')

    args = parser.parse_args()

    save_data = label_count(args.train, args.meSH_pair_path)
    with open('class_freq.pkl', 'wb') as f:
        pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()