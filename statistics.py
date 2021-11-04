import argparse

import ijson
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import mutual_info_score
import numpy as np
import matplotlib.pyplot as plt


def get_label_dictionary(file):
    f = open(file, encoding="utf8", errors='ignore')
    objects = ijson.items(f, 'articles.item')
    all_label = []
    label_id = []
    for obj in tqdm(objects):
        try:
            original_label = obj["meshMajor"]
            mesh_id = obj['meshId']
            all_label.append(original_label)
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    occurrence_counts = dict(Counter(x for labels in all_label for x in labels))
    return occurrence_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json')
    parser.add_argument('--test_json')
    parser.add_argument('--save')
    args = parser.parse_args()

    train_counts = get_label_dictionary(args.train_json)
    test_counts = get_label_dictionary(args.test_json)

    train = []
    test = []
    for k in sorted(train_counts.keys() & test_counts.keys()):
        train.append(train_counts[k])
        test.append(test_counts[k])

    train = [x / len(train_counts.keys()) for x in train]
    test = [x / len(test_counts.keys()) for x in test]

    mutual_score = mutual_info_score(train, test)
    print('mutual score:', mutual_score)

    def KL(a, b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)

        return np.sum(np.where(a != 0, a * np.log(a / b), 0))

    KL = KL(train, test)
    print('KL distance: ', KL)
    # plots
    # less_than_100 = []
    # between_100_and_500 = []
    # between_500_and_1000 = []
    # between_1000_and_5000 = []
    # grater_than_5000 = []
    # for key in occurrence_counts.keys():
    #     if occurrence_counts.get(key) < 100:
    #         less_than_100.append(key)
    #     elif 100 <= occurrence_counts.get(key) < 500:
    #         between_100_and_500.append(key)
    #     elif 500 <= occurrence_counts.get(key) < 1000:
    #         between_500_and_1000.append(key)
    #     elif 1000 <= occurrence_counts.get(key) < 5000:
    #         between_1000_and_5000.append(key)
    #     elif occurrence_counts.get(key) >= 5000:
    #         grater_than_5000.append(key)
    #
    # mesh_demographic = {'less_than_100': len(less_than_100),
    #                     'between_100_and_500': len(between_100_and_500),
    #                     'between_500_and_1000': len(between_500_and_1000),
    #                     'between_1000_and_5000': len(between_1000_and_5000),
    #                     'grater_than_5000': len(grater_than_5000)
    #                     }
    # mesh_range = list(mesh_demographic.keys())
    # numbers = list(mesh_demographic.values())
    #
    # fig = plt.figure(figsize=(5, 20))
    #
    # # creating the bar plot
    # plt.bar(mesh_range, numbers, color='maroon', width=0.4)
    #
    # plt.xlabel("MeSH range")
    # plt.ylabel("Number of MeSH Terms in Each Range")
    # plt.title("MeSH Demographics")
    # plt.savefig(args.save, dpi=400)
    # plt.show()


if __name__ == "__main__":
    main()
