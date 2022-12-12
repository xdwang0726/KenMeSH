import argparse
import pickle
from collections import Counter

import ijson
from tqdm import tqdm


def journal_stats(data_path):
    f = open(data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []
    journals = []
    for i, obj in enumerate(tqdm(objects)):
        try:
            journal = obj['journal']
            mesh_id = obj['mesh']
            label_id.append(mesh_id)
            journals.append(journal)
        except AttributeError:
            print(obj["pmid"].strip())

    journal_dict = {}
    mesh_counts = {}
    for i, journal in enumerate(journals):
        if journal in journal_dict:
            journal_dict[journal]['counts'] = journal_dict[journal]['counts'] + 1
            mesh_counts[journal].append(label_id[i])
        else:
            journal_dict[journal] = dict.fromkeys(['counts', 'mesh_counts'])
            journal_dict[journal]['counts'] = 1
            mesh_counts[journal] = [label_id[i]]

    for i, ids in enumerate(list(mesh_counts.values())):
        flat_list = [item for sublist in ids for item in sublist]
        occurrences = dict(Counter(flat_list))
        journal_name = list(mesh_counts.keys())[i]
        if journal_name in journal_dict:
            sorted_occurrence = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True))
            journal_dict[journal_name]['mesh_counts'] = sorted_occurrence
        else:
            print(journal_name, 'is not in the list')

    sorted_journal = dict(sorted(journal_dict.items(), key=lambda item: item[1]['counts'], reverse=True))
    return sorted_journal


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--save')

    args = parser.parse_args()

    journal_info = journal_stats(args.data)
    # print(journal_info)

    with open(args.save, 'wb') as f:
        pickle.dump(journal_info, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
