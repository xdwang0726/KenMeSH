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
        if i <= 100:
            try:
                journal = obj['journal']
                mesh_id = obj['meshId']
                label_id.append(mesh_id)
                journals.append(journal)
            except AttributeError:
                print(obj["pmid"].strip())
        else:
            break

    journal_dict = {}
    mesh_counts = {'journal': list()}
    for i, journal in enumerate(journals):
        journal_info = dict.fromkeys(['counts', 'mesh_counts'])
        mesh_counts['journal'] = mesh_counts['journal'].append(label_id[i])
        if journal in journal_dict:
            journal_dict[journal]['counts'] = journal_info['counts'] + 1
        else:
            journal_info[journal]['counts'] = 1

    for i, ids in enumerate(list(mesh_counts.values())):
        flat_list = []
        for item in ids:
            flat_list.append(item)
        occurrences = Counter(flat_list)
        journal_name = list(mesh_counts.keys())[i]
        if journal_name in journal_dict:
            journal_dict[journal_name]['mesh_counts'] = occurrences
        else:
            print(journal_name, 'is not in the list')

    return journal_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data')

    args = parser.parse_args()

    journal_info = journal_stats(args.data)
    with open('journal_info.pkl', 'wb') as f:
        pickle.dump(journal_info, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
