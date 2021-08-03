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
        if i <= 1000:
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
    mesh_counts = {}
    for i, journal in enumerate(journals):
        if journal in journal_dict:
            journal_dict[journal]['counts'] = journal_dict[journal]['counts'] + 1
            if journal in mesh_counts:
                mesh_counts[journal].append(label_id[i])
            else:
                mesh_counts[journal] = [label_id[i]]
                print('true')
        else:
            journal_dict[journal] = dict.fromkeys(['counts', 'mesh_counts'])
            journal_dict[journal]['counts'] = 1
            if journal in mesh_counts:
                print('false')
                mesh_counts[journal].append(label_id[i])
            else:
                mesh_counts[journal] = [label_id[i]]

    print('mesh_counts', mesh_counts)
    for i, ids in enumerate(list(mesh_counts.values())):
        flat_list = []
        for item in ids:
            flat_list.append(item)
        occurrences = Counter(flat_list[0])
        journal_name = list(mesh_counts.keys())[i]
        if journal_name in journal_dict:
            journal_dict[journal_name]['mesh_counts'] = dict(occurrences)
        else:
            print(journal_name, 'is not in the list')

    print('final_journal_dict', journal_dict)
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
