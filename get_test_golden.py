import argparse
import json

import ijson
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allMesh')
    parser.add_argument('--test_set')
    parser.add_argument('--completed_test')

    args = parser.parse_args()

    # Load all articles in file
    f = open(args.allMesh, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    all_text = []
    label = []
    label_id = []

    print('Start loading training data')
    for obj in enumerate(tqdm(objects)):
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

    # Load test set ids
    f_t = open(args.test_set, encoding="utf8")
    test_objects = ijson.items(f_t, 'documents.item')

    test_pmid = []

    print('Start loading test data')
    for obj in tqdm(test_objects):
        try:
            ids = obj["pmid"]
            test_pmid.append(ids)
        except AttributeError:
            print(obj["pmid"].strip())

    # Create new test set with labels
    dataset = []
    for i, id in enumerate(tqdm(test_pmid)):
        data_point = {}
        if id in pmid:
            data_point['pmid'] = id
            data_point['abstract'] = all_text[i]
            data_point['meshMajor'] = label[i]
            data_point['meshId'] = label_id[i]
            dataset.append(data_point)
        else:
            print('Not in the list: ', obj["pmid"].strip())

    pubmed = {'documents': dataset}

    with open(args.completed_test, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)


if __name__ == "__main__":
    main()
