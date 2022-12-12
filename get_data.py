import argparse
import json

import ijson
from tqdm import tqdm
import string

"""
Extract the articles from 2012-2016 from the BioASQ dataset for Taska

"""


def from_mesh2id(labels_list, mapping_id):
    mesh_id = []
    for mesh in labels_list:
        index = mapping_id.get(mesh.strip())
        if index is None:
            print(index)
            pass
        else:
            mesh_id.append(index.strip())
    return mesh_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allMesh')
    parser.add_argument('--MeshID')
    parser.add_argument('--train_json')
    parser.add_argument('--years', type=list, default=['2012', '2013', '2014', '2015', '2016'])
    args = parser.parse_args()

    table = str.maketrans('', '', string.punctuation)

    """ mapping mesh terms to meshIDs """
    mapping_id = {}
    with open(args.MeshID) as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value

    """ get text(abstract and title) and MeSH terms to each document """
    f = open(args.allMesh, encoding="utf8", errors='ignore')

    objects = ijson.items(f, 'articles.item')

    dataset = []

    for i, obj in enumerate(tqdm(objects)):
        data_point = {}
        if obj['year'] in args.years:
            try:
                ids = obj['pmid']
                title = obj['title']
                clean_title = title.translate(str.maketrans('', '', '[]'))
                text = obj["abstractText"].strip()
                clean_text = text.translate(table)
                if len(clean_title) == 0 or clean_title == 'In process':
                    print('paper ', ids, ' does not have title!')
                    continue
                elif len(clean_text) == 0:
                    print('paper ', ids, ' does not have abstract!')
                    continue
                else:
                    text = obj['abstractText'].strip()
                    label = obj["meshMajor"]
                    journal = obj['journal']
                    data_point['pmid'] = ids
                    data_point['title'] = title
                    data_point['abstractText'] = text
                    data_point['meshMajor'] = label
                    data_point['meshId'] = from_mesh2id(label, mapping_id)
                    data_point['journal'] = journal
                    dataset.append(data_point)
            except AttributeError:
                print(obj["pmid"])
        else:
            continue

    print('Total number of articles: ', len(dataset))
    print('Finished Loading Data!')

    """ write to json file """

    pubmed = {'articles': dataset}

    with open(args.train_json, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)

    print('Finished writing to json file!')


if __name__ == "__main__":
    main()

