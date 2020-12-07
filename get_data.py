import argparse
import json

import ijson
from tqdm import tqdm

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
    args = parser.parse_args()

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
    years = ['2012', '2013', '2014', '2015', '2016']

    for obj in tqdm(objects):
        data_point = {}
        if obj['year'] in years:
            try:
                ids = obj['pmid']
                title = obj['title']
                text = obj['abstractText'].strip()
                label = obj["meshMajor"]
                data_point['pmid'] = ids
                data_point['title'] = title
                data_point['abstractText'] = text
                data_point['meshMajor'] = label
                data_point['meshId'] = from_mesh2id(label, mapping_id)
                dataset.append(data_point)
            except AttributeError:
                print(obj["pmid"])

    print('Total number of articles: ', len(dataset))
    print('Finished Loading Data!')

    """ write to json file """

    pubmed = {'articles': dataset}

    with open(args.train_json, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)

    print('Finished writing to json file!')

if __name__ == "__main__":
    main()
