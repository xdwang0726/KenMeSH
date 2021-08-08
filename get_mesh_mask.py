import argparse
import heapq
import json

import ijson
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import gensim


from build_graph import tokenize


class DistributedCosineKnn:
    def __init__(self, k=3):
        self.k = k

    def fit(self, input_data, n_bucket=1):
        idxs = []
        dists = []
        buckets = np.array_split(input_data,n_bucket)
        for b in range(n_bucket):
            cosim = cosine_similarity(buckets[b], input_data)
            idx0 = [(heapq.nlargest((self.k+1), range(len(i)), i.take)) for i in cosim]
            idxs.extend(idx0)
            dists.extend([cosim[i][idx0[i]] for i in range(len(cosim))])
            return np.array(idxs), np.array(dists)


def idf_weighted_wordvec(doc, model):

    text = tokenize(doc)

    # get idf weighted word vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    words = vectorizer.get_feature_names()
    idf_weights = vectorizer.idf_
    idfs = dict(zip(words, idf_weights))

    # get pre-trained word embeddings
    weighted_word_vecs = torch.zeros(0)
    for word in text:
        word_vec = model.get_vector.reshape(1, 200)
        weighted_word_vec = torch.mul(word_vec, idfs[word]).view(1, 200)
        weighted_word_vecs = torch.cat((weighted_word_vecs, weighted_word_vec), dim=0)
    doc_vec = torch.sum(weighted_word_vecs, dim=1) / sum(idf_weights)

    return doc_vec


def get_knn_neighbors_mesh(train_path, vectors, k):
    print('hi')
    f = open(train_path, encoding="utf8")
    # objects = ijson.items(f, 'articles.item')
    objects = ijson.items(f, 'documents.item')

    doc_vecs = []
    pmid = []
    title = []
    all_text = []
    label = []
    label_id = []
    journals = []

    for i, obj in enumerate(tqdm(objects)):
        print('hi')
        ids = obj["pmid"]
        heading = obj['title'].strip()
        # heading = heading.translate(str.maketrans('', '', '[]'))
        # # print('heading', type(heading), heading)
        # data_point = {}
        # if len(heading) == 0:
        #     print('paper ', ids, ' does not have title!')
        # else:
        #     if heading == 'In process':
        #         continue
        #     else:
        #         abstract = obj["abstractText"].strip()
        #         abstract = abstract.translate(str.maketrans('', '', '[]'))
        #         text = title + ' ' + abstract
        #         doc_vec = idf_weighted_wordvec(text, vectors)
        #         doc_vecs.append(doc_vec)
        #         original_label = obj["meshMajor"]
        #         mesh_id = obj['meshId']
        #         journal = obj['journal']
        #         pmid.append(ids)
        #         title.append(heading)
        #         all_text.append(abstract)
        #         label.append(original_label)
        #         label_id.append(mesh_id)
        #         journals.append(journal)
        text = obj["abstract"].strip()
        doc_vec = idf_weighted_wordvec(text, vectors)
        doc_vecs.append(doc_vec)
        label = obj['meshId']
        pmid.append(ids)
        title.append(heading)
        text.append(text)
        label.append(label)
    print('Loading document done. ')

    # get k nearest neighors and return their mesh
    print('start to find the k nearest neibors for each article')
    neighbors = NearestNeighbors(n_neighbors=k).fit(doc_vecs)
    neighbors_meshs = []
    for i in range(len(doc_vecs)):
        _, idxes = neighbors.kneighbors(doc_vecs[i])
        neighbors_mesh = []
        for idx in idxes:
            mesh = label_id[idx]
            neighbors_mesh.append(mesh)
        neighbors_mesh = list(set([m for m in mesh for mesh in neighbors_mesh]))
        neighbors_meshs.append(neighbors_mesh)
    print('finding neighbors done')

    print('start collect data')
    dataset = []
    for i, doc in enumerate(title):
        data_point = {}
        data_point['title'] = doc
        data_point['abstractText'] = all_text[i]
        data_point['meshMajor'] = label[i]
        data_point['meshId'] = label_id[i]
        # data_point['journal'] = journals[i]
        data_point['neighbors'] = neighbors_mesh[i]
        # data_point['mesh_from_journal'] = journal_mesh
        dataset.append(data_point)

    pubmed = {'articles': dataset}

    return pubmed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allMesh')
    parser.add_argument('--vectors')
    parser.add_argument('--k')
    parser.add_argument('--save_path')
    args = parser.parse_args()

    model = gensim.models.KeyedVectors.load_word2vec_format(args.vectors, binary=True)
    pubmed = get_knn_neighbors_mesh(args.allMesh, model, args.k)

    with open(args.save_path, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)

if __name__ == "__main__":
    main()




