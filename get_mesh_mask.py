import argparse
import json
import os
import pickle
import string
import sys
import gc
from turtle import shape
import logging
import faiss
import ijson
import nltk
import torch
import torch.nn as nn
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, binarize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vectors
from tqdm import tqdm

from run_classifier_multigcn import weight_matrix
from utils import Preprocess

nltk.download('stopwords')
tokenizer = get_tokenizer('basic_english')


class Embedding(nn.Module):
    def __init__(self, weights):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)

    def forward(self, inputs, idf):
        embeddings = self.embedding(inputs)
        # packed_embedding = pack_padded_sequence(embeddings, input_length, batch_first=True, enforce_sorted=False)
        sum_idf = torch.sum(idf, dim=1).view(idf.shape[0], 1)
        weigthed_idf = sum_idf.view(sum_idf.shape[0], sum_idf.shape[1], 1)
        weighed_doc_embedding = torch.sum(torch.mul(weigthed_idf, embeddings), dim=1)
        return weighed_doc_embedding


def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    # check if the dataset if train or test
    if len(batch[0]) == 3:
        label = [entry[0] for entry in batch]

        # padding according to the maximum sequence length in batch
        text = [entry[1] for entry in batch]
        padded_text = pad_sequence(text, batch_first=True)
        # length = [len(seq) for seq in text]
        idf = [torch.Tensor(entry[2]) for entry in batch]
        padded_idf = pad_sequence(idf, batch_first=True)
        return padded_text, label, padded_idf
    else:
        text = [entry[0] for entry in batch]
        padded_text = pad_sequence(text, batch_first=True)
        # length = [len(seq) for seq in text]
        idf = [entry[1] for entry in batch]
        padded_idf = pad_sequence(idf, batch_first=True)
        return padded_text, padded_idf


def idf_weighted_wordvec(doc):

    tokens = tokenizer(doc)
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [k.lower() for k in stripped if k.isalpha()]
    # remove stopwords
    stop_words = stopwords.words('english')
    text = [w for w in tokens if not w in stop_words]
    # remove single character words
    text = [w for w in text if len(w) > 1]

    # print("Abstract Tokens: ", text)

    # get idf weighted word vectors
    vectorizer = TfidfVectorizer()
    if text:
        X = vectorizer.fit_transform(text)
        doc_vocab = vectorizer.vocabulary_
        idfs = vectorizer.idf_
    doc_idfs = []
    for t in text:
        idf = idfs[doc_vocab[t]]
        doc_idfs.append(idf)

    return doc_idfs


def get_idf_file(train_path):
    f = open(train_path, encoding="utf8")
    object = ijson.items(f, 'articles.item')

    idf_dataset = []
    for i, obj in enumerate(tqdm(object)):
        data_point = {}
        abstract = obj["abstractText"].strip()
        doc_idfs = []
        if abstract: 
            doc_idfs = idf_weighted_wordvec(abstract)

        # print("doc_idfs: ", doc_idfs)
        data_point['pmid'] = obj["pmid"]
        data_point['weighted_doc_vec'] = doc_idfs
        idf_dataset.append(data_point)
    f.close()
    
    return idf_dataset


def load_idf_file(idf_path):
    f = open(idf_path, encoding="utf8")
    object = ijson.items(f, 'articles.item')

    pmid = []
    weighted_doc_vec = []

    for i, obj in enumerate(tqdm(object)):
        ids = obj["pmid"]
        idf = obj['weighted_doc_vec']
        idf = [float(item) for item in idf]
        pmid.append(ids)
        weighted_doc_vec.append(idf)
    f.close()
    
    return pmid, weighted_doc_vec


def get_knn_neighbors_mesh(train_path, vectors, idf_path, k,  device, nprobe=5):

    # Contains the idfs of abstracts from all documents indexed with pmid_idf 
    pmid_idf, idfs = load_idf_file(idf_path)

    f = open(train_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    title = []
    all_text = []
    labels = []
    for i, obj in enumerate(tqdm(objects)):
        try:
            ids = obj["pmid"]
            heading = obj['title'].strip()
            heading = heading.translate(str.maketrans('', '', '[]'))
            abstract = obj["abstractText"].strip()
            clean_abstract = abstract.translate(str.maketrans('', '', '[]'))
            if len(heading) == 0 or heading == 'In process':
                print('paper ', ids, ' does not have title!')
                continue
            elif len(clean_abstract) == 0:
                print("Abstract: ", abstract)
                print("Clean Abstract: ", clean_abstract)

                print('paper ', ids, ' does not have abstract!')
                continue
            else:
                try:
                    # doc_vec, length = idf_weighted_wordvec(clean_abstract)
                    label = obj['mesh'] # dictionary
                    pmid.append(ids)
                    title.append(heading)
                    all_text.append(clean_abstract)
                    labels.append(label)
                except KeyError:
                    print('tfidf error', ids)
        except AttributeError:
            print(obj["pmid"].strip())
    f.close()

    print('Loading document done. ')

    dataset = Preprocess(all_text, idfs, labels)
    vocab = dataset.get_vocab()

    weights = weight_matrix(vocab, vectors)
    model = Embedding(weights)
    model.to(device)

    data = DataLoader(dataset, batch_size=1024, shuffle=False, collate_fn=generate_batch)
    pred = torch.zeros(0).double().cuda()

    for i, (text, label, idf) in enumerate(data):
        text, idf = text.to(device), idf.to(device)
        with torch.no_grad():
            output = model(text, idf)
            # print("Output: ", output)
            # print("Output type: ", type(output))

            pred = torch.cat((pred, output), dim=0)

    doc_vecs = pred.data.cpu().numpy()
    doc_vecs = doc_vecs.astype('float32')
    print('number of embedding articles', len(doc_vecs))

    # get k nearest neighors and return their mesh using faiss
    d = doc_vecs.shape[1]
    nlist = 60
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    assert not index.is_trained
    index.train(doc_vecs)
    assert index.is_trained

    index.add(doc_vecs)
    index.nprobe = nprobe
    neighbors_meshs = []
    for i in tqdm(range(doc_vecs.shape[0])):
        _, I = index.search(doc_vecs[i].reshape(1, 200), k)
        idxes = I[0]
        neighbors_mesh = []
        for idx in idxes:
            mesh = labels[idx]
            neighbors_mesh.append(mesh)
        neighbors_mesh = list(set([m for mesh in neighbors_mesh for m in mesh]))
        neighbors_meshs.append(neighbors_mesh)

    print('start collect data')
    dataset = []
    for i, id in enumerate(pmid):
        data_point = {}
        data_point['pmid'] = id
        mesh = ','.join(neighbors_meshs[i])
        data_point['neighbors'] = mesh
        dataset.append(data_point)

    pubmed = {'articles': dataset}

    return pubmed

def get_journal_mesh(journal_info, threshold, meshIDs):

    """
    journal_info structure
    "Journal name" : {
        "counts": (int) total mesh count,
        "mesh_counts": {
            "meshid": count,
            .
            .
        } 
    }

    journal_mesh structure
    {
        "journal name":[meshid1, meshid2,......],
        .
        .
    }
    
    """

    journal = pickle.load(open(journal_info, 'rb'))

    journal_mesh = {}
    for k, v in journal.items():
        # print("k, v: ", k,v)
        num = v['counts']
        # print("num: ", num)
        mesh = []
        new_mesh_index = []
        for i, (ids, counts) in enumerate(v['mesh_counts'].items()):
            if list(v['mesh_counts'].values())[i] / num >= threshold:
                mesh.append(ids)
        
        # Commented the below out because we are taking all the 
        # meshids realted to a journal name.
        # We will only inclue the mesh ids that has count/num >= threshold 
        
        # for m in mesh:
        #     try:
        #         m_id = meshIDs.index(m)
        #     except ValueError:
        #         continue
        #     new_mesh_index.append(m_id)
        if (k in journal_mesh.keys()):
            journal_mesh[k] = journal_mesh[k] + mesh
        else:
            journal_mesh[k] = mesh
    # print("Journal from journal: ", journal_mesh)

    return journal_mesh


def label2index(mesh_list, index_dic):

    idx_list = []
    for mesh in mesh_list:
        try:
            idx = index_dic[mesh]
            idx_list.append(idx)
        except:
            continue
    return idx_list


def read_neighbors(neighbors, index_dic = {}):
    f = open(neighbors, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    neighbors_mesh = []
    neigh_mask = []

    for i, obj in enumerate(tqdm(objects)):
        # data_point = {}
        ids = obj['pmid']
        mesh = obj['neighbors'].split(',')
        # mesh_idx = label2index(mesh, index_dic)
        # neigh_mask.append(mesh_idx)
        neighbors_mesh.append(mesh)
        pmid.append(ids)
    f.close()
    
    return pmid, neighbors_mesh


def mesh_mask(file, neigh_mask, journal_path):

    journal_mesh = pickle.load(open(journal_path, 'rb'))

    f = open(file, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    mesh_index = []
    for i, obj in enumerate(tqdm(objects)):
        journal = obj['journal']
        mesh_from_journal = journal_mesh[journal]
        mesh = list(set(mesh_from_journal + neigh_mask[i]))
        mesh_index.append(mesh)
    f.close()

    return mesh_index


def build_dataset(train_path, neighbors, journal_mesh, meshIDs, index_dic):

    # mesh_indexes = label2index(meshIDs, index_dic)
    # mesh_ids_str = [str(x) for x in meshIDs]

    # # TODO: Fix below for Mesh Indexes, Check MeshID and index positions
    # print('Total number of labels %d' % len(mesh_ids_str))
    # mlb = MultiLabelBinarizer(classes=mesh_ids_str)
    # # print("Mesh Id Confirmation: ", mesh_ids_str)
    # mlb.fit([mesh_ids_str])
    # # print("MLB Classes: ", mlb.classes_)

    print('Total number of labels %d' % len(meshIDs))
    mlb = MultiLabelBinarizer(classes=meshIDs)
    mlb.fit(meshIDs)

    pmid_neighbors, neighbors = read_neighbors(neighbors, index_dic)


    # print("pmid, neighbour: ", pmid_neighbors, neighbors_mesh)

    f = open(train_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')
    

    dataset = []
    print("pmid neighbors: ", pmid_neighbors, len(pmid_neighbors))
    print("neighbors_mesh: ", neighbors, len(neighbors))


    # with tqdm(unit_scale=0, unit='lines') as t:
    for i, obj in enumerate(tqdm(objects)):
        data_point = {}
        try:
            ids = obj["pmid"]
            heading = obj['title'].strip()
            heading = heading.translate(str.maketrans('', '', '[]'))
            abstract = obj["abstractText"].strip()
            clean_abstract = abstract.translate(str.maketrans('', '', '[]'))
            if len(heading) == 0 or heading == 'In process':
                print('paper ', ids, ' does not have title!')
                continue
            elif len(clean_abstract) == 0:
                print('paper ', ids, ' does not have abstract!')
                continue
            else:
                mesh_id = obj['mesh']
                journal = obj['journal']
                year = obj['year']
                mesh_from_journal = journal_mesh[journal]
                mesh_from_neighbors = []
                # print("negh test: ", ids, pmid_neighbors)
                if ids in pmid_neighbors:
                    _ = pmid_neighbors.index(ids)
                    # print("ID in pmid neigh: ", _)
                    # print("negh test 2: ", neighbors[_])
                    mesh_from_neighbors = neighbors[_]
                    # print("mesh_from_neighbors: ", mesh_from_neighbors)
                # mesh_from_journal_str = [str(x) for x in mesh_from_journal]
                # mesh_from_neighbors_str = [str(x) for x in mesh_from_neighbors]
                print("mesh_from_journal: ", len(mesh_from_journal))
                print("mesh_from_neighbors: ", len(mesh_from_neighbors))
                mesh = list(set(mesh_from_journal + mesh_from_neighbors))
                print("Mesh Line 403: ", type(mesh), len(mesh))
                print("Mesh Classes: ", len(mlb.classes_))
                mask = mlb.fit_transform([mesh])
                # mask = mask.astype(np.int_)
                mask = mask.tolist()
                if i == 0:
                    print("mesh: ", mesh)
                    print("mask: ", mask)
                print("Mask: ", np.bincount(mask[0]))
                print("MEsh Size: ", sys.getsizeof(mask))
                # print("Mesh content size: ", mask[0][0])
                print("Mesh content type: ", type(mask[0][0]))

                data_point['pmid'] = ids
                data_point['title'] = heading
                data_point['abstractText'] = clean_abstract
                data_point['meshID'] = mesh_id
                data_point['meshMask'] = mask
                data_point['year'] = year
                dataset.append(data_point)

        except AttributeError:
            print(f'An excaption occured for pmid: {obj["pmid"].strip()}', AttributeError.args())

    f.close()
    print("dataset Size: ", sys.getsizeof(dataset))

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allMesh')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--journal_info')
    parser.add_argument('--idfs_path')
    parser.add_argument('--neigh_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--save_path')
    parser.add_argument('--save_path_neigh')
    parser.add_argument('--save_path_idf')
    parser.add_argument('--journal')
    args = parser.parse_args()

    mapping_id = {}
    with open(args.meSH_pair_path, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            # mesh_id = value.strip('\n')
            mapping_id[key] = value.strip()
    meshIDs = list(mapping_id.values())
    # index_dic = {k: v for v, k in enumerate(meshIDs)} # MeshId: Index

    # print("mesh Ids: ", meshIDs)
    # print("Index Dic: ", index_dic)

    # 1. get idf vector
    # idfs = get_idf_file(args.allMesh)
    # idf_data = {'articles': idfs}
    # with open(args.save_path_idf, "w") as outfile:
    #     json.dump(idf_data, outfile)

    # 2. get masks using KNN
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # cache, name = os.path.split(args.word2vec_path)
    # vectors = Vectors(name=name, cache=cache)
    # # print(vectors.__getitem__("can"))
    # knn_mask = get_knn_neighbors_mesh(args.allMesh, vectors, args.idfs_path, args.k, device)
    # with open(args.save_path_neigh, "w") as outfile:
    #     json.dump(knn_mask, outfile)

    # 3. get masks from journal and merge the masks generated from neighbours
    journal_mesh = get_journal_mesh(args.journal_info, args.threshold, meshIDs)
    dataset = build_dataset(args.allMesh, args.neigh_path, journal_mesh, meshIDs, index_dic={})
    pubmed = {'articles': dataset}
    with open(args.save_path, "w") as outfile:
        json.dump(pubmed, outfile)
        print("file write complete on: ", outfile)

if __name__ == "__main__":
    main()



