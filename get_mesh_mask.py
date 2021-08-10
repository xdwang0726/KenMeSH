import argparse
import json
import os
import string

import ijson
import nltk
import torch
import torch.nn as nn
from itertools import islice
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vectors
from tqdm import tqdm

from run_classifier_multigcn import weight_matrix
from utils import Preprocess
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

nltk.download('stopwords')
tokenizer = get_tokenizer('basic_english')


class Embedding(nn.Module):
    def __init__(self, weights):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)

    def forward(self, inputs, idf):
        embeddings = self.embedding(inputs)
        # packed_embedding = pack_padded_sequence(embeddings, input_length, batch_first=True, enforce_sorted=False)
        print('embedding', embeddings.shape)
        print('idf shape', idf.shape)
        weighed_doc_embedding = torch.mul(idf.view(idf.shape[0], idf.shape[1], 1), embeddings) / torch.sum(idf, dim=1)
        return weighed_doc_embedding


def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    # check if the dataset if train or test
    if len(batch[0]) == 2:
        label = [entry[0] for entry in batch]

        # padding according to the maximum sequence length in batch
        text = [entry[1] for entry in batch]
        padded_text = pad_sequence(text, batch_first=True)
        length = [len(seq) for seq in text]
        idf = [entry[2] for entry in batch]
        padded_idf = pad_sequence(idf, batch_first=True)
        return padded_text, length, label, padded_idf
    else:
        text = [entry for entry in batch]
        padded_text = pad_sequence(text, batch_first=True)
        length = [len(seq) for seq in text]
        idf = [entry[2] for entry in batch]
        padded_idf = pad_sequence(idf, batch_first=True)
        return padded_text, length, padded_idf


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

    # get idf weighted word vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)
    doc_vocab = vectorizer.vocabulary_
    idfs = vectorizer.idf_
    doc_idfs = []
    for t in text:
        idf = idfs[doc_vocab[t]]
        doc_idfs.append(idf)


    # words = vectorizer.get_feature_names()
    # idf_weights = vectorizer.idf_
    # idfs = dict(zip(words, idf_weights))
    # doc_idfs = idf(text, vectorizer)

    # get pre-trained word embeddings
    # weighted_word_vecs = torch.zeros(0)
    # for word in text:
    #     if word in idfs.keys():
    #         try:
    #             word_vec = model.get_vector(word).reshape(1, 200)
    #             weighted_word_vec = torch.from_numpy(np.multiply(word_vec, idfs[word]))
    #             weighted_word_vecs = torch.cat((weighted_word_vecs, weighted_word_vec), dim=0)
    #         except KeyError:
    #             continue
    # doc_vec = torch.sum(weighted_word_vecs, dim=1) / sum(idf_weights)
    return doc_idfs


def load_idf_file(idf_path):
    f = open(idf_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    pmid = []
    weighted_doc_vec = []
    lengths = []

    for i, obj in enumerate(tqdm(objects)):
        ids = obj["pmid"]
        idf = obj['weighted_doc_vec']
        # idf_len = obj['length']
        pmid.append(ids)
        weighted_doc_vec.append(idf)
        # lengths.append(idf_len)
    return pmid, weighted_doc_vec


def get_knn_neighbors_mesh(train_path, vectors, idf_path, device):
    f = open(train_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')
    # objects = ijson.items(f, 'documents.item')

    pmid = []
    title = []
    all_text = []
    label = []
    label_id = []
    journals = []

    # for i, obj in enumerate(tqdm(objects)):
    #     try:
    #         ids = obj["pmid"]
    #         heading = obj['title'].strip()
    #         heading = heading.translate(str.maketrans('', '', '[]'))
    #         abstract = obj["abstractText"].strip()
    #         clean_abstract = abstract.translate(str.maketrans('', '', '[]'))
    #         if len(heading) == 0 or heading == 'In process':
    #             print('paper ', ids, ' does not have title!')
    #             continue
    #         elif len(clean_abstract) == 0:
    #             print('paper ', ids, ' does not have abstract!')
    #             continue
    #         else:
    #             try:
    #                 original_label = obj["meshMajor"]
    #                 mesh_id = obj['meshId']
    #                 journal = obj['journal']
    #                 pmid.append(ids)
    #                 all_text.append(abstract)
    #                 label.append(original_label)
    #                 label_id.appen(mesh_id)
    #                 journals.append(journal)
    #             except KeyError:
    #                 print('tfidf error', ids)
    #     except AttributeError:
    #         print(obj["pmid"].strip())

    pmid_idf, idfs = load_idf_file(idf_path)

    for i, obj in enumerate(tqdm(objects)):
        ids = obj["pmid"]
        if ids == pmid_idf[i]:
            heading = obj['title'].strip()
            text = obj["abstract"].strip()
            l = obj['meshId']
            pmid.append(ids)
            title.append(heading)
            all_text.append(text)
            label.append(l)
        else:
            print(ids, 'Not match')
    print('Loading document done. ')

    # doc_idfs = idf_weighted_wordvec(all_text)

    dataset = Preprocess(all_text, label, idfs)
    vocab = dataset.get_vocab()

    weights = weight_matrix(vocab, vectors)
    model = Embedding(weights)
    model.to(device)

    data = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=generate_batch)
    pred = torch.zeros(0).cuda()
    # lengths = []
    for i, (text, length, label, idf) in enumerate(data):
        text, idf = text.to(device), idf.to(device)
        with torch.no_grad():
            output = model(text, idf)
            pred = torch.cat((pred, output), dim=0)
        # pred = output.data.cpu().numpy()
        # output_iter = iter(pred)
        # vecs = [list(islice(output_iter, elem)) for elem in length]
        # calculated weigthed idf document vectors

        # doc_vec.extend(vecs)
        # lengths.extend(length)
    doc_vec = pred.data.cpu().numpy()

    print('number of embedding articles', len(doc_vec), type(doc_vec))
    # print('length', type(lengths))
    # # get k nearest neighors and return their mesh
    # print('start to find the k nearest neibors for each article')
    # neighbors = NearestNeighbors(n_neighbors=k).fit(doc_vecs)
    # neighbors_meshs = []
    # for i in range(len(doc_vecs)):
    #     _, idxes = neighbors.kneighbors(doc_vecs[i])
    #     neighbors_mesh = []
    #     for idx in idxes:
    #         mesh = label_id[idx]
    #         neighbors_mesh.append(mesh)
    #     neighbors_mesh = list(set([m for m in mesh for mesh in neighbors_mesh]))
    #     neighbors_meshs.append(neighbors_mesh)
    # print('finding neighbors done')

    print('start collect data')
    dataset = []
    for i, id in enumerate(pmid):
        data_point = {}
        data_point['pmid'] = id
        #print('doc_vec', type(doc_vec[i]), doc_vec[i])
        data_point['doc_vec'] = doc_vec[i]
        #print('doc_vec_len', type(lengths[i]), lengths[i])
        # data_point['doc_vec_len'] = lengths[i]
        # data_point['title'] = title[i]
        # data_point['abstractText'] = all_text[i]
        # data_point['meshMajor'] = label[i]
        # data_point['meshId'] = label_id[i]
        # data_point['journal'] = journals[i]
        # data_point['neighbors'] = neighbors_mesh[i]
        # data_point['mesh_from_journal'] = journal_mesh
        dataset.append(data_point)

    pubmed = {'articles': dataset}

    return pubmed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allMesh')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--idfs_path')
    parser.add_argument('--save_path')
    args = parser.parse_args()

    # model = gensim.models.KeyedVectors.load_word2vec_format(args.vectors, binary=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache, name = os.path.split(args.word2vec_path)
    vectors = Vectors(name=name, cache=cache)
    pubmed = get_knn_neighbors_mesh(args.allMesh, vectors, args.idfs_path, device)
    print('pubmed type', type(pubmed))
    with open(args.save_path, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)


if __name__ == "__main__":
    main()



