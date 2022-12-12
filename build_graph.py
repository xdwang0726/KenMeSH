import argparse
import os
import timeit
import dgl
import json
import ijson
import numpy as np
import pandas as pd
import torch
from math import ceil
from dgl.data.utils import save_graphs
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vectors
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import BertTokenizer

from mesh_description_embedding import get_bert_embedding

tokenizer = get_tokenizer('basic_english')


def get_edge_and_node_fatures(MeSH_id_pair_file, MeSH_description, parent_children_file, vectors, bert_model):
    """

    :param file:
    :return: edge:          a list of nodes pairs [(node1, node2), (node3, node4), ...] (39904 relations)
             node_count:    int, number of nodes in the graph
             node_features: a Tensor with size [num_of_nodes, embedding_dim]

    """
    print('load MeSH id and names')
    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

   
    with open(MeSH_description) as json_file:
        mesh_description = json.load(json_file)

    # count number of nodes and get edges
    print('count number of nodes and get edges of the graph')
    node_count = len(mapping_id)
    print('number of nodes: ', node_count)
    values = list(mapping_id.values())
    edges = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = (values.index(item[0]), values.index(item[1]))
            edges.append(index_item)
    print('number of edges: ', len(edges))

    label_embedding = torch.zeros(0)
    for k, value in tqdm(mapping_id.items()):
        key = tokenizer(k)
        key = [k.lower() for k in key]
        embedding = []
        for k in key:
            embedding.append(vectors.__getitem__(k))

        # feed embedding to LSTM
        # hidden layers are the embedding of the whole MeSH Terms
        # if you do batch tarining don't forget to pack or unpack 

        key_embedding = torch.mean(torch.stack(embedding), dim=0, keepdim=True)
        
        if value in mesh_description:
            description_embedding = get_bert_embedding(mesh_description[value], bert_model)
        else:
            description_embedding = get_bert_embedding(k, bert_model)

        description_embedding = description_embedding.view(1, 200)
        key_embedding = torch.cat((key_embedding, description_embedding), dim=1)
        
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

        print(f'key embedding: {key_embedding.size()} \n label embed: {label_embedding.size()} \n description: {description_embedding.size()}')
    
    print(f'key_embedding_dimension: {key_embedding.size()}')
    print(f'total label_embedding_dimension: {label_embedding.size()}')

    return edges, node_count, label_embedding


def get_edge_and_bert_node_fatures(MeSH_id_pair_file, parent_children_file, tokenizer, model):
    """

    :param file:
    :return: edge:          a list of nodes pairs [(node1, node2), (node3, node4), ...] (39904 relations)
             node_count:    int, number of nodes in the graph
             node_features: a Tensor with size [num_of_nodes, embedding_dim]

    """
    print('load MeSH id and names')
    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    # count number of nodes and get edges
    print('count number of nodes and get edges of the graph')
    node_count = len(mapping_id)
    print('number of nodes: ', node_count)
    values = list(mapping_id.values())
    edges = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = (values.index(item[0]), values.index(item[1]))
            edges.append(index_item)
    print('number of edges: ', len(edges))

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key_encoding = tokenizer.encode(key, add_special_tokens=True)
        with torch.no_grad():
            _, key_embedding = model(torch.tensor([key_encoding]))
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)
    return edges, node_count, label_embedding


def build_MeSH_graph(edge_list, nodes, label_embedding):
    print('start building the graph')
    g = dgl.DGLGraph()
    # add nodes into the graph
    print('add nodes into the graph')
    g.add_nodes(nodes)
    # add edges, directional graph
    print('add edges into the graph')
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # add node features into the graph
    print('add node features into the graph')
    g.ndata['feat'] = label_embedding
    return g


def multitype_GCN_get_node_and_edges(train_data_path, MeSH_id_pair_file, parent_children_file, threshold, vectors):
    # get MeSH in each example
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []

    print('Start loading training data')
    for i, obj in enumerate(tqdm(objects)):
        try:
            mesh_id = obj['meshId']
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    # count number of nodes and get parent and children edges
    print('count number of nodes and get edges of the graph')
    node_count = len(mapping_id)
    values = list(mapping_id.values())

    # count the co-occurrence between MeSH
    cooccurrence_counts = {}
    for doc in label_id:
        for label in doc:
            labelDict = dict()
            if label in cooccurrence_counts:
                labelDict = cooccurrence_counts[label]
            for target in doc:
                # if target != label:
                if target in labelDict:
                    labelDict[target] = labelDict[target] + 1
                else:
                    labelDict[target] = 1

            cooccurrence_counts[label] = labelDict

    cooccurrence_matrix = pd.DataFrame(cooccurrence_counts)
    label_union = cooccurrence_matrix.index.union(cooccurrence_matrix.columns)
    cooccurrence_matrix = cooccurrence_matrix.reindex(index=label_union, columns=label_union)
    cooccurrence_matrix = cooccurrence_matrix.fillna(0)  # replace all nan value to 0

    # calculate the occurrence times of each label in the training set
    num = np.diag(cooccurrence_matrix).tolist()

    # get co-occurrence edges
    edge_frame = cooccurrence_matrix.div(num, axis='index')
    edge_frame = (edge_frame >= threshold) * 1  # replacing each element larger than threshold by 1 else 0
    # remove the lower half of the matrix
    edge_index = np.argwhere(edge_frame.values == 1)
    train_mesh_list = list(cooccurrence_matrix)
    edge_cooccurrence = []
    for i in edge_index:
        if train_mesh_list[i[0]] != train_mesh_list[i[1]]:
            item = (train_mesh_list[i[0]], train_mesh_list[i[1]])
            idex_item = (values.index(item[0]), values.index(item[1]))
            edge_cooccurrence.append(idex_item)

    edges_parent_children = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = (values.index(item[0]), values.index(item[1]))
            edges_parent_children.append(index_item)

    edges = set(edge_cooccurrence + edges_parent_children)
    print('num_edge', len(edge_cooccurrence), len(edges_parent_children))

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key = tokenizer(key)
        key = [k.lower() for k in key]
        key_embedding = torch.zeros(0)
        for k in key:
            embedding = vectors.__getitem__(k).reshape(1, 200)
            key_embedding = torch.cat((key_embedding, embedding), dim=0)
        key_embedding = torch.mean(input=key_embedding, dim=0, keepdim=True)
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

    return edges, node_count, label_embedding


def build_MeSH_GCNgraph_multitype(edge_list, nodes, label_embedding):
    print('start building the graph')
    g = dgl.DGLGraph()
    # add nodes into the graph
    print('add nodes into the graph')
    g.add_nodes(nodes)
    # add edges, directional graph
    print('add edges into the graph')
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # add node features into the graph
    print('add node features into the graph')
    g.ndata['feat'] = label_embedding
    return g


def cooccurence_node_edge(train_data_path, MeSH_id_pair_file, threshold, vectors):
    # get MeSH in each example
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []

    print('Start loading training data')
    for i, obj in enumerate(tqdm(objects)):
        try:
            mesh_id = obj['meshId']
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    # count number of nodes and get parent and children edges
    print('count number of nodes and get edges of the graph')
    node_count = len(mapping_id)
    values = list(mapping_id.values())

    # count the co-occurrence between MeSH
    cooccurrence_counts = {}
    for doc in label_id:
        for label in doc:
            labelDict = dict()
            if label in cooccurrence_counts:
                labelDict = cooccurrence_counts[label]
            for target in doc:
                # if target != label:
                if target in labelDict:
                    labelDict[target] = labelDict[target] + 1
                else:
                    labelDict[target] = 1

            cooccurrence_counts[label] = labelDict

    cooccurrence_matrix = pd.DataFrame(cooccurrence_counts)
    label_union = cooccurrence_matrix.index.union(cooccurrence_matrix.columns)
    cooccurrence_matrix = cooccurrence_matrix.reindex(index=label_union, columns=label_union)
    cooccurrence_matrix = cooccurrence_matrix.fillna(0)  # replace all nan value to 0

    # calculate the frequency each label in the training set
    num = np.diag(cooccurrence_matrix).tolist()

    # get co-occurrence edges
    edge_frame = cooccurrence_matrix.div(num, axis='index')
    edge_frame = (edge_frame >= threshold) * 1  # replacing each element larger than threshold by 1 else 0
    # remove the lower half of the matrix
    # edge_frame[:] = np.where(np.arange(len(edge_frame))[:, None] >= np.arange(len(edge_frame)), np.nan, edge_frame)
    edge_index = np.argwhere(edge_frame.values == 1)
    train_mesh_list = list(cooccurrence_matrix)
    edge_cooccurrence = []
    for i in edge_index:
        if train_mesh_list[i[0]] != train_mesh_list[i[1]]:
            item = (train_mesh_list[i[0]], train_mesh_list[i[1]])
            idex_item = (values.index(item[0]), values.index(item[1]))
            edge_cooccurrence.append(idex_item)

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key = tokenizer(key)
        key = [k.lower() for k in key]
        key_embedding = torch.zeros(0)
        for k in key:
            embedding = vectors.__getitem__(k).reshape(1, 200)
            key_embedding = torch.cat((key_embedding, embedding), dim=0)
        key_embedding = torch.mean(input=key_embedding, dim=0, keepdim=True)
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

    return edge_cooccurrence, node_count, label_embedding


def RGCN_get_node_and_edges(train_data_path, MeSH_id_pair_file, parent_children_file, threshold, vectors):
    # get MeSH in each example
    f = open(train_data_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    label_id = []

    print('Start loading training data')
    for i, obj in enumerate(tqdm(objects)):
        try:
            mesh_id = obj['meshId']
            label_id.append(mesh_id)
        except AttributeError:
            print(obj["pmid"].strip())

    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    # count number of nodes and get parent and children edges
    print('count number of nodes and get edges of the graph', len(mapping_id))
    values = list(mapping_id.values())

    # count the co-occurrence between MeSH
    cooccurrence_counts = {}
    for doc in label_id:
        for label in doc:
            labelDict = dict()
            if label in cooccurrence_counts:
                labelDict = cooccurrence_counts[label]
            for target in doc:
                # if target != label:
                if target in labelDict:
                    labelDict[target] = labelDict[target] + 1
                else:
                    labelDict[target] = 1

            cooccurrence_counts[label] = labelDict

    cooccurrence_matrix = pd.DataFrame(cooccurrence_counts)
    label_union = cooccurrence_matrix.index.union(cooccurrence_matrix.columns)
    cooccurrence_matrix = cooccurrence_matrix.reindex(index=label_union, columns=label_union)
    cooccurrence_matrix = cooccurrence_matrix.fillna(0)  # replace all nan value to 0

    # calculate the occurrence times of each label in the training set
    num = np.diag(cooccurrence_matrix).tolist()
    # num_label = cooccurrence_matrix.sum(axis=1)
    # num = num_label.tolist()

    # get co-occurrence edges
    edge_frame = cooccurrence_matrix.div(num, axis='index')
    edge_frame = (edge_frame >= threshold) * 1  # replacing each element larger than threshold by 1 else 0
    # remove the lower half of the matrix
    # edge_frame[:] = np.where(np.arange(len(edge_frame))[:, None] >= np.arange(len(edge_frame)), np.nan, edge_frame)
    edge_index = np.argwhere(edge_frame.values == 1)
    train_mesh_list = list(cooccurrence_matrix)
    edge_cooccurrence = list()
    for i in edge_index:
        if train_mesh_list[i[0]] != train_mesh_list[i[1]]:
            item = (train_mesh_list[i[0]], train_mesh_list[i[1]])
            index_item = torch.tensor([values.index(item[0]), values.index(item[1])])
            edge_cooccurrence.append(index_item)

    cooccurrence_dic = {('mesh', 'cooccurrence', 'mesh'): edge_cooccurrence}

    edges_parent_children = list()
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = torch.tensor([values.index(item[0]), values.index(item[1])])
            edges_parent_children.append(index_item)

    parent_children_dic = {('mesh', 'hierarchy', 'mesh'): edges_parent_children}

    # updated data_dic
    parent_children_dic.update(cooccurrence_dic)

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key = tokenizer(key)
        key = [k.lower() for k in key]
        key_embedding = torch.zeros(0)
        for k in key:
            embedding = vectors.__getitem__(k).reshape(1, 200)
            key_embedding = torch.cat((key_embedding, embedding), dim=0)
        key_embedding = torch.mean(input=key_embedding, dim=0, keepdim=True)
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

    return parent_children_dic, label_embedding


def build_MeSH_RGCNgraph(edge_dic, label_embedding):
    print('start building the graph')
    g = dgl.heterograph(edge_dic)
    g.ndata['feat'] = label_embedding
    return g


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--meSH_description_path')
    parser.add_argument('--mesh_parent_children_path')
    parser.add_argument('--bert', default='bert-base-uncased')
    parser.add_argument('--output')
    parser.add_argument('--graph_type', type=str)

    args = parser.parse_args()

    if args.graph_type == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(args.bert)
        model = BertModel.from_pretrained(args.bert)
        model.eval()
        # BERT GCN
        edge, node_count, label_embedding = get_edge_and_bert_node_fatures(args.meSH_pair_path, args.mesh_parent_children_path, tokenizer, model)
        G = build_MeSH_graph(edge, node_count, label_embedding)
    else:
        print('Load pre-trained vectors')
        cache, name = os.path.split(args.word2vec_path)
        print(f"cache: {cache}, name: {name}")
        vectors = Vectors(name=name, cache=cache)
        if args.graph_type == 'GCN':
            edges, node_count, label_embedding = get_edge_and_node_fatures(args.meSH_pair_path, args.meSH_description_path, args.mesh_parent_children_path,
                                                                           vectors, args.bert)
            G = build_MeSH_graph(edges, node_count, label_embedding)
        elif args.graph_type == 'GCN_cooccurence':
            edge, node_count, label_embedding = cooccurence_node_edge(args.train, args.meSH_pair_path, args.threshold, vectors)
            G = build_MeSH_graph(edge, node_count, label_embedding)
        elif args.graph_type == 'GCN_multitype':
            edge, node_count, label_embedding =multitype_GCN_get_node_and_edges(args.train, args.meSH_pair_path,
                                                                                args.mesh_parent_children_path, args.threshold,
                                                                                vectors)
            G = build_MeSH_graph(edge, node_count, label_embedding)
        elif args.graph_type == 'RGCN':
            edge_dic, label_embedding = RGCN_get_node_and_edges(args.train, args.meSH_pair_path,
                                                                args.mesh_parent_children_path, args.threshold, vectors)
            G = build_MeSH_RGCNgraph(edge_dic, label_embedding)

    print("In build_graph Save graph: ")
    save_graphs(args.output, G)


if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print('Time: ', stop - start)


