import dgl
import ijson
import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm


def tokenize(text):
    tokens = []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        tokens.append(token.text)
    return tokens


def get_edge_and_node_fatures(MeSH_id_pair_file, parent_children_file, vectors):
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
    values = list(mapping_id.values())
    edges = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = (values.index(item[0]), values.index(item[1]))
            edges.append(index_item)

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key = tokenize(key)
        key = [k.lower() for k in key]
        key_embedding = torch.zeros(0)
        for k in key:
            embedding = vectors.__getitem__(k).reshape(1, 200)
            # if vectors.stoi.get(k) is None:
            #     embedding = torch.zeros([1, 200], dtype=torch.float32)
            # else:
            #     embedding = vectors.vectors[vectors.stoi.get(k)].reshape(1, 200)
            key_embedding = torch.cat((key_embedding, embedding), dim=0)
        key_embedding = torch.mean(input=key_embedding, dim=0, keepdim=True)
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

    # for key, value in mapping_id.items():
    #     embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=200)
    #     key = tokenize(key)
    #     key = [k.lower() for k in key]
    #     embedding.weight.data.copy_(field.vocab.vectors)
    #     key_seq = []
    #     for k in key:
    #         if field.vocab.stoi.get(k) is None:
    #             seq = 0
    #         else:
    #             seq = field.vocab.stoi.get(k)
    #         key_seq.append(seq)
    #
    #     embedded_key = embedding(torch.LongTensor(key_seq))  # size: (seq_len, embedding_sz)
    #     embedding = torch.mean(input=embedded_key, dim=0, keepdim=True)
    #     label_embedding = torch.cat((label_embedding, embedding), dim=0)

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
                if target != label:
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
    label_counts = {}
    for doc in label_id:
        for label in doc:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    labels = list(label_counts.keys())
    for key in labels:
        if label_counts.get(key) == None:
            label_counts[key] = 0
    num = list(label_counts.values())

    # get co-occurrence edges
    edge_frame = cooccurrence_matrix.div(num, axis='index')
    edge_frame = (edge_frame >= threshold) * 1  # replacing each element larger than threshold by 1 else 0
    edge_index = np.argwhere(edge_frame.values == 1)
    train_mesh_list = list(cooccurrence_matrix)
    edge_cooccurrence = []
    for i in edge_index:
        item = (train_mesh_list[i[0]], train_mesh_list[i[1]])
        idex_item = (values.index(item[0]), values.index(item[1]))
        edge_cooccurrence.append(idex_item)

    edges_parent_children = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = (values.index(item[0]), values.index(item[1]))
            edges_parent_children.append(index_item)

    edges = edge_cooccurrence + edges_parent_children
    edge_type = [0] * len(edge_cooccurrence) + [1] * len(edges_parent_children)
    edge_type = torch.from_numpy(np.array(edge_type))
    edge_norm = [1] * len(edges)
    edge_norm = torch.from_numpy(np.array(edge_norm)).float()

    print('get label embeddings')
    label_embedding = torch.zeros(0)
    for key, value in tqdm(mapping_id.items()):
        key = tokenize(key)
        key = [k.lower() for k in key]
        key_embedding = torch.zeros(0)
        for k in key:
            embedding = vectors.__getitem__(k).reshape(1, 200)
            key_embedding = torch.cat((key_embedding, embedding), dim=0)
        key_embedding = torch.mean(input=key_embedding, dim=0, keepdim=True)
        label_embedding = torch.cat((label_embedding, key_embedding), dim=0)

    return edges, edge_type, edge_norm, node_count, label_embedding


def build_MeSH_RGCNgraph(edge_list, edge_type, edge_norm, nodes, label_embedding):
    print('start building the graph')
    g = dgl.DGLGraph()
    # add nodes into the graph
    print('add nodes into the graph')
    g.add_nodes(nodes)
    # add edges, directional graph
    print('add edges into the graph')
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # add relation type to the graph
    print('add relation type into the graph')
    g.edata.update({'rel_type': edge_type})
    # add edge norm to the graph
    g.edata.update({'norm': edge_norm})
    # add node features into the graph
    print('add node features into the graph')
    g.ndata['feat'] = label_embedding
    return g
