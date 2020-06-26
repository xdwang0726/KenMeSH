import dgl
import torch
import torch.nn as nn
from utils import tokenize


def get_edge_and_node_fatures(MeSH_id_pair_file, parent_children_file, vocab_size, field):
    """

    :param file:
    :return: edge:          a list of nodes pairs [(node1, node2), (node3, node4), ...] (39904 relations)
             node_count:    int, number of nodes in the graph
             node_features: a Tensor with size [num_of_nodes, embedding_dim]

    """
    # get descriptor and MeSH mapped
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()

    # count number of nodes and get edges
    node_count = len(mapping_id)
    values = list(mapping_id.values())
    edges = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = (values.index(item[0]), values.index(item[1]))
            edges.append(index_item)

    label_embedding = {}
    for key, value in mapping_id.items():
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=200)
        key = tokenize(key)
        key = [k.lower() for k in key]
        embedding.weight.data.copy_(field.vocab.vectors)
        key_seq = []
        for k in key:
            if field.vocab.stoi.get(k) is None:
                seq = 0
            else:
                seq = field.vocab.stoi.get(k)
            key_seq.append(seq)

        embedded_key = embedding(torch.LongTensor(key_seq))
        embedding = torch.mean(embedded_key, 1)
        label_embedding[values.index(value)] = embedding

    return edges, node_count, label_embedding


def build_MeSH_graph(edge_list, nodes, label_embedding):
    g = dgl.DGLGraph()
    # add nodes into the graph
    g.add_nodes(nodes)
    # add edges, directional graph
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # add node feature to the graph
    g.ndata['feat'] = torch.tensor(label_embedding.values())
    return g
