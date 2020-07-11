import dgl
import torch
from tqdm import tqdm

from utils import tokenize


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
            # embedding = vectors.__getitem__(k).reshape(1, 200)
            if vectors.stoi.get(k) is None:
                embedding = torch.zeros([1, 200], dtype=torch.float32)
            else:
                embedding = vectors.vectors[vectors.stoi.get(k)].reshape(1, 200)
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
