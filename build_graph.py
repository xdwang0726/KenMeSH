import collections
import dgl
from model import Embeddings_OOV


def get_edge_and_node_fatures(MeSH_id_pair_file, parent_children_file):
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
            mapping_id[key] = value

    # count number of nodes and get edges
    node_count = len(mapping_id)
    values = list(mapping_id.values())
    edges = []
    with open(parent_children_file, 'r') as f:
        for line in f:
            item = tuple(line.strip().split(" "))
            index_item = tuple(values.index(item[0]), values.index(item[1]))
            edges.append(index_item)

    # get features for each node
    embeddings = []
    for descriptor in mapping_id.keys():
        # average word embeddings
        embedding = Embeddings_OOV(descriptor)
        embeddings.append(embedding)

    return edges, node_count, embeddings


def build_MeSH_graph(MeSH_id_pair_file, parent_children_file):
    edge_list, nodes, node_features = get_edge_and_node_fatures(MeSH_id_pair_file, parent_children_file)
    g = dgl.DGLGraph()
    # add nodes into the graph
    g.add_nodes(nodes)
    # add edges, directional graph
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # add node feature to the graph
    g.ndata['feat'] = node_features
    return g
