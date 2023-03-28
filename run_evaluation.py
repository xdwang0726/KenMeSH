import argparse
import os, sys
import pickle
import random
import ijson
import json
import logging

from dgl.data.utils import load_graphs
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors
import matplotlib.pyplot as plt

from eval_helper import precision_at_ks, example_based_evaluation, micro_macro_eval, zero_division
from model import *
from threshold import *
from losses import *
from utils import MeSH_indexing, pad_sequence


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn


def flatten(l):
    flat = [i for item in l for i in item]
    return flat

def load_meshid(MeSH_id_pair_file):
    print('load and prepare Mesh')
    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()
    meshIDs = list(mapping_id.values())
    print('Total number of labels %d' % len(meshIDs))
    return meshIDs

# def prepare_dataset(title_path, abstract_path, label_path, mask_path, MeSH_id_pair_file, word2vec_path, graph_file, is_multichannel=True): #graph_cooccurence_file
def prepare_dataset(dataset_path, meshIDs, word2vec_path, graph_file, is_multichannel):
    """ Load Dataset and Preprocessing """
    
    print('Start loading all data')

    mesh_mask = []
    all_title = []
    all_text = []
    label_id = []

    f = open(dataset_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    for i, obj in enumerate(tqdm(objects)):
        title = obj["title"]
        abstractText = obj["abstractText"]

        if len(title) < 1 or len(abstractText) < 1:
            continue      

        titles = title.split(" ")
        if len(titles) < 2:
            continue

        abstractTexts = abstractText.split(" ")
        if len(abstractTexts) < 2:
            continue
        
        mesh_mask.append(obj["meshMask"])
        all_title.append(obj["title"])
        all_text.append(obj["abstractText"])
        label_id.append(list(obj["meshID"].keys()))

    assert len(all_text) == len(all_title), 'title and abstract in the test set are not matching'
    print('Finish loading All data')
    f.close()
    print('number of training data %d' % len(all_title))

    print('load and prepare Mesh')

    assert len(all_text) == len(all_title), 'title and abstract in the training set are not matching'
    print('Finish loading training data')
    print('number of training data %d' % len(all_title))

    # load test data
    print('Start loading test data')
    print('number of test data %d' % len(all_title[-20000:]))
    
    mlb = MultiLabelBinarizer(classes=meshIDs)
    mlb.fit(meshIDs)


    # create Vector object map tokens to vectors
    print('load pre-trained BioWord2Vec')
    cache, name = os.path.split(word2vec_path)
    vectors = Vectors(name=name, cache=cache)

    # Preparing training and test datasets
    print('prepare training and test sets')
    dataset = MeSH_indexing(all_text, all_title, all_text[:-20], all_title[:-20], label_id[:-20], mesh_mask[:-20], all_text[:20],
                            all_title[:20], label_id[:20], mesh_mask[:20], is_test=True, is_multichannel=is_multichannel)
    
    # build vocab
    print('building vocab')
    vocab = dataset.get_vocab()

    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]
    print('graph', G.ndata['feat'].shape)

    print('prepare dataset and labels graph done!')
    return len(meshIDs), mlb, vocab, dataset, vectors, G#, neg_pos_ratio#, train_sampler, valid_sampler #, G_c


def weight_matrix(vocab, vectors, dim=200):
    itos = vocab.get_itos()
    stoi = vocab.get_stoi()
    weight_matrix = np.zeros([len(itos), dim])
    for i, token in enumerate(stoi):
        try:
            weight_matrix[i] = vectors.__getitem__(token)
        except KeyError:
            weight_matrix[i] = np.random.normal(scale=0.5, size=(dim,))
    return torch.from_numpy(weight_matrix)


def generate_batch(batch):
    """
    Output:
        text: the text entries in the data_batch are packed into a list and
            concatenated as a single tensor for the input of nn.EmbeddingBag.
        cls: a tensor saving the labels of individual text entries.
    """
    # check if the dataset is multi-channel or not
    if len(batch[0]) == 4:
        label = [entry[0] for entry in batch]
        mesh_mask = [entry[1] for entry in batch]

        # padding according to the maximum sequence length in batch
        abstract = [entry[2] for entry in batch]
        abstract_length = [len(seq) for seq in abstract]
        abstract = pad_sequence(abstract, ksz=3, batch_first=True)

        title = [entry[3] for entry in batch]
        title_length = []
        for i, seq in enumerate(title):
            if len(seq) == 0:
                length = len(seq) + 1
            else:
                length = len(seq)
            title_length.append(length)
        title = pad_sequence(title, ksz=3, batch_first=True)
        return label, mesh_mask, abstract, title, abstract_length, title_length

    else:
        label = [entry[0] for entry in batch]
        mesh_mask = [entry[1] for entry in batch]

        text = [entry[2] for entry in batch]
        text_length = [len(seq) for seq in text]
        text = pad_sequence(text, ksz=3, batch_first=True)

        return label, mesh_mask, text, text_length


def test(test_dataset, model, mlb, G, batch_sz, device, model_name="Full"):
    test_data = DataLoader(test_dataset, batch_size=batch_sz, collate_fn=generate_batch, shuffle=False, pin_memory=True)

    pred = []
    true_label = []

    print('Testing....')
    with torch.no_grad():
        model.eval()

        for label, mask, abstract, title, abstract_length, title_length in test_data:
            # print("----------------------------------------------------")
            # print(f'label: ${label}, \n mask: ${mask}, \n abstract: ${abstract}, \n title: ${title}, \n abstract_length: ${abstract_length}, \n title_length: ${title_length}', "\n-----------Test Data-----------")
            # for l in label: 
            #     if len(l) == 0:
            #         print("0 label: ", abstract)
            # for m in mask:
            #     print("Mask: ", np.bincount(m[0]))
            try:
                # mask = torch.from_numpy(mlb.fit_transform(mask)).type(torch.float)
                abstract = torch.from_numpy(np.asarray(abstract))
                title = torch.from_numpy(np.asarray(title))
                mask = torch.from_numpy(np.asarray(mask))
                abstract_length = torch.from_numpy(np.asarray(abstract_length))
                title_length = torch.from_numpy(np.asarray(title_length))
                mask, abstract, title, abstract_length, title_length = mask.to(device), abstract.to(device), title.to(device), abstract_length, title_length
                G, G.ndata['feat'] = G.to(device), G.ndata['feat'].to(device)
                # label = mlb.fit_transform(label)
                label = torch.from_numpy(mlb.fit_transform(label)).type(torch.float)
                label = torch.Tensor(np.asarray(label))
                with torch.no_grad():
                    output = model(abstract, title, mask, abstract_length, title_length, G, G.ndata['feat'])

                # print(results, "\n----------------Results-----------------")
                # print(label,"\n---------------label-------------" )

                # calculate precision at k
                results = output.data.cpu().numpy()
                pred.append(results)
                true_label.append(label)

                # print("Precision: ", precisions)
            except BaseException as exception:
                logging.warning(f"Exception Name: {type(exception).__name__}")
                logging.warning(f"Exception Desc: {exception}")
                # print("----------------------------------------------------")
                # print(f'label: ${label}, \n mask: ${mask}, \n abstract: ${abstract}, \n title: ${title}, \n abstract_length: ${abstract_length}, \n title_length: ${title_length}', "\n-----------Test Data-----------")
                # print("-----Except Test Data-------")
        
        print('###################TEST - DONE#########################')
        return pred, true_label

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path')

    parser.add_argument('----meSH_pair_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--graph')
    parser.add_argument('--model_name', default='Full', type=str)

    parser.add_argument('--pred_path', default='pred', type=str)
    parser.add_argument('--true_label_path', default='true_label', type=str)

    parser.add_argument('--model')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=200)
    parser.add_argument('--ksz', default=3)
    parser.add_argument('--hidden_gcn_size', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--atten_dropout', type=float, default=0.5)

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_sz', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=2)
    parser.add_argument('--lr_gamma', type=float, default=0.9)

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    n_gpu = torch.cuda.device_count()  # check if it is multiple gpu
    print('{} gpu is avaliable'.format(n_gpu))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    #Load Mesh IDs
    meshIDs = load_meshid(args.meSH_pair_path)


    # Get dataset and label graph & Load pre-trained embeddings
    num_nodes, mlb, vocab, test_dataset, vectors, G = prepare_dataset(args.dataset_path, meshIDs,
                                                                        args.word2vec_path, args.graph, is_multichannel=True)
    
    
    vocab_size = len(vocab)
    model = multichannel_dilatedCNN_with_MeSH_mask(vocab_size, args.dropout, args.ksz, num_nodes, G, device,
                                                    embedding_dim=200, rnn_num_layers=2, cornet_dim=1000,
                                                    n_cornet_blocks=2)
    model.embedding_layer.weight.data.copy_(weight_matrix(vocab, vectors)).to(device)

    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    # testing
    pred, true_label = test(test_dataset, model, mlb, G, args.batch_sz, device, args.model_name)

    np.save("pred_kenmesh", pred)
    torch.save(true_label, "true_label_kenmesh")
    

if __name__ == "__main__":
    main()
