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
    dataset = MeSH_indexing(all_text, all_title, all_text[:-20000], all_title[:-20000], label_id[:-20000], mesh_mask[:-20000], all_text[:20000],
                            all_title[:20000], label_id[:20000], mesh_mask[:20000], is_test=True, is_multichannel=is_multichannel)
    
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
    weight_matrix = np.zeros([len(vocab.itos), dim])
    for i, token in enumerate(vocab.stoi):
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
    top_k_precisions = []
    sum_pred = 0.
    sum_target = 0.
    sum_product = 0.
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.

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
                test_labelsIndex = getLabelIndex(label)
                precisions, recall = precision_at_ks(results, test_labelsIndex, ks=[1, 3, 5])
                top_k_precisions.append(precisions)
                # calculate example-based evaluation
                sums = example_based_evaluation(results, label, 0.5, 20000)
                sum_pred += sums[0]
                sum_target += sums[1]
                sum_product += sums[2]
                # calculate label-based evaluation
                confusion = micro_macro_eval(results, label, threshold=0.5)
                tp += confusion[0]
                tn += confusion[1]
                fp += confusion[2]
                fn += confusion[3]

                # print("Precision: ", precisions)
            except BaseException as exception:
                logging.warning(f"Exception Name: {type(exception).__name__}")
                logging.warning(f"Exception Desc: {exception}")
                # print("----------------------------------------------------")
                # print(f'label: ${label}, \n mask: ${mask}, \n abstract: ${abstract}, \n title: ${title}, \n abstract_length: ${abstract_length}, \n title_length: ${title_length}', "\n-----------Test Data-----------")
                # print("-----Except Test Data-------")
                
        # Evaluations
        print('Calculate Precision at K...')
        p_at_1 = np.mean(flatten(p_at_k[0] for p_at_k in top_k_precisions))
        # print(p_at_1)
        p_at_3 = np.mean(flatten(p_at_k[1] for p_at_k in top_k_precisions))
        p_at_5 = np.mean(flatten(p_at_k[2] for p_at_k in top_k_precisions))
        for k, p in zip([1, 3, 5], [p_at_1, p_at_3, p_at_5]):
            print('p@{}: {:.5f}'.format(k, p))

        print('Calculate Example-based Evaluation')
        ebp = sum_product / sum_pred
        ebr = sum_product / sum_target
        ebf = (2 * sum_product) / (sum_pred + sum_target)
        for n, m in zip(['EBP', 'EBR', 'EBF'], [ebp, ebr, ebf]):
            print('{}: {:.5f}'.format(n, m))

        print('Calculate Label-based Evaluation')
        mip = zero_division(np.sum(tp), (np.sum(tp) + np.sum(fp)))
        mir = zero_division(np.sum(tp), (np.sum(tp) + np.sum(fn)))
        mif = zero_division(2 * mir * mip, (mir + mip))
        for n, m in zip(['MiP', 'MiR', 'MiF'], [mip, mir, mif]):
            print('{}: {:.5f}'.format(n, m))

        print('###################DONE#########################')

def run_evaluation(meshIDs, pred, true_label):
    # threshold tuning
    _N = len(meshIDs)  # number of class
    _n = 20000  # number of test data
    maximum_iteration = 10
    P_score = pred.tolist()
    T_score = true_label.tolist()
    print(P_score, "\n-----------------P Score--------------------")
    print(T_score, "\n-----------------T Score--------------------")
    # threshold = get_threshold(_N, _n, P_score, T_score)
    threshold = 0.005

    # evaluation
    ks = [1, 3, 5, 10, 15]
    test_labelsIndex = getLabelIndex(T_score)
    precisions = precision_at_ks(P_score, test_labelsIndex, ks=ks)
    for i in range(len(ks)):
        print("precision 1: ", np.mean(precisions[0][i]))
    for i in range(len(ks)):
        print("precision 2: ", np.mean(precisions[1][i]))
    emb = example_based_evaluation(P_score, T_score, threshold, 20000)
    print('emb', emb)
    micro = micro_macro_eval(P_score, T_score, threshold)
    print('micro', micro)

def top_k_predicted(goldenTruth, predictions, k):
    predicted_label = np.zeros(predictions.shape)
    for i in range(len(predictions)):
        goldenK = len(goldenTruth[i])
        if goldenK <= k:
            top_k_index = (predictions[i].argsort()[-goldenK:][::-1]).tolist()
        else:
            top_k_index = (predictions[i].argsort()[-k:][::-1]).tolist()
        for j in top_k_index:
            predicted_label[i][j] = 1
    predicted_label = predicted_label.astype(np.int64)
    return predicted_label

def getLabelIndex(labels):
    label_index = np.zeros((len(labels), len(labels[1])))
    for i in range(0, len(labels)):
        index = np.where(labels[i] == 1)
        index = np.asarray(index)
        N = len(labels[1]) - index.size
        index = np.pad(index, [(0, 0), (0, N)], 'constant')
        label_index[i] = index

    label_index = np.array(label_index, dtype=int)
    label_index = label_index.astype(np.int32)
    return label_index


def binarize_probs(probs, thresholds):
    nb_classes = probs.shape[-1]
    binarized_output = np.zeros_like(probs)

    for k in range(nb_classes):
        binarized_output[:, k] = (np.sign(probs[:, k] - thresholds[k]) + 1) // 2

    return binarized_output


def plot_loss(train_loss, valid_loss, save_path):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, bbox_inches='tight')

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
    test(test_dataset, model, mlb, G, args.batch_sz, device, args.model_name)


if __name__ == "__main__":
    main()
