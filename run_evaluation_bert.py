# System
import argparse
import io
import pickle

# Utils
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import ijson
import random
from dgl.data.utils import load_graphs

# ML Libraries
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint

# Modules
from bert_data_module import KenmeshDataModule
from bert_classifier_lightning import KenmeshClassifier
from eval_helper import getLabelIndex, precision_at_ks, example_based_evaluation, micro_macro_eval, zero_division

# Global Variables
BERT_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
N_EPOCHS = 12
BATCH_SIZE = 16
MAX_LEN = 512
LR = 2e-05

def prepare_dataset(text_test, label_test, mesh_mask_test, MeSH_id_pair_file, graph_file, device):
    """ Load Dataset and Preprocessing """

    print("text_test: ", len(text_test), len(text_test[0]))
    print("label_test: ", len(label_test), label_test[0])
    print("mesh_mask_test: ", len(mesh_mask_test), len(mesh_mask_test[0]))
    print('number of test data %d' % len(text_test))

    # print('load and prepare Mesh')

    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()
    meshIDs = list(mapping_id.values())

    n_classes = len(meshIDs)

    print('Total number of labels %d' % n_classes)

    # Encode the tags(labels) in a binary format in order to be used for training

    # mlb = MultiLabelBinarizer(classes=meshIDs)
    # yt = mlb.fit_transform(label_test)

    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]
    # print('graph', G.ndata['feat'].shape) # [28415, 768]
    


    steps_per_epoch = len(text_test)//BATCH_SIZE

    # Instantiate and set up the data_module
    Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    kenmesh_data_Module = KenmeshDataModule([], [], [],\
         [], [], [], text_test, label_test, mesh_mask_test, Bert_tokenizer,\
          device, G, G.ndata['feat'], BATCH_SIZE, MAX_LEN)
    print("Big Print: ", kenmesh_data_Module.get_test_data())



    print('prepare dataset and labels graph done!')
    # return len(meshIDs), mlb, train_dataset, valid_dataset, vectors, G
    # return text_train, label_train, mesh_mask_train, text_val, label_val, mesh_mask_val, text_test, label_test, mesh_mask_test
    return kenmesh_data_Module, steps_per_epoch, n_classes, G


def evaluation(P_score, T_score):
    T_score = torch.tensor(T_score)
    P_score = np.concatenate(P_score, axis=0) # 3d -> 2d array
    T_score = np.concatenate(T_score, axis=0)
    # T_score = T.numpy()
    print("True Label load done", type(T_score))
    print("P Label load done", type(P_score))
    # print(T_score)
    threshold = np.array([0.0005] * 28415)

    test_labelsIndex = getLabelIndex(T_score)
    print("test_labelsIndex: ", test_labelsIndex, type(test_labelsIndex))
    print("P_score: ", P_score, type(P_score))
    precisions = precision_at_ks(P_score, test_labelsIndex, ks=[1, 3, 5])
    print('p@k', precisions)

    emb = example_based_evaluation(P_score, T_score, threshold, 16)
    print('(ebp, ebr, ebf): ', emb)

    micro = micro_macro_eval(P_score, T_score, threshold)
    print('mi/ma(MiF, MiP, MiR, MaF, MaP, MaR): ', micro) 

def test(kenmesh_data_Module, model, G, device):
    data_loader = kenmesh_data_Module.test_dataloader()

    with torch.no_grad():

        model.eval()

        # initialize empty lists to store predicted label features and true labels
        predicted_label_features = []
        true_labels = []

        # iterate over batches in the evaluation data loader
        for batch in tqdm(data_loader):
            # get input features and true labels from batch
            torch.cuda.empty_cache()

            input_ids = batch['input_ids'].to(device) 
            attention_mask = batch['attention_mask'].to(device)
            mesh_mask = batch['mesh_mask'].to(device)
            g, g_node_feature = G, G.ndata['feat']
            g = g.to(device)
            g_node_feature = g_node_feature.to(device)
            label = batch['label']

            print("1 input_ids: ", input_ids, input_ids.shape)
            print("2 attention_mask: ", attention_mask, attention_mask.shape)
            print("3 label: ", label, label.shape)
            print("4 Mesh_mask: ", mesh_mask, mesh_mask.shape)

            # make predictions using the model
            with torch.no_grad():
                output = model(input_ids,attention_mask, mesh_mask)
            o = torch.sigmoid(output)
            o = output.data.cpu().numpy()
            print("Output: ", type(output), output)
            print("O: ", o)
            predicted_label_features.append(o)
            del output
            true_labels.append(label)
            torch.cuda.empty_cache()

        # concatenate the predicted label features and true labels across batches
        # predicted_label_features = torch.cat(predicted_label_features)
        # true_labels = torch.cat(true_labels)

    return predicted_label_features, true_labels

    
def main():
    #region All parser arguments

    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--dataset_path')
    parser.add_argument('----meSH_pair_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--graph')
    parser.add_argument('--save-model-path')
    parser.add_argument('--model_name', default='Full', type=str)

    # environment
    parser.add_argument('--model')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=768) # Changed here for embed dimension
    parser.add_argument('--ksz', default=3)
    parser.add_argument('--hidden_gcn_size', type=int, default=768) # Changed here for embed dimension
    parser.add_argument('--embedding_dim', type=int, default=768) # Changed here for embed dimension
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--atten_dropout', type=float, default=0.5)
    
    # hyper params
    # lr -> 0.001, 0.0001, 0.0003, 0.0005
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--scheduler_step_sz', type=int, default=2)
    parser.add_argument('--lr_gamma', type=float, default=0.9)

    args = parser.parse_args()

    #endregion

    # GPU Device assignmentt
    torch.backends.cudnn.benchmark = True
    n_gpu = torch.cuda.device_count()  # check if it is multiple gpu
    print('{} gpu is avaliable'.format(n_gpu))
    torch.set_float32_matmul_precision('medium')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('Device:{}'.format(device))

    # Saving test dataset to pickle for using in evaluation
    text_test = pickle.load(open("text_train.pkl", 'rb'))
    label_test = pickle.load(open("label_train.pkl", 'rb'))
    mesh_mask_test = pickle.load(open("mesh_mask_train.pkl", 'rb'))

    # Get dataset and label graph & Load pre-trained embeddings
    kenmesh_data_Module, steps_per_epoch, n_classes, G= prepare_dataset(text_test, label_test, mesh_mask_test, args.meSH_pair_path, args.graph, device)
    
    # checkpoint_callback = pickle.load(open('checkpoint.pkl', 'rb'))
    # model_path = checkpoint_callback.best_model_path
    # Inittialising Bert Classifier Model
    model = KenmeshClassifier(n_classes=n_classes, steps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS, lr=LR)

    # checkpoint = torch.load('/KenMeSH-master/lightning_logs/version_250/checkpoints/QTag-epoch=09-val_loss=0.01.ckpt')
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(torch.load(args.model))

    # model.load_state_dict(torch.load(model_path))

    model.to(device)

    print("Model load successful...")

    G.to(device)

    trainer = pl.Trainer(max_epochs = N_EPOCHS , devices = 1, accelerator='gpu')
    result = trainer.predict(model, kenmesh_data_Module)

    predicted_labels = []
    true_labels = []
    for batch in result:
        predicted_labels.append(batch["predictions"])
        true_labels.append(batch["labels"])

    # predictions = result[0]["predictions"]

    # Print the predictions
    print("predictions: ", result)
   
    # Evaluate the model performance on the test dataset
    # print("Evaluation: ")
    # predicted_label_features, true_labels = test(kenmesh_data_Module, model, G, device)  

    # print("predicted_label_features", type(predicted_label_features))
    # print("true_labels", type(true_labels))

    print("predicted_labels: ", type(predicted_labels), len(predicted_labels), predicted_labels)
    print("true_labels: ", type(true_labels), len(true_labels), true_labels)
    # np.save("pred2", predicted_labels)
    torch.save(predicted_labels, "pred2")  
    torch.save(true_labels, "true_label2")  
    # print("pred and true labels saved")
    # evaluation(predicted_label_features, true_labels)


if __name__ == "__main__":
    main()