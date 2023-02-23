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
from torch.utils.data import DataLoader,Dataset,RandomSampler, SequentialSampler, TensorDataset
import pytorch_lightning as pl
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from transformers import BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint

# Modules
from bert_data_module import KenmeshDataModule
from bert_classifier_lightning import KenmeshClassifier

# Global Variables
BERT_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
N_EPOCHS = 12
BATCH_SIZE = 16
MAX_LEN = 512
LR = 2e-05

# convert probabilities into 0 or 1 based on a threshold value
def classify(pred_prob,thresh):
    y_pred = []

    for tag_label_row in pred_prob:
        temp=[]
        for tag_label in tag_label_row:
            # print("tag_label: ", tag_label)
            if tag_label >= thresh:
                temp.append(1) # Infer tag value as 1 (present)
            else:
                temp.append(0) # Infer tag value as 0 (absent)
        y_pred.append(temp)

    return y_pred

def prepare_dataset(MeSH_id_pair_file, graph_file, device):
    """ Load Dataset and Preprocessing """
    
    text_test = pickle.load(open("text_test.pkl", 'rb'))
    label_test = pickle.load(open("label_test.pkl", 'rb'))
    mesh_mask_test = pickle.load(open("mesh_mask_test.pkl", 'rb'))

    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()
    meshIDs = list(mapping_id.values())

    n_classes = len(meshIDs)

    print('Total number of labels %d' % n_classes)


    # Prepare label features
    print('Load graph')
    G = load_graphs(graph_file)[0][0]
    # print('graph', G.ndata['feat'].shape) # [28415, 768]

    steps_per_epoch = len(text_train)//BATCH_SIZE

    # Instantiate and set up the data_module
    Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    kenmesh_data_Module = KenmeshDataModule(text_train, label_train, mesh_mask_train,\
         text_val, label_val, mesh_mask_val, text_test, label_test, mesh_mask_test, Bert_tokenizer,\
          device, G, G.ndata['feat'], BATCH_SIZE, MAX_LEN)
    kenmesh_data_Module.setup()

    print('prepare dataset and labels graph done!')
    # return len(meshIDs), mlb, train_dataset, valid_dataset, vectors, G
    # return text_train, label_train, mesh_mask_train, text_val, label_val, mesh_mask_val, text_test, label_test, mesh_mask_test
    return kenmesh_data_Module, steps_per_epoch, n_classes

def main():
    #region Arguments
     
    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--dataset_path')
    parser.add_argument('----meSH_pair_path')
    parser.add_argument('--word2vec_path')
    parser.add_argument('--meSH_pair_path')
    parser.add_argument('--graph')
    parser.add_argument('--model')
    parser.add_argument('--model_name', default='Full', type=str)

    # environment
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

    prepare_dataset()



    n_classes = 28415
    steps_per_epoch = len(text_test)//BATCH_SIZE

    model = KenmeshClassifier(n_classes=n_classes, steps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS, lr=LR)
    model.load_state_dict(torch.load(args.model))

    # Tokenize all questions in x_test
    input_ids = []
    attention_masks = []

    Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # Set the batch size.  
    TEST_BATCH_SIZE = 16  

    # Create the DataLoader.
    # pred_data = TensorDataset(input_ids, attention_masks, mesh_mask_test, labels)
    pred_data = KenmeshDataset(texts=text_test, labels=label_test, mesh_masks = mesh_mask_test, g = g, g_node_feature = g_node_feature, tokenizer=tokenizer, max_len = max_token_len, device=device)
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=TEST_BATCH_SIZE, drop_last=True)

    flat_pred_outs = 0
    flat_true_labels = 0

    # Put model in evaluation mode
    model = model.to(device) # moving model to cuda
    model.eval()

    # Tracking variables 
    pred_outs, true_labels = [], []
    #i=0
    # Predict 
    for batch in pred_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
    
        # Unpack the inputs from our dataloader
        b_input_ids, b_attn_mask, b_label_attn_mask, b_labels = batch
    
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            pred_out = model(b_input_ids,b_attn_mask, b_label_attn_mask)
            pred_out = torch.sigmoid(pred_out)
            # Move predicted output and labels to CPU
            pred_out = pred_out.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            #i+=1
            # Store predictions and true labels
            #print(i)
            #print(outputs)
            #print(logits)
            #print(label_ids)
        print(pred_out[0], label_ids)
        pred_outs.append(pred_out[0])
        true_labels.append(label_ids)

    print("Pred output: ", pred_outs)
    print("True Labels output: ", true_labels)

    # Combine the results across all batches. 
    flat_pred_outs = np.concatenate(pred_outs, axis=0)

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    print("flat_pred_outs, flat_true_labels:", flat_pred_outs.shape , flat_true_labels.shape)

    #define candidate threshold values
    threshold  = np.arange(0.4,0.51,0.01)
    print(threshold)
    
    # print(flat_pred_outs[3])
    # print(flat_true_labels[3])

    scores=[] # Store the list of f1 scores for prediction on each threshold

    #convert labels to 1D array
    y_true = flat_true_labels.ravel() 

    for thresh in threshold:
        
        #classes for each threshold
        pred_bin_label = classify(flat_pred_outs,thresh) 

        #convert to 1D array
        y_pred = np.array(pred_bin_label).ravel()
        print("y_true: ", len(y_true))
        print("y_pred: ", len(y_pred))
        scores.append(metrics.f1_score(y_true,y_pred))
    
    # find the optimal threshold
    opt_thresh = threshold[scores.index(max(scores))]
    print(f'Optimal Threshold Value = {opt_thresh}')

    #predictions for optimal threshold
    y_pred_labels = classify(flat_pred_outs,opt_thresh)
    y_pred = np.array(y_pred_labels).ravel() # Flatten

    print(metrics.classification_report(y_true,y_pred))

if __name__ == "__main__":
    main()