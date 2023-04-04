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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Modules
from bert_data_module import KenmeshDataModule
from bert_classifier_lightning import KenmeshClassifier

# Global Variables
BERT_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
BATCH_SIZE = 16
MAX_LEN = 512
# LR = 2e-05

def split_train_test_val(texts, labels, mesh_masks, test_size = 0.1):
    """
    * This Custom function can take as much arguments as required which 
    * train_test_split from sklearn doesn't support.
    * x_train,x_test,y_train,y_test = train_test_split(texts, yt, test_size=0.1, random_state=42,shuffle=True)
    """
    text_train, label_train, mesh_mask_train, text_test, label_test, mesh_mask_test = [], [], [], [], [], []
    total_len = len(texts)
    test_len = int(total_len * test_size)
    train_len = total_len - test_len

    nums = list(range(0, total_len))
    random.shuffle(nums)
    print(test_len, train_len)

    for i, num in enumerate(nums):
        if i < train_len:  
            text_train.append(texts[num])
            label_train.append(labels[num])
            mesh_mask_train.append(mesh_masks[num])
        else:
            text_test.append(texts[num])
            label_test.append(labels[num])
            mesh_mask_test.append(mesh_masks[num]) 
        # if i == 0:
        #     print("First doc: ", type(label_train), type(mesh_mask_train))
        #     print(label_train[0], type(label_train[0]), np.count_nonzero(label_train[0]))           

    print(f"Total number of documents: {total_len}, train set: {len(text_train)}, test set: {len(text_test)}")
    
    return text_train, label_train, mesh_mask_train, text_test, label_test, mesh_mask_test

def prepare_dataset(dataset_path, MeSH_id_pair_file, graph_file, device):
    """ Load Dataset and Preprocessing """
    
    print('Start loading training data')

    mesh_mask = []
    texts = []
    label_id = []
    pubmed_ids = []
    titles = []

    f = open(dataset_path, encoding="utf8")
    objects = ijson.items(f, 'articles.item') 

    for i, obj in enumerate(tqdm(objects)):
        title = obj["title"]
        abstractText = obj["abstractText"]

        if len(title) < 1 or len(abstractText) < 1:
            continue      

        # titles = title.split(" ")
        # abstractTexts = abstractText.split(" ")
        text = title + " " +abstractText
        pubmed_ids.append(obj["pmid"])
        mesh_mask.append(obj["meshMask"])
        texts.append(text)
        label_id.append(list(obj["meshID"].keys()))
        titles.append(title)
        # if i == 0:
        #     # print("Mesh Mask: ", len(mesh_mask[0]), len(mesh_mask[0][0]), mesh_mask[0].count(1), mesh_mask[0][0].count(1))
        #     print("Label: ", obj["meshID"])
        #     print("Label ID: ", len(label_id[0]), label_id[0])
    
    print('Finish loading training data')
    f.close()

    print('number of training data %d' % len(texts))

    print('load and prepare Mesh')

    mapping_id = {}
    with open(MeSH_id_pair_file, 'r') as f:
        for line in f:
            (key, value) = line.split('=')
            mapping_id[key] = value.strip()
    meshIDs = list(mapping_id.values())

    # print("meshIDs: ", meshIDs)

    n_classes = len(meshIDs)

    print('Total number of labels %d' % n_classes)

    # Encode the tags(labels) in a binary format in order to be used for training

    mlb = MultiLabelBinarizer(classes=meshIDs)
    mlb.fit(meshIDs)
    yt = mlb.fit_transform(label_id)

    # print()
    # print("title: ", titles[0])
    # print("pmid: ", pubmed_ids[0])
    # print("1: ", np.count_nonzero(yt[0])) # yt[0] -> [28415] -> count_of_nonzero == number of lebels for the doc
    # print("2: ", mlb.inverse_transform(yt[0].reshape(1,-1)))
    # print("3: ", mlb.classes_)

    # print("4: ", label_id[0])
    # print("5: ", mesh_mask[0][0].count(1)) # mesh_mask -> [1, 28415]

    # Prepare label features
    print('Loading graph...')
    G = load_graphs(graph_file)[0][0]
    # print('graph', G.ndata['feat'].shape) # [28415, 768]
    
    # Preparing training and test datasets
    print('prepare training and test sets')

    # First Split for Train and Test
    text_train, label_train, mesh_mask_train, text_test, label_test, mesh_mask_test = split_train_test_val(texts, yt, mesh_mask, test_size = 0.2)
    text_train, label_train, mesh_mask_train, text_val, label_val, mesh_mask_val = split_train_test_val(text_train, label_train, mesh_mask_train, test_size = 0.2)
    
    # print("text_test, label_test, mesh_mask_test: ", text_test, label_test, mesh_mask_test)
    # Saving test dataset to pickle for using in evaluation
    pickle.dump(text_train, open("text_train_full.pkl", 'wb'))
    pickle.dump(label_train, open("label_train_full.pkl", 'wb'))
    pickle.dump(mesh_mask_train, open("mesh_mask_train_full.pkl", 'wb'))

    pickle.dump(text_test, open("text_test_full.pkl", 'wb'))
    pickle.dump(label_test, open("label_test_full.pkl", 'wb'))
    pickle.dump(mesh_mask_test, open("mesh_mask_test_full.pkl", 'wb'))

    pickle.dump(text_val, open("text_val_full.pkl", 'wb'))
    pickle.dump(label_val, open("label_val_full.pkl", 'wb'))
    pickle.dump(mesh_mask_val, open("mesh_mask_val_full.pkl", 'wb'))

    print("-"*10, "pickle data dumpled", "-"*10)

    print("Batch size: ", BATCH_SIZE)
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

def evaluation():
    pass 
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
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--nKernel', type=int, default=768) # Changed here for embed dimension
    parser.add_argument('--ksz', default=5)
    parser.add_argument('--hidden_gcn_size', type=int, default=768) # Changed here for embed dimension
    parser.add_argument('--embedding_dim', type=int, default=768) # Changed here for embed dimension
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--atten_dropout', type=float, default=0.5)
    
    # hyper params
    # lr -> 0.001, 0.0001, 0.0003, 0.0005
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_sz', type=int, default=16)
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

    global BATCH_SIZE
    BATCH_SIZE = args.batch_sz
    num_epochs = args.num_epochs
    # Get dataset and label graph & Load pre-trained embeddings
    kenmesh_data_Module, steps_per_epoch, n_classes= prepare_dataset(args.dataset_path, args.meSH_pair_path, args.graph, device)
    
    # Inittialising Bert Classifier Model
    model = KenmeshClassifier(num_labels=n_classes, steps_per_epoch=steps_per_epoch, n_epochs=num_epochs, lr=args.lr)
    
    model.to(device)
   
    #Initialize Pytorch Lightning callback for Model checkpointing

    # saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',# monitored quantity
        filename='QTag-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3, #  save the top 3 models
        mode='min', # mode of the monitored quantity  for optimization
    )

    # Instantiate the Model Trainer
    trainer = pl.Trainer(max_epochs = num_epochs , devices = 1, accelerator='gpu', callbacks=[EarlyStopping(monitor="val_loss", mode="max"), checkpoint_callback])
    # trainer = pl.Trainer(max_epochs = N_EPOCHS , devices = 1, accelerator='gpu', profiler="simple", callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_callback])
    # print("Best checkpoint path: ", checkpoint_callback.best_model_path)
    # pickle.dump(checkpoint_callback.best_model_path, open("checkpoint.pkl", "wb"))

    # Train the Classifier Model
    print("Training...")
    trainer.fit(model, kenmesh_data_Module)

    torch.save(model.state_dict(), args.save_model_path)

    # Evaluate the model performance on the test dataset
    print("Evaluation: ")
    print(trainer.test(model,datamodule=kenmesh_data_Module))
    


if __name__ == "__main__":
    main()