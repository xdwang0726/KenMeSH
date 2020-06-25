import os

import ijson
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm

from utils import tokenize, TextMultiLabelDataset

########## Load Dataset and Preprocessing ##########

##### load training data #####
f = open('train.json', encoding="utf8")
objects = ijson.items(f, 'articles.item')

pmid = []
all_text = []
label = []
label_id = []

for obj in tqdm(objects):
    try:
        ids = obj["pmid"].strip()
        text = obj["abstractText"].strip()
        original_label = obj["meshMajor"]
        mesh_id = obj['meshId']
        pmid.append(ids)
        all_text.append(text)
        label.append(original_label)
        label_id.append(mesh_id)
    except AttributeError:
        print(obj["pmid"].strip())

# read full MeSH ID list
with open("MeshIDList.txt", "r") as ml:
    meshIDs = ml.readlines()

meshIDs = [ids.strip() for ids in meshIDs]
print('Number of labels: ', len(meshIDs))
mlb = MultiLabelBinarizer(classes=meshIDs)

label_vectors = mlb.fit_transform(label_id)

# write to dataframe
df_train = pd.DataFrame(pmid, columns=['PMID'])
df_train['text'] = all_text
df_train = pd.concat([df_train, pd.DataFrame(label_vectors, columns=meshIDs)], axis=1)

text_col = 'text'
label_col = meshIDs

##### load test data #####


TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, truncate_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.FloatTensor)

train = TextMultiLabelDataset(df_train, text_field=TEXT, label_field=LABEL, txt_col=text_col, lbl_cols=label_col)

# build vocab
file_path = 'BioWord2Vec.vec'
cache, name = os.path.split(file_path)
vectors = Vectors(name=name, cache=cache)
vectors.unk_init = init.xavier_uniform
TEXT.build_vocab(train, vectors=vectors)

# using the training corpus to create the vocabulary
using_gpu = True
train_iter = data.Iterator(dataset=train, batch_size=32, train=True, repeat=False, device=0 if using_gpu else -1)
vocab_size = len(TEXT.vocab.itos)
num_classes = len(meshIDs)
