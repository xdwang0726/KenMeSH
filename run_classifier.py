import ijson
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import torch

import spacy
from spacy.tokenizer import Tokenizer
from torchtext.data import Field
from torchtext import data

########## Load Dataset and Preprocessing ##########

# load the data
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
df = pd.DataFrame(pmid, columns=['PMID'])
df['text'] = all_text
pd.concat([df, pd.DataFrame(label_vectors)], axis=1)

#
TEXT = data.Field(tokenize=tokenize, lower=True, batch_first=True, truncate_first=True)
LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, tensor_type=torch.FloatTensor)







# Tokenize
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=True, use_vocab=True)
fields = [('abstractText', TEXT)]

# load dataset
meshIndexing = data.TabularDataset(path='train.json', format='json', fields=fields)
