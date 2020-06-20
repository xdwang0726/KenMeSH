from utils import text_preprocess

import spacy
from spacy.tokenizer import Tokenizer

########## Load Dataset and Preprocessing ##########
with open("test_text.txt", "r") as f:
    text = f.readlines()

text = list(filter(None, text))
print('Total number of document: %s' % len(text))

processed_text = []
data_length = len(text)
for i in range(0, data_length):
    token = text[i].lstrip('0123456789.- ')
    token = text_preprocess(token)
    processed_text.append(token)

# Tokenize
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
