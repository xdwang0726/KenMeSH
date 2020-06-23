import ijson
import spacy
from spacy.tokenizer import Tokenizer
from torchtext.data import Field
from torchtext import data

########## Load Dataset and Preprocessing ##########

f = open('', encoding="utf8")
objects = ijson.items(f, 'articles.item')


# Tokenize
nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)

TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=True, use_vocab=True)
fields = [('abstractText', TEXT)]

# load dataset
meshIndexing = data.TabularDataset(path='train.json', format='json', fields=fields)
