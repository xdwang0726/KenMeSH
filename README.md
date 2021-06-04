# MeSH_Indexing_RGCN


## Prerequisites
dgl_gpu==0.6.1
ijson
sklearn
tokenizers==0.9.3
torch==1.6.0
torchtext==0.6.0
transformers==3.5.1
tqdm==4.60.0


How to embed the nodes?
  1. use average word embeddings in the descriptor (remove comma, and cover to lower case)
  2. node2vec 
