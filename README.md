# MeSH_Indexing_RGCN


## Prerequisites
----
dgl_gpu==0.6.1 <br/>
ijson <br/>
sklearn <br/>
tokenizers==0.9.3 <br/>
torch==1.6.0 <br/>
torchtext==0.6.0 <br/>
transformers==3.5.1 <br/>
tqdm==4.60.0 <br/>


How to embed the nodes?
  1. use average word embeddings in the descriptor (remove comma, and cover to lower case)
  2. node2vec 
