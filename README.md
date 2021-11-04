# KenMeSH: Knowledge-enhanced End-to-end Biomedical Text Labelling
Currently, Medical Subject Headings (MeSH) are manually assigned to every biomedical article published and subsequently recorded in the PubMed database to facilitate retrieving relevant information. With the rapid growth of the PubMed database, large-scale biomedical document indexing becomes increasingly important. MeSH indexing is a challenging task for machine learning, as it needs to assign multiple labels to each article from an extremely large hierachically organized collection. To address this challenge, we propose an end-to-end model that combines new text features and a dynamic knowledge-enhanced mask attention that integrates document features with MeSH label hierarchy and journal correlation features to index MeSH terms. Experimental results show the proposed method achieves state-of-the-art performance on a number of measures.

## Required Packages
- Python 3.7
- numpy==1.11.1
- dgl-gpu==0.6.1
- nltk==3.5
- scikit-learn==0.23.0
- scipy==1.4.1
- sklearn==0.0
- spacy==2.2.2
- tokenizers==0.9.3
- torch==1.6.0
- torchtext==0.6.0
- tqdm==4.60.0
- transformers==3.5.1

## Usage
### Training 
