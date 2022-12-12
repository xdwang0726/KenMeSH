# KenMeSH: Knowledge-enhanced End-to-end Biomedical Text Labelling
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
- faiss

## Usage
### Build a graph
```
python -u build_graph.py --meSH_pair_path MeSH_name_id_mapping_2019.txt --mesh_parent_children_path MeSH_parent_child_mapping_2019 --word2vec_path BioWord2Vec_standard.w2v --graph_type ‘GCN’
```
### Get journal name and MeSH term corelations
```
python -u journal_info.py --data train.json --save journal_info.pkl
```
### Get MeSH mask using the training data
Steps (see details in ```get_mesh_mask.py```):
* get idf vector for each document
```
python -u get_mesh_mask.py --allMeSH train.json --meSH_pair_path MeSH_name_id_mapping_2019.txt --  save_path_idf idf.json
```
* get masks using KNN
```
python -u get_mesh_mask.py --allMeSH train.json --meSH_pair_path MeSH_name_id_mapping_2019.txt --save_path_idf idf.json
```
* get masks from journal and merge the masks generated from neighbours 
```commandline
python -u get_mesh_mask.py --allMeSH train.json --meSH_pair_path MeSH_name_id_mapping_2019.txt --neigh_path neigh.json --journal_info journal_info.pkl --threshold 0.5 --save_path dataset.json
```

### Training 
```commandline
python -u run_classifier_multigcn.py --title_path pmc_title.pkl --abstract_path pmc_abstract.pkl --label_path pmc_meshLabel.pkl --mask_path mesh_mask.pkl --meSH_pair_path MeSH_name_id_mapping_pmc_2020.txt --word2vec_path BioWord2Vec_standard.w2v --graph gcn_pmc.bin --save-model-path model.pt --batch_sz 32 --model_name 'Full'
```

### Evaluation
```commandline
python -u run_classifier_multigcn.py --title_path pmc_title.pkl --abstract_path pmc_abstract.pkl --label_path pmc_meshLabel.pkl --mask_path mesh_mask.pkl --meSH_pair_path MeSH_name_id_mapping_pmc_2020.txt --word2vec_path BioWord2Vec_standard.w2v --graph gcn_pmc.bin model model.pt --batch_sz 32 --model_name 'Full'
```
## Citing
If you use KenMeSH in your work, please consider citing our paper
