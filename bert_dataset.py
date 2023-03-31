import ast

from torch.utils.data import Dataset
import torch
import torch.nn as nn

from utils import sparse_to_dense

class KenmeshDataset (Dataset):
    def __init__(self, texts, labels, mesh_masks, tokenizer, max_len, device, g, g_node_feature):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.mesh_masks = mesh_masks
        self.max_len = max_len
        self.device = device
        self.g = g
        self.g_node_feature = g_node_feature
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item_idx):
        text = self.texts[item_idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=False,
            max_length= self.max_len,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True,
            truncation=True,
            return_tensors = 'pt'
          )
        
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()

        res = ast.literal_eval(self.mesh_masks[item_idx])
        mesh_mask = sparse_to_dense(res)

        label = torch.tensor(self.labels[item_idx]).long()
        
        # print("Word tokens: ", text)
        # print("Dataset Labels: ", type(self.labels[item_idx]), torch.tensor(self.labels[item_idx]).size() ) # <class 'numpy.ndarray'> torch.Size([28415])
        # print("input_ids: ", type(input_ids) ) # <class 'torch.Tensor'>
        # print("mesh_masks: ", type(mesh_mask) ,mesh_mask.shape) # <class 'numpy.ndarray'> (1, 28415)
        # print("label: ", type(label) ,label.shape) # <class 'torch.Tensor'> torch.Size([28415])
        # print("attn_mask: ", type(attn_mask)) # <class 'torch.Tensor'>

        return {
            'input_ids': input_ids.long() , # Number of tokens in the text
            'attention_mask': attn_mask.long(),
            'label': label, # Number of labels associated with each document
            'mesh_mask': torch.tensor(mesh_mask).long(), #[1, 28415]
        }