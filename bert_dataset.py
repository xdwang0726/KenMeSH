from torch.utils.data import Dataset
import torch
import torch.nn as nn

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
        
        embedding_dim = 768
        
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item_idx):
        text = self.texts[item_idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length= self.max_len,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True,
            truncation=True,
            return_tensors = 'pt'
          )
        
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()



        # print("Dataset Labels: ", type(self.labels[item_idx]), torch.tensor(self.labels[item_idx]).size() )
        # print("label feature: ", type(label_feature.transpose(0, 1)), label_feature.transpose(0, 1).size() )
        # print("mesh_masks: ", type(mesh_masks), mesh_masks.size() )
        # print("label_attn_mask: ", type(label_attn_mask), label_attn_mask.size() )

        return {
            'input_ids': input_ids.long() , # Number of tokens in the text
            'attention_mask': attn_mask.long(),
            'label':torch.tensor(self.labels[item_idx]).long(), # Number of labels associated with each document
            'mesh_mask': torch.tensor(self.mesh_masks[item_idx]).long(), #[1, 28415]
        }