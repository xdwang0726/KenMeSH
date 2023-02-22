from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

gcn_msg = fn.copy_u('h', 'm')
gcn_reduce = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        """
        inputs: g,       object of Graph
                feature, node features
        """
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class LabelNet(nn.Module):
    def __init__(self, hidden_gcn_size, num_classes, in_node_features):
        super(LabelNet, self).__init__()
        self.gcn1 = GCNLayer(in_node_features, hidden_gcn_size)
        self.gcn2 = GCNLayer(hidden_gcn_size, num_classes)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        return x

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
        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)
        
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

        """
        * component-wise multiplication A[i,j] * B[i,j] = C[i,j]
        * label_feature [node, feature] -> [28415,768] -(transpose)-> [768,28415] 
        * mesh_mask -> [1,28415]
        * label_attn_mask -> [768, 28415]   
        """
        label_feature = self.gcn(self.g, self.g_node_feature)
        mesh_masks = torch.tensor(self.mesh_masks[item_idx])  
        label_attn_mask = label_feature.transpose(0, 1) * mesh_masks 

        # print("Dataset Labels: ", type(self.labels[item_idx]), torch.tensor(self.labels[item_idx]).size() )
        # print("label feature: ", type(label_feature.transpose(0, 1)), label_feature.transpose(0, 1).size() )
        # print("mesh_masks: ", type(mesh_masks), mesh_masks.size() )
        # print("label_attn_mask: ", type(label_attn_mask), label_attn_mask.size() )

        return {
            'input_ids': input_ids.long() , # Number of tokens in the text
            'attention_mask': attn_mask.long(),
            'label':torch.tensor(self.labels[item_idx]).long(), # Number of labels associated with each document
            'mesh_mask': torch.tensor(self.mesh_masks[item_idx]).long(), #[1, 28415]
            'label_attn_mask': label_attn_mask.long() # [768, 28415] 
        }