from bert_dataset import KenmeshDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
class KenmeshDataModule (pl.LightningDataModule):
    
    def __init__(self, text_train, label_train, mesh_mask_train, text_val, label_val, mesh_mask_val, text_test, label_test, mesh_mask_test, tokenizer, device, g, g_node_feature, batch_size=16, max_token_len=200):
        super().__init__()
        self.tr_text = text_train
        self.tr_label = label_train
        self.tr_mesh_mask = mesh_mask_train
        self.val_text = text_val
        self.val_label = label_val
        self.val_mesh_mask = mesh_mask_val
        self.test_text = text_test
        self.test_label = label_test
        self.test_mesh_mask = mesh_mask_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.device = device
        self.g = g
        self.g_node_feature = g_node_feature

    def setup(self, stage=None):
        print("What UP!!!")
        self.train_dataset = KenmeshDataset(texts=self.tr_text,  labels=self.tr_label, mesh_masks = self.tr_mesh_mask, g = self.g, g_node_feature = self.g_node_feature, tokenizer=self.tokenizer, max_len= self.max_token_len, device=self.device)
        self.val_dataset = KenmeshDataset(texts=self.val_text, labels=self.val_label, mesh_masks = self.val_mesh_mask, g = self.g, g_node_feature = self.g_node_feature, tokenizer=self.tokenizer, max_len = self.max_token_len, device=self.device)
        self.test_dataset = KenmeshDataset(texts=self.test_text, labels=self.test_label, mesh_masks = self.test_mesh_mask, g = self.g, g_node_feature = self.g_node_feature, tokenizer=self.tokenizer, max_len = self.max_token_len, device=self.device)

    def get_test_data(self):
        self.test_dataset = KenmeshDataset(texts=self.test_text, labels=self.test_label, mesh_masks = self.test_mesh_mask, g = self.g, g_node_feature = self.g_node_feature, tokenizer=self.tokenizer, max_len = self.max_token_len, device=self.device)
        print("get_test_data: ", self.test_dataset)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size= self.batch_size, shuffle = True , num_workers=8, multiprocessing_context='spawn',  drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader (self.val_dataset,batch_size= self.batch_size, num_workers=8, multiprocessing_context='spawn',  drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader (self.test_dataset,batch_size= self.batch_size, num_workers=8, multiprocessing_context='spawn',  drop_last=False, pin_memory=True)
    
    def predict_dataloader(self):
        return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=4,
        shuffle=False)
    
    def collate_graphs(self):

        return (self.g, self.g_node_feature, self.device)