import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from sklearn.metrics import f1_score, precision_score, recall_score
import dgl.function as fn

BERT_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

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
    
class KenmeshClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self, n_classes=10,steps_per_epoch=None,n_epochs=3, lr=2e-5):
        super().__init__()

        self.bert=BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True, output_hidden_states = True)
        embedding_dim = self.bert.config.hidden_size
        ksz = 3
        self.cnn = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim , kernel_size=ksz, padding=0, dilation=1),
                                        nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=2),
                                        nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=3),
                                        nn.SELU(), nn.AlphaDropout(p=0.05))
        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)
        self.classifier=nn.Linear(self.bert.config.hidden_size,n_classes) 
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,input_ids, attn_mask, mesh_masks):

        data_module = self.trainer.datamodule

        # Call the collate_graphs function from the DataModule
        g, g_node_feature, device = data_module.collate_graphs()

        g = g.to(device)
        g_node_feature = g_node_feature.to(device)

        """
        * component-wise multiplication A[i,j] * B[i,j] = C[i,j]
        * label_feature [node, feature] -> [28415,768] -(transpose)-> [768,28415] 
        * mesh_mask -> [1,28415]
        * label_attn_mask -> [768, 28415]   
        """
        label_feature = self.gcn(g, g_node_feature)
        mesh_masks = torch.tensor(mesh_masks)  
        label_attn_mask = label_feature.transpose(0, 1) * mesh_masks 
    
        # print("Forward Label Attn: ", type(label_attn_mask), label_attn_mask.size()) # [16, 768, 28415]
        # print("Forward Input Id: ", type(input_ids), input_ids.size()) # [16, 512]
        
        #Take the word tokens to have the sequence length
        outputs = self.bert(input_ids=input_ids,attention_mask=attn_mask)
        output = outputs.last_hidden_state #[16,768] #[16, 512, 768]
        print("Pooler output: ", type(output), output.size()) 

        #CNN output should be: (bs, embed_dim*2, seq_len-ksz+1)
        output = self.cnn(output.permute(0,2,1)) # [1,768,1] # [16, 768, 500]
        print("CNN output: ", type(output), output.size(), output.view(1,-1).size()) 

        # output = self.classifier(output.view(1,-1)) # [1, 28415]
        # print("Classifier output: ", type(output), output.size()) 

        # apply non-linear activation
        # output = nn.functional.relu(output)
        # print("Relu output: ", type(output), output.size()) # [1, 28415]

        # apply label attention mask
        # output_masked = output.unsqueeze(0) * label_attn_mask.float().mean(dim=0, keepdim=True) # [1, 768, 28415]
        # [16, 768, 500] * [768, 28415]
        
        alpha = torch.softmax(torch.matmul(output.transpose(1, 2), label_attn_mask.float()), dim=1)
        print("output_masked: ", type(alpha), alpha.size()) # [16, 500, 28415]

        print("Test 1: ", type(output), type(alpha))
        output_features = torch.matmul(output, alpha).transpose(1, 2) # [16, 28415, 768]
        print("output_features", type(output_features), output_features.size())

        x_feature = torch.sum(output_features * label_feature, dim=2) # [16, 28415]
        print("x_feature: ", x_feature, x_feature.size())

        # alpha_text = torch.softmax(torch.matmul(output, label_attn_mask), dim=0)

        # apply sigmoid activation to get predicted probabilities
        # probs = torch.sigmoid(x_feature)

        # print("Probs: ", type(probs), probs.size())
            
        return x_feature

    def training_step(self,batch,batch_idx):
        print("training_step")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        mesh_mask = batch['mesh_mask']
        
        probs = self(input_ids,attention_mask, mesh_mask) # torch.Size([32, 28415]); 32 = batch_size
        
        # print("Hello Train: ", probs.size(), labels.size())
        # # flatten predicted probabilities
        # probs_flat = probs.mean(dim=1).view(-1, 28415)

        # # flatten target labels
        # targets_flat = labels.float().mean(dim=0).view(-1, 28415)
        loss = self.criterion(probs,labels.float())
        
        self.log('train_loss',loss , prog_bar=True,logger=True)
        
        return {"loss" :loss, "predictions":probs, "labels": labels }


    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'] # Number of labels associated with each document Ex: 12 [16,28415]
        mesh_mask = batch['mesh_mask'] # [1,28415]
        
        probs = self(input_ids,attention_mask, mesh_mask)
        print("Hello Val: ", probs.size(), labels.size())

        # # flatten predicted probabilities
        # probs_flat = probs.mean(dim=1).view(-1, 28415)
        # print("Hello probs_flat: ", probs_flat.size())

        # # flatten target labels
        # targets_flat = labels.float().mean(dim=0).view(1, 28415)
        # print("Hello targets_flat: ", targets_flat.size())

        loss = self.criterion(probs,labels.float())

        # loss = self.criterion(outputs,labels.float())
        self.log('val_loss',loss , prog_bar=True,logger=True)
        
        # Calculate the metrics
        y_true = labels.cpu().numpy()
        print("Val Cal")
        print("y_true : ", y_true, type(y_true), len(y_true))
        print("y_pred : ", probs, type(probs), len(probs))
        y_pred = torch.sigmoid(probs).cpu().numpy() > 0.5

        print("y_pred sigmoid: ", y_pred, type(y_pred), len(y_pred))
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        # Log the metrics to the output
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)

        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        mesh_mask = batch['mesh_mask']
        
        probs = self(input_ids,attention_mask, mesh_mask)
        print("Hello test: ", probs.size(), labels.size())

        # # flatten predicted probabilities
        # probs_flat = probs.mean(dim=1).view(-1, 28415)

        # # flatten target labels
        # targets_flat = labels.float().mean(dim=0).view(-1, 28415)
        loss = self.criterion(probs,labels.float())
        self.log('test_loss',loss , prog_bar=True,logger=True)

        # Calculate the metrics
        y_true = labels.cpu().numpy()
        y_pred = torch.sigmoid(probs).cpu().numpy() > 0.5
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        # Log the metrics to the output
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        
        return loss
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.lr)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)

        return [optimizer], [scheduler]
