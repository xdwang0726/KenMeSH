import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from sklearn.metrics import f1_score, precision_score, recall_score
import dgl.function as fn

# from eval_helper import getLabelIndex, precision_at_ks, example_based_evaluation, micro_macro_eval, zero_division

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
    def __init__(self, num_labels=28415,steps_per_epoch=None,n_epochs=3, lr=2e-5):
        super().__init__()

        self.bert=BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True, output_hidden_states = True)
        for param in self.bert.parameters():
            param.requires_grad = False 
        embedding_dim = self.bert.config.hidden_size
        ksz = 5
        # self.dense_layer = nn.Linear(768, num_labels)
        self.cnn = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim , kernel_size=ksz, padding=0, dilation=2),
                                        nn.BatchNorm1d(embedding_dim), nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=4),
                                        nn.BatchNorm1d(embedding_dim), nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=8),
                                        nn.BatchNorm1d(embedding_dim), nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=16),
                                        nn.BatchNorm1d(embedding_dim), nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=8),
                                        nn.BatchNorm1d(embedding_dim), nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=4),
                                        nn.BatchNorm1d(embedding_dim), nn.SELU(), nn.AlphaDropout(p=0.05),
                                        nn.Conv1d(embedding_dim, embedding_dim, kernel_size=ksz, padding=0, dilation=2),
                                        nn.BatchNorm1d(embedding_dim), nn.SELU(), nn.AlphaDropout(p=0.05))
        self.gcn = LabelNet(embedding_dim, embedding_dim, embedding_dim)
        # self.flatten = nn.Flatten(start_dim=1)
        # self.classifier=nn.Linear(768, 28415) 
        # self.classifier2=nn.Linear(96000,28415) 
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,input_ids, attn_mask, mesh_mask, g=[], g_node_feature=[], device="cuda"):

        # print("1 input_ids: ", input_ids, input_ids.shape)
        # print("2 attention_mask: ", attn_mask, attn_mask.shape)
        # print("4 Mesh_mask: ", mesh_mask, mesh_mask.shape)
        # ones = (mesh_mask[0][0] == 1).sum(dim=0)
        # print("Mesh_mask Ones: ", ones)



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
        # print("g_node_feature", g_node_feature)
        # print("graph", g)
        label_feature = self.gcn(g, g_node_feature)
        # print("Forward Label Feature: ", label_feature) #[28415,768]
        # mesh_mask = torch.tensor(mesh_mask) # [16,28415] 
        label_attn_mask = label_feature.transpose(0, 1) * mesh_mask 
        # print("Forward Label Attention: ", label_attn_mask)

        # print("mesh_mask: ", type(mesh_mask), mesh_mask.shape)
        # print("Forward Label Attn: ", type(label_attn_mask), label_attn_mask.size()) # [16, 768, 28415]
        # print("Forward Input Id: ", type(input_ids), input_ids.size()) # [16, 512]
        
        #Take the word tokens to have the sequence length
        outputs = self.bert(input_ids=input_ids,attention_mask=attn_mask)
        bert_output = outputs.last_hidden_state #[16,768] #[16, 512, 768]
        # print("Pooler output: ", type(output), output.size()) 
        # print("Bert output: ", output)
        #CNN output should be: (bs, embed_dim*2, seq_len-ksz+1)
        cnn_output = self.cnn(bert_output.permute(0,2,1)) # [1,768,1] # [16, 768, 500]
        # print("CNN output: ", type(output), output.size()) 
        # print("CNN output: ", output)

        # # Apply the linear projection to the input tensor
        # projected_tensor = self.classifier(output.transpose(1, 2)).transpose(1, 2)

        # # Apply mean-pooling along the token dimension
        # pooled_tensor = torch.mean(projected_tensor, dim=2)

        # print("Output tensor: ", pooled_tensor.size())

        # apply non-linear activation
        # output = nn.functional.relu(output)
        # print("Relu output: ", type(output), output.size()) # [1, 28415]

        # apply label attention mask
        # output_masked = output.unsqueeze(0) * label_attn_mask.float().mean(dim=0, keepdim=True) # [1, 768, 28415]
        # [16, 768, 500] * [768, 28415]

        # dense_output = torch.matmul(bert_output, self.dense_layer.weight.t()) + self.dense_layer.bias  # dense layer
        # dense_output = dense_output.permute(0,2,1)
        # print("dense_output: ", dense_output.size()) # torch.Size([16, 512, 28415])
        # label_attn_mask = label_attn_mask.transpose(1, 2)  # transpose to shape [16, 28415, 768]
        # print("label_attn_mask: ", label_attn_mask.size()) 
        # alpha = torch.softmax(torch.matmul(dense_output, label_attn_mask), dim=2)
        
        alpha = torch.softmax(torch.matmul(cnn_output.transpose(1, 2), label_attn_mask.float()), dim=1)
        # print("alpha output: ", type(alpha), alpha.size()) # [16, 500, 28415] [16,512,768]
        # print("Forward Alpha output: ", alpha)
        # print("Test 1: ", type(output), type(alpha))
        output_features = torch.matmul(cnn_output, alpha).transpose(1, 2) # [16, 28415, 768]
        # print("output_features", output_features.size()) # torch.Size([16, 768, 28415])
        # print("Forward Output features: ", output_features)
        x_feature = torch.sum(output_features * label_feature, dim=2) # [16, 28415]
        # print("x_feature: ", x_feature.size())
        # print("Forward X feature: ", x_feature)
        # alpha_text = torch.softmax(torch.matmul(output, label_attn_mask), dim=0)

        # apply sigmoid activation to get predicted probabilities
        # probs = torch.sigmoid(x_feature)

        # print("Probs classifier bert: ", type(probs), probs.size(), probs)
        # print("x_feature classifier bert: ", type(x_feature), x_feature)

        torch.cuda.empty_cache()
            
        return x_feature

    def training_step(self,batch,batch_idx):
        # print("training_step")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        mesh_mask = batch['mesh_mask']
        
        probs = self(input_ids,attention_mask, mesh_mask) # torch.Size([32, 28415]); 32 = batch_size
        # count = 0 
        # count_label = 0
        # for i in range(len(labels)):
        #     for j in range(len(labels[0])):
        #         if labels[i][j] == 1:
        #             count_label += 1
        #             if labels[i][j] == mesh_mask[i][0][j]: 
        #                 count += 1
        # print("Intersection: ", count, count_label, labels.size(), mesh_mask.size())
        # print("labels, meshmask: ", labels.size(), mesh_mask.size())

        # print("Hello Train: ", probs.size(), labels.size())
        # # flatten predicted probabilities
        # probs_flat = probs.mean(dim=1).view(-1, 28415)

        # # flatten target labels
        # targets_flat = labels.float().mean(dim=0).view(-1, 28415)
        loss = self.criterion(probs,labels.float())

        probs = probs.data.cpu().numpy()
        labels = labels.cpu().numpy()
        k = 1
        n_samples, n_labels = labels.shape
        precision = 0.0
        for i in range(n_samples):
            true_labels = labels[i]
            pred_labels = probs[i]
            sorted_indices = np.argsort(pred_labels)[::-1] # sort in descending order
            top_k = sorted_indices[:k]
            correct_labels = np.sum(true_labels[top_k])
            precision += correct_labels / k
        precision /= n_samples

        # print("P@1: ", precision)
    
        
        self.log('train_loss',loss , prog_bar=True,logger=True)
        self.log("P@1", precision , prog_bar=True,logger=True)
        
        return loss


    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'] # Number of labels associated with each document Ex: 12 [16,28415]
        mesh_mask = batch['mesh_mask'] # [1,28415]
        
        probs = self(input_ids,attention_mask, mesh_mask)
        # print("Hello Val: ", probs.size(), labels.size())

        # # flatten predicted probabilities
        # probs_flat = probs.mean(dim=1).view(-1, 28415)
        # print("Hello probs_flat: ", probs_flat.size())

        # # flatten target labels
        # targets_flat = labels.float().mean(dim=0).view(1, 28415)
        # print("Hello targets_flat: ", targets_flat.size())

        loss = self.criterion(probs,labels.float())

        # loss = self.criterion(outputs,labels.float())
        self.log('val_loss',loss , prog_bar=True,logger=True)
        
        # # Calculate the metrics
        # y_true = labels.cpu().numpy()
        # y_pred = probs.cpu().numpy()

        # print("Val Cal")
        # print("Val labels: ", labels, type(labels), labels.shape)
        # print("y_true : ", y_true, type(y_true), len(y_true), y_true.shape)
        # print("y_pred : ", y_pred, type(y_pred), len(y_pred), y_pred.shape)

        # self.evaluate(y_true, y_pred)

        # print("y_pred sigmoid: ", y_pred, type(y_pred), len(y_pred))
        # f1 = f1_score(y_true, y_pred, average='weighted')
        # precision = precision_score(y_true, y_pred, average='weighted')
        # recall = recall_score(y_true, y_pred, average='weighted')

        # # Log the metrics to the output
        # self.log('val_f1', f1, prog_bar=True)
        # self.log('val_precision', precision, prog_bar=True)
        # self.log('val_recall', recall, prog_bar=True)

        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        mesh_mask = batch['mesh_mask']
        
        probs = self(input_ids,attention_mask, mesh_mask)
        # print("Hello test: ", probs.size(), labels.size())

        # # flatten predicted probabilities
        # probs_flat = probs.mean(dim=1).view(-1, 28415)

        # # flatten target labels
        # targets_flat = labels.float().mean(dim=0).view(-1, 28415)
        loss = self.criterion(probs,labels.float())
        self.log('test_loss',loss , prog_bar=True,logger=True)

        # Calculate the metrics
        # y_true = labels.cpu().numpy()
        # y_pred = torch.sigmoid(probs).cpu().numpy() > 0.5
        # y_pred = probs.cpu().numpy()
        # self.evaluate(y_true, y_pred)
        # f1 = f1_score(y_true, y_pred, average='weighted')
        # precision = precision_score(y_true, y_pred, average='weighted')
        # recall = recall_score(y_true, y_pred, average='weighted')

        # Log the metrics to the output
        # self.log('test_f1', f1, prog_bar=True)
        # self.log('test_precision', precision, prog_bar=True)
        # self.log('test_recall', recall, prog_bar=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        # Extract input data and labels from the batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        mesh_mask = batch['mesh_mask']
        
        # Make predictions on the input data
        probs = self(input_ids, attention_mask, mesh_mask)
        
        # Print the shape of the predicted probabilities tensor and the labels tensor
        # print("Probs shape:", probs.shape)
        # print("Labels shape:", labels.shape)

        # Compute the loss for the predictions
        loss = self.criterion(probs, labels.float())

        # Apply sigmoid function to get predicted probabilities
        predictions = torch.sigmoid(probs)

        # Return dictionary with predictions
        return {"test_loss": loss, "predictions": probs, "labels": labels.float()}
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.lr)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)

        return [optimizer], [scheduler]