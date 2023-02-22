import torch
import pytorch_lightning as pl
import torch.nn as nn
import transformers
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW 


BERT_MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

class KenmeshClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self, n_classes=10,steps_per_epoch=None,n_epochs=3, lr=2e-5):
        super().__init__()

        self.bert=BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        embedding_dim = self.bert.config.hidden_size
        self.cnn = nn.Sequential( # [1, 768, 16]
                nn.Conv1d(in_channels=embedding_dim, out_channels=768, kernel_size=3, dilation=1),
                nn.ReLU(), # [1,128,7]
                nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, dilation=2),
                nn.ReLU(), # [1,256,5]
                nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, dilation=4),
                nn.ReLU(), # [1,512,1]
                # nn.Conv1d(in_channels=768, out_channels=768, kernel_size=1, dilation=8),
                # nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            )

        self.classifier=nn.Linear(self.bert.config.hidden_size,n_classes) 
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,input_ids, attn_mask, label_attn_mask):
        # print("Forward Label Attn: ", type(label_attn_mask), label_attn_mask.size()) # [16, 768, 28415]
        # print("Forward Input Id: ", type(input_ids), input_ids.size()) # [16, 512]
        
        outputs = self.bert(input_ids=input_ids,attention_mask=attn_mask)
        output = outputs.pooler_output #[16,768]
        print("Pooler output: ", type(output), output.size()) 

        output = self.cnn(output.unsqueeze(0).permute(0,2,1)) # [1,768,1]
        print("CNN output: ", type(output), output.size(), output.view(1,-1).size()) 

        output = self.classifier(output.view(1,-1)) # [1, 28415]
        print("Classifier output: ", type(output), output.size()) 

        # apply non-linear activation
        output = nn.functional.relu(output)
        print("Relu output: ", type(output), output.size()) # [1, 28415]

        # apply label attention mask
        output_masked = output.unsqueeze(0) * label_attn_mask.float().mean(dim=0, keepdim=True) # [1, 768, 28415]
        print("output_masked: ", type(output_masked), output_masked.size()) 

        # alpha_text = torch.softmax(torch.matmul(output, label_attn_mask), dim=0)

        # apply sigmoid activation to get predicted probabilities
        probs = torch.sigmoid(output_masked)

        print("Probs: ", type(probs), probs.size())
            
        return probs

    def training_step(self,batch,batch_idx):
        print("training_step")
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        mesh_mask = batch['mesh_mask'] 
        label_attn_mask = batch['label_attn_mask'] 
        
        probs = self(input_ids,attention_mask, label_attn_mask) # torch.Size([32, 28415]); 32 = batch_size
        
        print("Hello Train: ", probs.size(), labels.size())
        # flatten predicted probabilities
        probs_flat = probs.mean(dim=1).view(-1, 28415)

        # flatten target labels
        targets_flat = labels.float().mean(dim=0).view(-1, 28415)
        loss = self.criterion(probs_flat,targets_flat)
        
        self.log('train_loss',loss , prog_bar=True,logger=True)
        
        return {"loss" :loss, "predictions":probs, "labels": labels }


    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label'] # Number of labels associated with each document Ex: 12 [16,28415]
        mesh_mask = batch['mesh_mask'] # [1,28415]
        label_attn_mask = batch['label_attn_mask'] 
        
        probs = self(input_ids,attention_mask, label_attn_mask)
        print("Hello Val: ", probs.size(), labels.size())

        # flatten predicted probabilities
        probs_flat = probs.mean(dim=1).view(-1, 28415)
        print("Hello probs_flat: ", probs_flat.size())

        # flatten target labels
        targets_flat = labels.float().mean(dim=0).view(1, 28415)
        print("Hello targets_flat: ", targets_flat.size())

        loss = self.criterion(probs_flat,targets_flat)

        # loss = self.criterion(outputs,labels.float())
        self.log('val_loss',loss , prog_bar=True,logger=True)
        
        return loss

    def test_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        mesh_mask = batch['mesh_mask'] 
        label_attn_mask = batch['label_attn_mask'] 
        
        probs = self(input_ids,attention_mask, label_attn_mask)
        print("Hello test: ", probs.size(), labels.size())

        # flatten predicted probabilities
        probs_flat = probs.mean(dim=1).view(-1, 28415)

        # flatten target labels
        targets_flat = labels.float().mean(dim=0).view(-1, 28415)
        loss = self.criterion(probs_flat,targets_flat)
        self.log('test_loss',loss , prog_bar=True,logger=True)
        
        return loss
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters() , lr=self.lr)
        warmup_steps = self.steps_per_epoch//3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)

        return [optimizer], [scheduler]
