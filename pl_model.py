# XLMClassification LightningModule
import torch
import numpy as np
from transformers import XLMRobertaForSequenceClassification, AutoModel, get_linear_schedule_with_warmup, BertTokenizer, BertModel, BertForSequenceClassification, XLMRobertaTokenizer
from config import Config
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pl_dataloader import dataset_loader
import logging
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAccuracy
from utils import get_nonzero_cols_n_rows
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional
from pytorch_lightning import LightningModule
import uuid


class PubMedBERT_Optuna(LightningModule):
    def __init__(self, 
                config: Config, 
                num_labels, 
                lr: float = 1e-04, 
                warmup_steps: int = None,
                hidden_dropout: float = None,
                attention_dropout: float = None,
                training_steps: int = None,
                gradient_accumulation_steps: int = None,
                model_id: str = None,
                ):
        super().__init__()
        self.config = config
        self.model_name = config.model.name_or_path
        self.learning_rate = lr or config.train.learning_rate
        self.warmup_steps = warmup_steps or config.train.warmup_steps
        self.hidden_dropout = hidden_dropout or config.train.hidden_dropout
        self.attention_dropout = attention_dropout or config.train.attention_dropout
        self.training_steps = training_steps or config.train.training_steps
        self.num_labels = num_labels
        self.gradient_accumulation_steps = gradient_accumulation_steps or config.train.gradient_accumulation_steps
        self.model_id = model_id or f'{self.model_name}_{self.learning_rate}_{uuid.uuid4().hex}'

        self.l1 = AutoModel.from_pretrained(self.model_name, 
                                                                    num_labels=self.num_labels,
                                                                    output_attentions=False,
                                                                    output_hidden_states=False,
                                                                    hidden_dropout_prob=self.hidden_dropout,
                                                                    attention_probs_dropout_prob=self.attention_dropout,
                                                                    )
        # self.l2 = torch.nn.Linear(768, self.num_labels)

        self.classifier = torch.nn.Linear(self.l1.config.hidden_size, self.num_labels)

        self.average = config.train.average
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.f1 = MultilabelF1Score(num_labels=self.num_labels, average= self.average)
        self.accu = MultilabelAccuracy(num_labels=self.num_labels, average= self.average)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        output = self.l1(input_ids, attention_mask=attention_mask)[0]
        output = self.classifier(output[:, 0, :])
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        y_probs = torch.sigmoid(logits)
        y_pred = (y_probs >= 0.5).float()
        self.validation_step_outputs.append({'y_pred': y_pred, 'y_true': labels, 'y_prob': y_probs, 'loss': loss})
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "y_pred": y_pred, "y_true": labels, 'y_prob': y_probs}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs])
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs])
        y_prob = torch.cat([x['y_prob'] for x in self.validation_step_outputs])

        if self.average in ['macro', 'weighted']:
            remaining, _ = get_nonzero_cols_n_rows(y_true)
            y_pred = y_pred[:,remaining]
            y_true = y_true[:,remaining]
            y_prob = y_prob[:,remaining]
            auroc_func = MultilabelAUROC(num_labels=y_true.shape[1], average=self.average)
        else:
            auroc_func = MultilabelAUROC(num_labels=self.num_labels, average=self.average)

        auroc = auroc_func(y_prob, y_true.long())
        metrics = self.multi_label_metrics(y_pred.cpu().numpy(), y_true.cpu().numpy(), y_prob.cpu().numpy())
        logging.info(f'Validation epoch end, metrics: {metrics["roc_auc"]}, torchmetrics.auroc {auroc}')
      #  self.log('sklearn.auroc', metrics['roc_auc'])
        self.log('torchmetrics.auroc', auroc)
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        y_probs = torch.sigmoid(logits)
        y_pred = (y_probs >= 0.5).float()
        self.test_step_outputs.append({'y_pred': y_pred, 'y_true': labels, 'y_prob': y_probs, 'loss': loss})
        # if batch_idx % 50 == 0:
        #     logging.info(f"Test step {batch_idx}, loss: {loss.item()}")
        return {"loss": loss, "y_pred": y_pred, "y_true": labels, 'y_prob': y_probs}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs])
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs])
        y_prob = torch.cat([x['y_prob'] for x in self.test_step_outputs])
        
        if self.average in ['macro', 'weighted']:
            remaining, _ = get_nonzero_cols_n_rows(y_true)
            y_pred = y_pred[:,remaining]
            y_true = y_true[:,remaining]
            y_prob = y_prob[:,remaining]
            auroc_func = MultilabelAUROC(num_labels=y_true.shape[1], average=self.average)
        else:
            auroc_func = MultilabelAUROC(num_labels=self.num_labels, average=self.average)
            
        auroc = auroc_func(y_prob, y_true.long())
        metrics = self.multi_label_metrics(y_pred.cpu().numpy(), y_true.cpu().numpy(), y_prob.cpu().numpy())
        logging.info(f"Test epoch end, metrics: {metrics}, torch.metrics {auroc}")
        # self.log('f1', metrics['f1'])
        # self.log('roc_auc_{self.average}', metrics['roc_auc'])
        self.log('torchmetrics.auroc', auroc)
        # self.log("test_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=self.warmup_steps,
                                                       num_training_steps=self.training_steps,
                                                    )
        lr_scheduler = {
                    'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                    'interval': 'step',  # The unit of the scheduler's step size
                    'frequency': 1  # The frequency of the scheduler
                    }

        return [optimizer], [lr_scheduler]


    def multi_label_metrics(self, y_pred, y_true, y_prob, threshold=0.5):
        f1_average = f1_score(y_true=y_true, y_pred=y_pred, average=self.average)
        roc_auc = roc_auc_score(y_true, y_prob, average=None)
        roc_auc = np.mean(roc_auc)
        accuracy = accuracy_score(y_true, y_pred)
        return {'f1': f1_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
    





class XLMClassification_Optuna(LightningModule):
    def __init__(self, 
                config: Config, 
                num_labels, 
                lr: float = 1e-04, 
                warmup_steps: int = None,
                hidden_dropout: float = None,
                attention_dropout: float = None,
                training_steps: int = None,
                gradient_accumulation_steps: int = None,
                model_id: str = None,
                ):
        super().__init__()
        self.model_id = model_id or uuid.uuid4().hex
        self.config = config
        self.model_name = config.model.name_or_path
        self.learning_rate = lr or config.train.learning_rate
        self.warmup_steps = warmup_steps or config.train.warmup_steps
        self.hidden_dropout = hidden_dropout or config.train.hidden_dropout
        self.attention_dropout = attention_dropout or config.train.attention_dropout
        self.training_steps = training_steps or config.train.training_steps
        self.num_labels = num_labels
        self.gradient_accumulation_steps = gradient_accumulation_steps or config.train.gradient_accumulation_steps

        self.l1 = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, 
                                                                    num_labels=self.num_labels,
                                                                    output_attentions=False,
                                                                    output_hidden_states=False,
                                                                    hidden_dropout_prob=self.hidden_dropout,
                                                                    attention_probs_dropout_prob=self.attention_dropout,
                                                                    )
        self.average = config.train.average
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.f1 = MultilabelF1Score(num_labels=self.num_labels, average= self.average)
        self.accu = MultilabelAccuracy(num_labels=self.num_labels, average= self.average)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        output = self.l1(input_ids, attention_mask=attention_mask)[0]
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
        # ...

    # def validation_step(self, batch, batch_idx):
    #     input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

    #     logits = self(input_ids, attention_mask)
    #     loss = self.loss(logits, labels)
    #     y_probs = torch.sigmoid(logits)
    #     y_pred = (y_probs >= 0.5).float()

    #     self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
    #     return {"loss": loss, "y_pred": y_pred, "y_true": labels, 'y_prob': y_probs}

    # def validation_epoch_end(self, validation_step_outputs):
    #     avg_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
    #     y_pred = torch.cat([x['y_pred'] for x in validation_step_outputs])
    #     y_true = torch.cat([x['y_true'] for x in validation_step_outputs])
    #     y_prob = torch.cat([x['y_prob'] for x in validation_step_outputs])

    #     # ... you can compute and log additional metrics here using y_pred, y_true, y_prob


    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # input_ids, attention_mask, labels = batch[0], batch[1], batch[2]

        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        y_probs = torch.sigmoid(logits)
        y_pred = (y_probs >= 0.5).float()
        self.validation_step_outputs.append({'y_pred': y_pred, 'y_true': labels, 'y_prob': y_probs, 'loss': loss})
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "y_pred": y_pred, "y_true": labels, 'y_prob': y_probs}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs])
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs])
        y_prob = torch.cat([x['y_prob'] for x in self.validation_step_outputs])

        if self.average in ['macro', 'weighted']:
            remaining, _ = get_nonzero_cols_n_rows(y_true)
            y_pred = y_pred[:,remaining]
            y_true = y_true[:,remaining]
            y_prob = y_prob[:,remaining]
            auroc_func = MultilabelAUROC(num_labels=y_true.shape[1], average=self.average)
        else:
            auroc_func = MultilabelAUROC(num_labels=self.num_labels, average=self.average)

        auroc = auroc_func(y_prob, y_true.long())
        metrics = self.multi_label_metrics(y_pred.cpu().numpy(), y_true.cpu().numpy(), y_prob.cpu().numpy())
        logging.info(f'Validation epoch end, metrics: {metrics["roc_auc"]}, torchmetrics.auroc {auroc}')
      #  self.log('sklearn.auroc', metrics['roc_auc'])
        self.log('torchmetrics.auroc', auroc)
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        # input_ids, attention_mask, labels = batch[0], batch[1], batch[2]
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        y_probs = torch.sigmoid(logits)
        y_pred = (y_probs >= 0.5).float()
        self.test_step_outputs.append({'y_pred': y_pred, 'y_true': labels, 'y_prob': y_probs, 'loss': loss})
        # if batch_idx % 50 == 0:
        #     logging.info(f"Test step {batch_idx}, loss: {loss.item()}")
        return {"loss": loss, "y_pred": y_pred, "y_true": labels, 'y_prob': y_probs}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs])
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs])
        y_prob = torch.cat([x['y_prob'] for x in self.test_step_outputs])
        
        if self.average in ['macro', 'weighted']:
            remaining, _ = get_nonzero_cols_n_rows(y_true)
            y_pred = y_pred[:,remaining]
            y_true = y_true[:,remaining]
            y_prob = y_prob[:,remaining]
            auroc_func = MultilabelAUROC(num_labels=y_true.shape[1], average=self.average)
        else:
            auroc_func = MultilabelAUROC(num_labels=self.num_labels, average=self.average)
            
        auroc = auroc_func(y_prob, y_true.long())
        metrics = self.multi_label_metrics(y_pred.cpu().numpy(), y_true.cpu().numpy(), y_prob.cpu().numpy())
        logging.info(f"Test epoch end, metrics: {metrics}, torch.metrics {auroc}")
        # self.log('f1', metrics['f1'])
        # self.log('roc_auc_{self.average}', metrics['roc_auc'])
        self.log('torchmetrics.auroc', auroc)
        # self.log("test_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=self.warmup_steps,
                                                       num_training_steps=self.training_steps,
                                                    )
        lr_scheduler = {
                    'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                    'interval': 'step',  # The unit of the scheduler's step size
                    'frequency': 1  # The frequency of the scheduler
                    }

        return [optimizer], [lr_scheduler]


    def multi_label_metrics(self, y_pred, y_true, y_prob, threshold=0.5):
        f1_average = f1_score(y_true=y_true, y_pred=y_pred, average=self.average)
        roc_auc = roc_auc_score(y_true, y_prob, average=None)
        roc_auc = np.mean(roc_auc)
        accuracy = accuracy_score(y_true, y_pred)
        return {'f1': f1_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
    






    
class XLMClassification(LightningModule):
    def __init__(self, config: Config, num_labels: int):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model_name = config.model.name_or_path
        self.learning_rate = config.train.learning_rate
        self.weight_decay = config.train.weight_decay
        self.warmup_steps = config.train.warmup_steps
        self.average = config.train.average
        self.num_labels = num_labels
        self.hidden_dropout = config.train.hidden_dropout
        self.attention_dropout = config.train.attention_dropout
        base_model = XLMRobertaForSequenceClassification.from_pretrained(self.model_name,
                                                                         num_labels=num_labels,
                                                                         output_attentions=False,
                                                                        output_hidden_states=False,
                                                                        hidden_dropout_prob=self.hidden_dropout,
                                                                        attention_probs_dropout_prob=self.attention_dropout,
                                                                         )
        
        
        self.l1 = base_model
        # self.l2 = torch.nn.Dropout(config.train.dropout)
        # self.l3 = torch.nn.Linear(768, self.num_labels)
        self.loss = torch.nn.BCEWithLogitsLoss(reduce='mean')
        self.f1 = MultilabelF1Score(num_labels=self.num_labels, average= self.average)
        self.accu = MultilabelAccuracy(num_labels=self.num_labels, average= self.average)

    def forward(self, input_ids, attention_mask):
        output = self.l1(input_ids, attention_mask=attention_mask)[0]
        # output = self.l2(output[:, 0, :])
        # output = self.l3(output)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        self.log('training_loss', loss, on_step=(batch_idx % 50 == 0), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        y_prob = torch.sigmoid(logits)
        y_pred = (y_prob >= 0.5).float()
        self.log('validation_loss', loss, on_step=(batch_idx % 50 == 0), on_epoch=True, prog_bar=True)
        return {"loss": loss, "y_pred": y_pred, "y_true": labels, 'y_prob': y_prob}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        y_true = torch.cat([x["y_true"] for x in outputs])
        y_prob = torch.cat([x['y_prob'] for x in outputs])

        if self.average in ['macro', 'weighted']:
            remaining, _ = get_nonzero_cols_n_rows(y_true)
            y_pred = y_pred[:,remaining]
            y_true = y_true[:,remaining]
            y_prob = y_prob[:,remaining]
            auroc_func = MultilabelAUROC(num_labels=y_true.shape[1], average=self.average)
        else:
            auroc_func = MultilabelAUROC(num_labels=self.num_labels, average=self.average)
        
        
        auroc = auroc_func(y_prob, y_true.long())
        # auroc = np.mean(auroc.cpu().numpy()
        metrics = self.multi_label_metrics(y_pred.cpu().numpy(), y_true.cpu().numpy(), y_prob.cpu().numpy())
        logging.info(f'Validation epoch end, metrics: {metrics}, torchmetrics.auroc {auroc}')
        self.log_dict(metrics)
        print(metrics)
        self.log('torchmetrics.auroc', auroc)
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        logits = self(input_ids, attention_mask)
        loss = self.loss(logits, labels)
        y_probs = torch.sigmoid(logits)
        y_pred = (y_probs >= 0.5).float()
        # if batch_idx % 50 == 0:
        #     logging.info(f"Test step {batch_idx}, loss: {loss.item()}")
        return {"loss": loss, "y_pred": y_pred, "y_true": labels, 'y_prob': y_probs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y_pred = torch.cat([x["y_pred"] for x in outputs])
        y_true = torch.cat([x["y_true"] for x in outputs])
        y_prob = torch.cat([x["y_prob"] for x in outputs])
        
        if self.average in ['macro', 'weighted']:
            remaining, _ = get_nonzero_cols_n_rows(y_true)
            y_pred = y_pred[:,remaining]
            y_true = y_true[:,remaining]
            y_prob = y_prob[:,remaining]
            auroc_func = MultilabelAUROC(num_labels=y_true.shape[1], average=self.average)
        else:
            auroc_func = MultilabelAUROC(num_labels=self.num_labels, average=self.average)
            
        auroc = auroc_func(y_prob, y_true.long()) # torchmetrics 
        auroc = np.mean(auroc)
        metrics = self.multi_label_metrics(y_pred.cpu().numpy(), y_true.cpu().numpy(), y_prob.cpu().numpy()) # sklearn metrics
        logging.info(f"Test epoch end, metrics: {metrics}, torch.metrics {auroc}")
        self.log('f1', metrics['f1'])
        self.log('sklearn.{self.average}', metrics['roc_auc'])
        self.log('torchmetrics.auroc', auroc)
        self.log("test_loss", avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        warmup_steps = self.warmup_steps

        lr_lambda = lambda step: min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)

        return [optimizer], [scheduler]
    
    def multi_label_metrics(self, y_pred, y_true, y_prob, threshold=0.5):
        f1_average = f1_score(y_true=y_true, y_pred=y_pred, average=self.average)
        roc_auc = roc_auc_score(y_true, y_prob, average=self.average) #average=self.average)
        roc_auc = np.mean(roc_auc)
        accuracy = accuracy_score(y_true, y_pred)
        return {'f1': f1_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
    

class SpanishBertBaseline(LightningModule):
    def __init__(self, 
                config,
                num_labels, 
                num_training_steps
                ):

        super().__init__()
        # Define hparams
        self.hparams = {
            "model_name": config.model.name_or_path,
            "hidden_dropout_prob": config.train.hidden_dropout,
            "attention_probs_dropout_prob": config.train.attention_dropout,
            "lr": config.train.learning_rate,
            "batch_size": config.train.batch_size,
            "num_warmup_steps": config.train.warmup_steps,
            "num_labels": num_labels,
            "num_training_steps": num_training_steps,
        }
        self.save_hyperparameters(self.hparams)
        self.num_labels = self.hparams["num_labels"]
        self.num_training_steps = self.hparams["num_training_steps"]
        self.model_name = self.hparams["model_name"]
        self.hidden_dropout_prob = self.hparams["hidden_dropout_prob"]
        self.attention_probs_dropout_prob = self.hparams["attention_probs_dropout_prob"]
        self.spanish_bert = XLMRobertaForSequenceClassification.from_pretrained(self.model_name, 
                                                                                    num_labels=num_labels,
                                                                                    output_attentions=False,
                                                                                    output_hidden_states=False,
                                                                                    hidden_dropout_prob=self.hidden_dropout_prob,
                                                                                    attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                                                                                    )
      
        self.lr = self.hparams["lr"]
        self.batch_size = self.hparams["batch_size"]

        self.dummy_f1_func = f1_score
        self.auc_func = roc_auc_score
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduce="mean")
        self.num_warmup_steps = config.train.warmup_steps

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        logits = self.spanish_bert(input_ids, 
                                token_type_ids=None,
                                   attention_mask=attention_mask)[0]
        return logits

    def training_step(self, batch, batch_idx):
        # b_input_ids = batch['input_ids']
        # b_input_mask = batch['attention_mask']
        # b_labels = batch['labels']
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        
        logits = self.spanish_bert(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)[0]
    
        logits = logits.view(-1, self.num_labels)
        b_labels = b_labels.type_as(logits).view(-1, self.num_labels)

        train_loss = self.loss_func(logits, b_labels)

        self.log('train_loss', train_loss, on_step=False, on_epoch=True)
        return {'loss': train_loss}


    def validation_step(self, batch, batch_idx):
        # b_input_ids = batch['input_ids']
        # b_input_mask = batch['attention_mask']
        # b_labels = batch['labels']
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        logits = self.spanish_bert(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)[0]

        
        pred_logits = logits.view(-1, self.num_labels)
        y_true = b_labels.type_as(logits).view(-1, self.num_labels)
        val_loss = self.loss_func(pred_logits, y_true)

        self.log('step_val_loss', val_loss, on_step=False, on_epoch=True)        

        return {'val_loss': val_loss, 'y_pred': torch.sigmoid(pred_logits), 'y_true': y_true}


    def validation_epoch_end(self, outputs):

        val_loss = torch.tensor([x['val_loss'].item() for x in outputs]).mean()
        y_true = torch.cat([x['y_true'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])

        selected_cols = set([248, 267, 246, 93, 259])
        cols, selected_cols = get_nonzero_cols_n_rows(y_true, 
                                                     selected_cols=selected_cols)
        
        auc_score = self.auc_func(y_true[:, cols].cpu(),
                                  y_pred[:, cols].cpu(),
                                  average= None)
                
        auc_score_macro = self.auc_func(y_true[:, cols].cpu(),
                                  y_pred[:, cols].cpu(),
                                  average= 'macro')

        avg_auc = np.mean(auc_score)

        print(f'auc scores equal: {auc_score_macro == avg_auc}')
        logging.info(f'Validation epoch end - Average AUC macro: {auc_score_macro}')
        logging.info(f'Validation epoch end - Average AUC by mean: {avg_auc}')
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)  # Ensure val_loss is on the progress bar.
        self.log("val_auc", avg_auc, on_step=False, on_epoch=True, prog_bar=True)  # Ensure val_auc is on the progress bar.
        self.log('val_auc_macro', auc_score_macro, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        logits = self.spanish_bert(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)[0]

        
        pred_logits = logits.view(-1, self.num_labels)
        y_true = b_labels.type_as(logits).view(-1, self.num_labels)
        test_loss = self.loss_func(pred_logits, y_true)
        
        return {'test_loss': test_loss, 'y_pred': torch.sigmoid(pred_logits), 'y_true': y_true}


    def test_epoch_end(self, outputs):

        test_loss = torch.tensor([x['test_loss'].item() for x in outputs]).mean()
        y_true = torch.cat([x['y_true'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        
        selected_cols = set([248, 267, 246, 93, 259])
        cols, selected_cols = get_nonzero_cols_n_rows(y_true, 
                                                     selected_cols=selected_cols)
        
        auc_score = self.auc_func(y_true[:, cols].cpu(),
                                  y_pred[:, cols].cpu(),
                                  average= None)

        
        micro_auc = self.auc_func(y_true[:, cols].cpu(),
                                  y_pred[:, cols].cpu(),
                                  average='micro')
        avg_auc = np.mean(auc_score)

        results = {"eval_auc": auc_score, 
                "eval_val_auc": avg_auc,
                'eval_test_loss': test_loss,
                'eval_samples_per_label': y_true.sum(axis=0),
                'eval_cols': cols,
                'eval_micro_auc': micro_auc, 
                'eval_y_true': y_true, 
                'eval_y_pred': y_pred,
                }
                
        self.test_results = results 
        self.log(results)
        logging.info(f'Test epoch end - Average AUC: {avg_auc}')
    
        return results

   

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=self.num_warmup_steps,
                                                       num_training_steps=self.num_training_steps,
                                                    )
        lr_scheduler = {
                    'scheduler': lr_scheduler,  # The LR scheduler instance (required)
                    'interval': 'step',  # The unit of the scheduler's step size
                    'frequency': 1  # The frequency of the scheduler
                    }
                    
        return [optimizer], [lr_scheduler]
       
    
if __name__ == '__main__':
    config = Config('/pvc/ContinualClinicalNLU/pipeline_test/config.yml') 
    dataset = dataset_loader(config)
    num_labels = dataset.num_labels
    model = SpanishBertBaseline(config=config, num_labels=num_labels)
    print(model)