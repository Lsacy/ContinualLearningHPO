from pl_model import SpanishBertBaseline
from config import Config
import torch
from torch.utils.data import DataLoader
import pandas as pd
import logging
from transformers import AutoTokenizer


from pl_dataloader import dataset_loader, MyDataset, dataset_loader_stratified
from pl_model import XLMClassification, SpanishBertBaseline
from typing import List
import gc
from datetime import datetime
from utils import current_time, get_data,set_seeds
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(data_paths, language, config, task):
    print(f'torch cuda available: {torch.cuda.is_available()}')
    print(f'torch cuda device count: {torch.cuda.device_count()}')

    set_seeds(seed=42)
    average = config.train.average
    print(f'average: {average}')
    train_dataloader, validation_dataloader, _, labels = get_data('xlm-roberta-base',
                                                                        data_paths, 
                                                                        language,
                                                                        data_paths['eval_dataset'],
                                                                        config.train.batch_size, 
                                                                        task)
    num_labels = len(labels)

    
    
    # dataset = dataset_loader(config)
    # num_labels = dataset.num_labels
    # params_train = {'batch_size' : int(config.train.batch_size),
    #           'shuffle': config.train.shuffle,
    #           'num_workers':int(config.train.num_workers),
    #           'pin_memory': config.train.pin_memory}
    # params_val_test = {'batch_size' : int(config.train.batch_size),
    #           'shuffle': config.dev.shuffle,
    #           'num_workers':int(config.train.num_workers),
    #           'pin_memory': config.train.pin_memory}
    # tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
    # dataset_train = MyDataset(dataset, f'mimic_codiesp_train', tokenizer)
    # dataset_val = MyDataset(dataset, f'mimic_codiesp_dev', tokenizer)
    # train = DataLoader(dataset_train, **params_train)
    # val = DataLoader(dataset_val, **params_val_test)
    # num_training_steps = len(train) * config.train.epochs_per_lang

    num_training_steps = 88550
    print(num_training_steps)

    model_spanish = SpanishBertBaseline(config, num_labels, num_training_steps=num_training_steps)
    # model_xlm = XLMClassification(config, num_labels)

    checkpoint_callback = ModelCheckpoint(
                monitor=config.train.validation_metric, 

                mode='max' if config.train.validation_metric == 'val_auc' else 'min', 
                dirpath=config.model.saved_path,
                save_top_k=1,
                filename=f"best_model_mimic", # The file name format for saving the best model
            )

    early_stop_callback = EarlyStopping(monitor=config.train.validation_metric, 
                                                patience=5, 
                                                mode='max' if config.train.validation_metric  == 'val_auc' else 'min')
    
    logger = TensorBoardLogger(save_dir=config.train.tensorboard_path, name=f'logs_{current_time()}')

    trainer = Trainer(max_epochs=config.train.epochs_per_lang, accumulate_grad_batches=8,
                              logger=logger, gpus=1, num_sanity_val_steps=1,
                              callbacks=[checkpoint_callback, early_stop_callback],)

    with torch.cuda.amp.autocast():
        # trainer.fit(model, train_dataloader, validation_dataloader)
        trainer.fit(model_spanish, train_dataloader, validation_dataloader)

if __name__ == '__main__':
    language = 'clinical_spanish_V3'
    eval_dataset = 'mimic'
    filter_set_name = 'ccs_codie'
    config = Config('/pvc/ContinualClinicalNLU/pipeline_test/config.yml')
    task = 'diagnosis'
    translator_data_selector = None


    data_paths = {'train_data_path_mimic': f"/pvc/data/continualLearning/mimic_codiesp_filtered_CCS_fold_1_train.csv",
                'validation_data_path_mimic': f"/pvc/data/continualLearning/mimic_codiesp_filtered_CCS_fold_1_dev.csv",
                'test_data_path_mimic': f"/pvc/data/continualLearning/mimic_codiesp_filtered_CCS_fold_1_test.csv",

                'train_data_path_achepa': f"/pvc/data/continualLearning/achepa_codiesp_filtered_CCS_fold_1_train.csv",
                'validation_data_path_achepa': f"/pvc/data/continualLearning/achepa_codiesp_filtered_CCS_fold_1_dev.csv",
                'test_data_path_achepa': f"/pvc/data/continualLearning/achepa_codiesp_filtered_CCS_fold_1_test.csv",

                'train_data_path_codie': f"/pvc/data/continualLearning/codiesp_CCS_fold_1_train.csv",
                'validation_data_path_codie': f"/pvc/data/continualLearning/codiesp_CCS_fold_1_dev.csv",
                'test_data_path_codie': f"/pvc/data/continualLearning/codiesp_CCS_fold_1_test.csv",

                'all_labels_path': f"/pvc/data/continualLearning/{filter_set_name}_labels.pcl",
                'eval_dataset': eval_dataset,
                'translator_data_selector': translator_data_selector,
                }


    main(data_paths, language, config, task)
