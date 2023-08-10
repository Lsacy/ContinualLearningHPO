import pandas as pd
import ast
import json
from psutil import test
from torch.utils import data
from transformers import BertTokenizerFast as fast_tokenizer
from transformers import AutoTokenizer
import torch 
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
from transformers.utils.dummy_pt_objects import TransfoXLLMHeadModel
sys.path.append('/pvc/')
from pytorch_lightning.utilities.seed import seed_everything


def load_codiesp(data_paths, eval_dataset):
        test_data = pd.DataFrame()

        #assert task in ["diagnosis", "procedure"] and language in ['english', 'spanish']
        if not data_paths['translator_data_selector']:

            train_data = pd.read_csv(data_paths[f"train_data_path_{eval_dataset}"]).rename(columns={'ICD10':'label', 'TEXT': 'notes', })
            train_data.labels = train_data.labels.apply(lambda row: ast.literal_eval(row))
            dev_data = pd.read_csv(data_paths[f"validation_data_path_{eval_dataset}"]).rename(columns={'ICD10':'label', 'TEXT': 'notes'})
            dev_data.labels = dev_data.labels.apply(lambda row: ast.literal_eval(row))
            try:
                test_data = pd.read_csv(data_paths[f"test_data_path_{eval_dataset}"]).rename(columns={'ICD10':'label', 'TEXT': 'notes'})
                test_data.labels = test_data.labels.apply(lambda row: ast.literal_eval(row))
            except: 
                print("test_data_cutoff is not splitted in create_dataset.py because no test set is used")

        elif data_paths['translator_data_selector'] in ['official_translation', 'Opus_el_en']: 

            train_data = pd.read_csv(data_paths[f"train_data_path_{eval_dataset}"])
            train_data = train_data.rename(columns={'notes': 'org_notes'})
            train_data = train_data.rename(columns={'ICD10':'label', 'TEXT': 'notes', data_paths['translator_data_selector']: "notes" })
            train_data.labels = train_data.labels.apply(lambda row: ast.literal_eval(row))
            dev_data = pd.read_csv(data_paths[f"validation_data_path_{eval_dataset}"])
            dev_data = dev_data.rename(columns={'notes': 'org_notes'})
            dev_data = dev_data.rename(columns={'ICD10':'label', 'TEXT': 'notes', data_paths['translator_data_selector']: "notes"})
            dev_data.labels = dev_data.labels.apply(lambda row: ast.literal_eval(row))
            try:
                test_data = pd.read_csv(data_paths[f"test_data_path_{eval_dataset}"])
                test_data = test_data.rename(columns={'notes': 'org_notes'})
                test_data = test_data.rename(columns={'ICD10':'label', 'TEXT': 'notes', data_paths['translator_data_selector']: "notes"})
                test_data.labels = test_data.labels.apply(lambda row: ast.literal_eval(row))
            except: 
                print("test_data_cutoff is not splitted in create_dataset.py because no test set is used")
        
        with open(data_paths['all_labels_path'], 'rb') as f: 
            labels = pickle.load(f)
    
        train_data = train_data.loc[train_data.labels.apply(len) > 0]
        dev_data = dev_data.loc[dev_data.labels.apply(len) > 0] 
        test_data = test_data.loc[test_data.labels.apply(len) > 0] 

        return train_data, dev_data, test_data, labels


def get_data(model_name,
            data_paths, 
            language,
            eval_dataset,
            batch_size,
            task, 
            do_eval=False
            ):
            
    seed_val = 42
    import os 
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    train_data, dev_data, test_data, labels = load_codiesp(data_paths, eval_dataset)
    #TODO
    
    #train_data, dev_data, test_data, labels = load_CodieSp_datav2(task, language)
    # to create dev and test to have always batch of 8!!! Elsewise AUC does not work.
    #train_data, dev_data, test_data = rearange_datasets(train_data, dev_data, test_data)
    #train_data, dev_data, test_data, labels = get_datav2(data_paths, train_lng=language)
    
    #DUMMY
    #train_data=train_data.iloc[:100]
    #dev_data = dev_data.iloc[:100]
    
    logging.warning(f'train: {len(train_data)}, "dev:{len(dev_data)}, test:{len(test_data)}, labels:{len(labels)}')
    ############################# load tokenizer ####################################
    if model_name != 'xlm-roberta-base' and 'BSC' not in model_name:
        tokenizer = fast_tokenizer.from_pretrained(model_name)
    else:
        tokenizer =  AutoTokenizer.from_pretrained(model_name)
    ############################# map icd codes to array positios and reverse ####################################
    label_to_pos, pos_to_label = label_to_pos_map(labels)
    ############################# preprocess data for BERT ####################################
    train_dataset = preprocessing_for_bert(train_data, label_to_pos, tokenizer=tokenizer, language=language)
    dev_dataset = preprocessing_for_bert(dev_data, label_to_pos, tokenizer=tokenizer, language=language)
    
    if not do_eval:
        codie_cond = task =='zero_shot_diag_ccs' and eval_dataset == 'codie'
        achepa_cond = task == 'zero_shot_diag_achepa' and eval_dataset == 'achepa'
                                                    
        if codie_cond or achepa_cond: 
            ccs = load_zero_shot_ccs_codes(data_paths)
            selected_labels = get_zero_shot_ccs_idx(label_to_pos, ccs)
            train_dataset =  manipulate_zero_shot_diagnosis(train_dataset, selected_labels)
            dev_dataset = manipulate_zero_shot_diagnosis(train_dataset, selected_labels)

        elif task == 'zero_shot_diag_css' or task == 'zero_shot_diag_achepa': 
                logger.warning('zero_shot_diag is only valid for codiesp for ccs and for achepa for achepa diagnoses')
                raise

    try:
        test_dataset = preprocessing_for_bert(test_data, label_to_pos, tokenizer=tokenizer, language=language)
        logging.warning(f'train: {len(train_dataset)}, "dev:{len(dev_dataset)}, test:{len(test_dataset)}, labels:{len(labels)}')
    except: 
        logging.warning("due to no test dataset use it is not split and non existent")

    ############################# create iterator for the datasets ####################################
    train_dataloader = DataLoader(train_dataset,  # The training samples.
                                batch_size=batch_size, # Trains with this batch size.
                                shuffle=True,
                                pin_memory=True,
                                num_workers=4, 
                                persistent_workers=False,
                                )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(dev_dataset, # The validation samples.
                                    batch_size = batch_size, # Evaluate with this batch size.
                                    shuffle = False,
                                    pin_memory=True,
                                    num_workers=4, 
                                    persistent_workers=False,
                                    )
    try:
        test_dataloader = DataLoader(test_dataset, # The validation samples.
                                    batch_size=batch_size, # Evaluate with this batch size.
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=4, 
                                    persistent_workers=False,
                                    )
        logging.warning(f'{len(train_dataloader)}, {len(validation_dataloader)}, {len(test_dataloader)}')
    except: 
        logging.warning("due to no test dataset use it is not split and non existent")
        test_dataloader = None
        
    return train_dataloader, validation_dataloader, test_dataloader, labels
