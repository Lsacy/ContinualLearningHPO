from datetime import datetime
import logging
from torch.utils.data import DataLoader
from pathlib import Path

def get_nonzero_cols_n_rows(y_true, selected_cols=set([248, 267, 246, 93, 259])):
    # sum y_true column-wise, check_for_one_class.shape = torch.Size([334]) = num_labels
    check_for_one_class = y_true.sum(axis=0)

    # select only those columns with at least 1 entry and not all entries (0, len(y_true))
    not_one_class_present = [idx for idx in range(len(check_for_one_class)) 
                             if check_for_one_class[idx] not in (0, len(y_true))]
    not_one_class_present_selected = [not_one_class_present.index(idx) for idx in selected_cols 
                                      if idx in not_one_class_present]
    return not_one_class_present, not_one_class_present_selected

def current_time():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def save_csv(results_df, config):
    experiment_name = config.experiment_name
    path = Path(config.paths.log_path) / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    sequence = sequence_keys(config.train.sequence, config)
    sequence = '_'.join(sequence)
    file_name = path / f"results_{config.train.sequence}_{current_time()}.csv"
    results_df.to_csv(file_name, index=False)
    print(results_df)
    return file_name
    # logging.info(f'train_size: {config.dataset.train_size}')

def sequence_keys(sequence, config):
    revert_sequence = {v: k for k, v in config.train.sequence_mapping.items()}
    return [revert_sequence[item] for item in sequence]

class Loggers:
    def __init__(self, tensorboard_logger, custom_logger):
        self.tensorboard = tensorboard_logger
        self.custom_logger = custom_logger


class CustomLogger:
    def __init__(self, config):
        self.config = config
        self.experiment_name = config.experiment_name
        self.model = config.model.base_model
        self.path = config.paths.log_path
        self.sequence = config.train.sequence
        # revert convert self.sequence, which is a list of items, to mapping from config.train.sequence_mapping
        self.sequence = sequence_keys(self.sequence, config)
        self.sequence = '_'.join(self.sequence)
        self.log_path = Path(config.paths.log_path)
        self.log_file_name = self.log_path / self.experiment_name / f"{self.sequence}_{current_time()}.log"
        self._setup_logger()
        
    def _setup_logger(self):
        # Make sure the directory for the log file exists
        log_dir = Path(self.log_file_name).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=self.log_file_name, 
                            level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        self._log_experiment_details()

    def _log_experiment_details(self):
        logging.info(f"Experiment name: {self.experiment_name}")
        logging.info(f"Base Model name: {self.config.model.base_model}")
        for i in self.config.train.sequence_mapping:
            logging.info(f"Sequence mapping: {i}: {self.config.train.sequence_mapping[i]}")

    def found_existing_model(self, paths):
        """Log the existing model(s) found at the given path(s)"""
        logging.info(f"Found existing model(s) at: {paths}")
        logging.info(f'using: (self.model_name_or_path): {paths[0]}')
        
    def log_model(self, model, trial, language):
        logging.info(f"Model name: {self.config.model.name_or_path}")
        logging.info(f'Running trial {trial.number} of language {language} with following hyperparameters:')
        logging.info(f'model_id: {model.model_id}, lr: {trial.params["lr"]}, \
                        warmup_steps: {trial.params["warmup_steps"]}, hidden_dropout: {trial.params["hidden_dropout"]}')
        
    def final_print(self, csv):
        print(f"Experiment name: {self.experiment_name}")
        print(f"Basic Model name: {self.config.model.base_model}")
        print(f"Final results saved at: {csv}")
        print(f"log file saved at: {self.log_file_name}")


    def size_info(self, split, new_size, sizes):
        split = f'{split}_size'
        split_size_config = getattr(self.config.dataset, split)
        logging.info(f'{split} from config - {split_size_config}')
        logging.info(f'actual {split} sizes - {sizes}')
        logging.info(f'new {split} size - {new_size}')


def log_model(model, model_name_or_path, trial, language, sequence):
    logging.info(f"Model name: {model_name_or_path}")
    logging.info(f'Running trial {trial.number} of language {language} with following hyperparameters:')
    logging.info(f'model_id: {model.model_id}, lr: {trial.params["lr"]}, \
                    warmup_steps: {trial.params["warmup_steps"]}, hidden_dropout: {trial.params["hidden_dropout"]}')








## rest of the utils are from the Michas repo

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
import pickle
from pytorch_lightning import seed_everything


def preprocessing_for_bert(data, label_to_pos, tokenizer, language):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attentimport time
    import datetimeion_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    #if len(data.notes.shape) == 2 :
    #    data.columns = ['patient_id', 'label', 'notes', 'translated_notes', 'official_translation_notes']
    sent = data.notes.tolist()
    #for sent in data.notes.tolist():
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs

        #use batch encode plus to process batch by batch and not per sentence
        #pytorch dataset class allows different lenghts of tokens
    encoded_sent = tokenizer.batch_encode_plus(
                                        batch_text_or_text_pairs=sent,  # Preprocess sentence
                                        add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                                        max_length=512,                  # Max length to truncate/pad
                                        pad_to_max_length=True,         # Pad sentence to max length
                                        #return_tensors='pt',           # Return PyTorch tensor
                                        return_attention_mask=True,      # Return attention mask
                                        #padding='longest', 
                                        truncation=True,
                                        return_token_type_ids=False,
                                        return_tensors='pt')
    
    # Add the outputs to the lists
    input_ids = encoded_sent.get('input_ids')
    attention_masks = encoded_sent.get('attention_mask')

    # Convert lists to tensors
    #input_ids = torch.tensor(input_ids)
    #attention_masks = torch.tensor(attention_masks)
    label_tensor = label_to_tensor(data, label_to_pos)
    #label_tensor = label_to_index(data, label_to_pos)
    dataset = TensorDataset(input_ids, attention_masks, label_tensor)
    return dataset

def label_to_index(data, label_to_pos):
    def row_to_pos(row):
        tmp = list()
        for i, code in enumerate(row): 
            tmp.append(label_to_pos[code])
        return tmp 
    return torch.tensor(data['labels'].apply(row_to_pos))

def label_to_tensor(data, label_to_pos):
    tmp = np.zeros((len(data), 
                    len(label_to_pos))
                )
    c = 0
    test_me = list()
    for idx, row in data.iterrows():
        for code in row['labels']:
            try:
                tmp[c, label_to_pos[code]] = 1
                test_me.append(code)
            except: 
                #print('WARNING Number of labels you are not using the english_spanish label filter')
                pass
        c += 1
    return torch.tensor(tmp)

def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)

def load_CodieSp_data(task, language, is_cutoff=True, is_cluster=True,):
    if is_cluster:
        prefix = 'pvc'
    else: 
        prefix = 'tmp'

    test_data = pd.DataFrame()

    assert task in ["diagnosis", "procedure"] and language in ['english', 'spanish']

    path = f'/{prefix}/codiesp/xl-training-data/v4/{task}_task/{language}_clinical_text'
    
    if is_cutoff: 
        train_data = pd.read_csv(f'/{path}/train_data_cutoff.csv')
        train_data.labels = train_data.labels.apply(lambda row: ast.literal_eval(row))
        dev_data = pd.read_csv(f'/{path}/dev_data_cutoff.csv')
        dev_data.label = dev_data.label.apply(lambda row: ast.literal_eval(row))
        try:
            test_data = pd.read_csv(f'/{path}/test_data_cutoff.csv')
            test_data.label = test_data.label.apply(lambda row: ast.literal_eval(row))
        except: 
            print("test_data_cutoff is not splitted in create_dataset.py because no test set is used")

        with open(f'/{path}/all_codes_cutoff.json'.format(path)) as f: 
            labels = json.load(f)['all_D_codes']
    else:
        train_data = pd.read_csv('/{}/train_data.csv'.format(path))
        train_data.label = train_data.label.apply(lambda row: ast.literal_eval(row))
        dev_data = pd.read_csv('/{}/dev_data.csv'.format(path))
        dev_data.label = dev_data.label.apply(lambda row: ast.literal_eval(row))
        try:
            test_data = pd.read_csv('/{}/test_data.csv'.format(path))
            test_data.label = test_data.label.apply(lambda row: ast.literal_eval(row))
            test_data = test_data.loc[test_data.label.apply(len) > 0] 
        except: 
            print("test_data is not splitted in create_dataset.py because no test set is used")

        with open('/{}/all_codes.json'.format(path)) as f: 
            labels = json.load(f)['all_D_codes']
    
    train_data = train_data.loc[train_data.label.apply(len) > 0]
    dev_data = dev_data.loc[dev_data.label.apply(len) > 0] 

    return train_data, dev_data, test_data, labels

def load_CodieSp_datav2(task, language, is_cutoff=True, is_cluster=True,):
    if is_cluster:
        prefix = 'pvc'
    else: 
        prefix = 'tmp'

    test_data = pd.DataFrame()

    assert task in ["diagnosis", "procedure"] and language in ['english', 'spanish']

    path = f'/{prefix}/codiesp/xl-training-data/v4/{task}_task/rebalanced/{language}_clinical_text'
    
    if is_cutoff: 
        train_data = pd.read_csv(f'/{path}/{language}_train_data_cutoff.csv')
        train_data.label = train_data.label.apply(lambda row: ast.literal_eval(row))
        dev_data = pd.read_csv(f'/{path}/{language}_dev_data_cutoff.csv')
        dev_data.label = dev_data.label.apply(lambda row: ast.literal_eval(row))
        try:
            test_data = pd.read_csv(f'/{path}/{language}_test_data_cutoff.csv')
            test_data.label = test_data.label.apply(lambda row: ast.literal_eval(row))
        except: 
            print("test_data_cutoff is not splitted in create_dataset.py because no test set is used")

        with open(f'/{path}/all_codes_cutoff.json'.format(path)) as f: 
            labels = json.load(f)['all_D_codes']
    else:
        train_data = pd.read_csv('/{}/train_data.csv'.format(path))
        train_data.labels = train_data.labels.apply(lambda row: ast.literal_eval(row))
        dev_data = pd.read_csv('/{}/dev_data.csv'.format(path))
        dev_data.labels = dev_data.labels.apply(lambda row: ast.literal_eval(row))
        try:
            test_data = pd.read_csv('/{}/test_data.csv'.format(path))
            test_data.labels = test_data.labels.apply(lambda row: ast.literal_eval(row))
            test_data = test_data.loc[test_data.labels.apply(len) > 0] 
        except: 
            print("test_data is not splitted in create_dataset.py because no test set is used")

        with open('/{}/all_codes.json'.format(path)) as f: 
            labels = json.load(f)['all_D_codes']
    
    train_data = train_data.loc[train_data.labels.apply(len) > 0]
    dev_data = dev_data.loc[dev_data.labels.apply(len) > 0] 
    test_data = test_data.loc[test_data.labels.apply(len) > 0] 

    return train_data, dev_data, test_data, labels


def label_to_pos_map(all_codes):
        label_to_pos = dict([(code,pos) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
        pos_to_label = dict([(pos,code) for code, pos in zip(sorted(all_codes),range(len(all_codes)))])
        return label_to_pos, pos_to_label


def filter_mimic_text(notes_df) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "BRIEF_HOSPITAL_COURSE": 'brief hospital course'
        #"MEDICATION_ADM": "medications on admission:",
        #"ALLERGIES": "allergies:",
        #"PHYSICAL_EXAM": "physical exam:",
        #"FAMILY_HISTORY": "family history:",
        #"SOCIAL_HISTORY": "social history:"
    }

    # replace linebreak indicators
    notes_df['TEXT'] = notes_df['TEXT'].str.replace(r"\n", r"\\n")

    # extract each section by regex
    for key in admission_sections.keys():
        section = admission_sections[key]
        notes_df[key] = notes_df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                  .format(section))

        notes_df[key] = notes_df[key].str.replace(r'\\n', r' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df[notes_df[key].str.startswith("[]")][key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "") | (notes_df.BRIEF_HOSPITAL_COURSE != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "HOSPITAL_COURSE: " + notes_df.BRIEF_HOSPITAL_COURSE.astype(str)
                                    #"MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    #+ '\n\n' +
                                    #"ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    #+ '\n\n' +
                                    #"PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    #+ '\n\n' +
                                    #"FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    #+ '\n\n' +
                                    #"SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str)
                                )
    return notes_df


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

def rearange_datasets(train_data, dev_data, test_data): 
    dev_len = len(dev_data)
    test_len = len(test_data)


    if (dev_len%8) != 0 : 
        new_size = [dev_len + i for i in range(10) if (dev_len + i)%8 == 0][0]
        dev_samples = new_size - dev_len
        dev_conc = train_data.sample(n=dev_samples, random_state=42)
        idx = dev_conc.index
        train_data = train_data.drop(idx)
        dev_data = pd.concat([dev_data, dev_conc])

    if (test_len%8) != 0 : 
        new_size = [test_len + i for i in range(10) if (test_len + i)%8 == 0][0]
        test_samples = new_size - test_len
        test_conc = train_data.sample(n=test_samples, random_state=42)
        idx = test_conc.index
        train_data = train_data.drop(idx)
        test_data = pd.concat([test_data, test_conc])
    
    return train_data, dev_data, test_data
    
def manipulate_zero_shot_diagnosis(dataset, idx): 
        labels = dataset.tensors[2]
        labels[:, idx] = 0
        return TensorDataset(dataset.tensors[0], dataset.tensors[1], labels)

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
                logging.warning('zero_shot_diag is only valid for codiesp for ccs and for achepa for achepa diagnoses')
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

# for the adapters
def get_datav2(data_paths, dataset_name): 
        data_class = codiespDataset(data_paths, dataset_name)
        codiesp_dataset, all_labels = data_class.load_codiesp_dataset()
        #codiesp_dataset, all_labels = data_class.load_codiesp_mixed_lng_dataset(train_lng)
        if dataset_name == 'mimic':
            codiesp_dataset = codiesp_dataset.rename_column("TEXT", "notes")
        codiesp_dataset = data_class.tokenize_dataset(codiesp_dataset)
        codiesp_dataset = data_class.transform_labels(codiesp_dataset, 
                                                      all_labels=all_labels)
        num_labels = len(all_labels)
        print(codiesp_dataset.num_rows, num_labels)
        return codiesp_dataset["train"], codiesp_dataset["validation"], codiesp_dataset["test"], all_labels

def get_nonzero_cols_n_rows(label_ids_tmp, selected_cols=set([248, 267, 246, 93, 259])):
    check_for_one_class = label_ids_tmp.sum(axis=0)
    not_one_class_present = [idx for idx in range(len(check_for_one_class)) if check_for_one_class[idx] not in (0, len(label_ids_tmp))]
    not_one_class_present_selected = [not_one_class_present.index(idx) for idx in selected_cols if idx in not_one_class_present]
    return not_one_class_present, not_one_class_present_selected


def load_zero_shot_ccs_codes(data_paths):
    zero_shot_ccs = pd.read_csv(data_paths['zero_shot_ccs_path'])
    return zero_shot_ccs

def get_zero_shot_ccs_idx(label_to_pos, zero_shot_ccs): 
    idx = zero_shot_ccs['ccs'].map(label_to_pos)
    return idx.valuesm

def load_zero_shot_diagnoses_codes(): 
    pass
