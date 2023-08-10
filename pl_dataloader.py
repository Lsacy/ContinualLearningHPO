import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import os
import glob
from typing import Tuple
from torch.utils.data import Dataset as torch_Dataset, DataLoader
from utils import CustomLogger
from config import Config
import torch
import logging

# to do: 

## implement the one hot encoding in a way that each time a new label is encountered, it is added to the list of labels, and the one hot encoded vector is updated accordingly
## seperate the loading of the test data from the training data, so that the test data is not loaded into the memory during training
## when loading 


class dataset_loader:   
    def __init__(self, config, logger: CustomLogger):
        self.logger = logger
        self.path = config.dataset.path
        self.randomize = config.dataset.randomize # randomize the order of the dataset
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        self.num_entries = config.dataset.num_entries # num of samples to be loaded, if all then all samples are loaded, otherwise only a small subset e.g. '256'
        self.specific_sets = config.train.sequence # specific sets to be loaded, if None then all sets are loaded, otherwise only the specified sets e.g. ['mimic', 'codiesp']
        self.train_size = config.dataset.train_size
        self.dev_size = config.dataset.dev_size
        self.test_size = config.dataset.test_size
        self.frames, self.id2label, self.label2id = self.get_dataset_dict() # dictionary of dataframes, dictionary mapping nummeric labels to text labels, dictionary mapping text labels to nummeric labels
        self.num_labels = self.get_num_labels() # number of labels in the dataset

    def get_dataset_dict(self) -> Tuple[pd.DataFrame, dict, dict]:
        """ Create a dictionary of datasets from a dictionary of dataframes. """

        frames = self.load_multiple_dfs()

        frames, id2label, label2id = self.convert_labels(frames)
        
        # keep only the specific sets of the dataset
        if self.specific_sets:
            frames = {k: v for k, v in frames.items() if any(i in k for i in self.specific_sets)}

        # check for the number of entries in the train, dev and test sets
        self.train_size = self.get_size(frames, self.train_size, 'train')
        self.dev_size = self.get_size(frames, self.dev_size, 'dev')
        self.test_size = self.get_size(frames, self.test_size, 'test')

        # drop unnecessary columns
        columns_to_keep =['text', 'labels', 'id2label']
        for i in frames.keys():
            frames[i].drop([col for col in frames[i].columns if col not in columns_to_keep], axis=1, inplace=True)

        # log the number of entries in the train, dev and test sets    

        return frames, id2label, label2id
    
    def get_size(self, frames: dict, size: int, split: str) -> int:
        """ take the lowest common denominator of the number of entries in the train, dev and test sets. """
        if size == 'all':
            return size
        # select the keys of the frames dictionary that contain the split name
        keys = [i for i in frames if split in i]
        sizes = [len(frames[i]) for i in keys]
        new_size = min(len(frames[i]) for i in keys)
        new_size = min(new_size, size)

        self.logger.size_info(split, new_size, sizes)
        return new_size

    
    def load_multiple_dfs(self) -> dict:
        """ screens all .csv files in a given directory, with the help of self.load_single_df function. /n
        return: dictionary with dataframe(s), specific_sets: str, default None, could be 'mimic' or 'codiesp' to load only one set of dataset. """
        
        # get all files in the directory
        all_files = glob.glob(os.path.join(self.path, "*.csv"))
            
        #    

        key_names = [('.').join(i.split('/')[-1].split('.')[:-1]) for i in all_files]  # get the file name without the path to use as dictionary key

        return {
            i: self.load_single_df(all_files[idx]).head(self.num_entries)
            if self.num_entries != 'all'
            else self.load_single_df(all_files[idx])
            for idx, i in enumerate(key_names)
        }

    def load_single_df(self, all_files: str) -> pd.DataFrame:
        """ Load a single .csv file, transform the labels from text into nummeric, one-hot encoded version. """

        converter = {'labels': eval}
        df = pd.read_csv(all_files, converters = converter)
        df.rename(columns={'TEXT': 'text', 'notes': 'text'}, inplace=True)

        # randomize the order of the entire dataframe in place
        if self.randomize:
            df = df.sample(frac=1).reset_index(drop=True)

        return df
    
    def convert_labels(self, frames: dict) -> Tuple[pd.DataFrame, dict, dict]:
        """Convert the labels from text to nummeric, one-hot encoded version."""

        # get all unique labels
        unique_labels = self.get_unique_labels(frames)
        
        # create label mappings
        id2label = dict(enumerate(unique_labels))
        label2id = {label: i for i, label in enumerate(unique_labels)}

        # convert text labels to nummeric labels
        frames = self.convert_text_labels_to_numeric(frames, label2id)

        # create one-hot encoded vectors
        frames = self.create_one_hot_vectors(frames, unique_labels)
        return frames, id2label, label2id

    def convert_ohe_to_labels(self, ohe: list) -> list:
        labels = []
        for idx, item in enumerate(ohe):
            if item == 1:
                labels.append(self.id2label[idx])
                print(idx)
        return labels

    def get_unique_labels(self, frames: dict) -> set:
        """get all unique labels from frames"""
        unique_labels = set()
        for frame in frames.values():
            unique_labels = unique_labels.union(set(frame['labels'].explode().unique()))
        unique_labels = sorted(unique_labels)

        return unique_labels
        
    def convert_text_labels_to_numeric(self, frames: dict, label2id: dict) -> pd.DataFrame:
        """convert text labels to numeric labels in frames
            select only those specific frames that are given in the 'specific_sets' argument
        """
        if self.specific_sets:
            frames = {k: v for k, v in frames.items() if any(i in k for i in self.specific_sets)}

        for frame in frames.values():
            # create empty 'labels_ohe' and 'labels_ids' columns for each dataframe inside the 
            # frames dictionary
            frame['labels_ohe'] = ''
            frame['labels_ids'] = ''
            #locate the position of 'labels_ids' column for populating the 'labels_ids' in the next
            # step, 'labels_ids' are nummeric labels
            column_idx = frame.columns.get_loc('labels_ids')
            # populating 'labels_ids' column with nummeric labels, each corresponding 
            # an entry from 'labels' column
            for idx, _ in enumerate(frame['labels']):
                frame.iat[idx, column_idx] = [label2id[i] for i in frame['labels'][idx]]
        
        return frames


    def create_one_hot_vectors(self, frames: dict, unique_labels: set) -> pd.DataFrame:
        """create one-hot encoded vectors for numeric labels in frames"""
        for frame in frames.values():
            column_idx = frame.columns.get_loc('labels_ohe')
            column_idx2 = frame.columns.get_loc('labels')
            ohe_length = len(unique_labels)
            for idx, i in enumerate(frame['labels_ids']):
                zeros = np.zeros(ohe_length)
                for j in i:
                    zeros[j] = 1
                frame.iat[idx, column_idx] = zeros
                frame.iat[idx, column_idx2] = zeros

        return frames

    

    def convert_ohe_to_labels(self, ohe: list) -> list:
        labels = []
        for idx, item in enumerate(ohe):
            if item == 1:
                labels.append(self.id2label[idx])
                print(idx)
        return labels

    
    
    def get_num_labels(self) -> int:
        """ len(labels) to retrieve the amount of labels.."""
        keys = list(self.frames.keys())
        return len(self.frames[keys[0]]['labels'][0])
    
    
class MyDataset(torch_Dataset):
    """takes a DatasetDict from the custom dataset_loader class,
        returns"""
    def __init__(self, dataset: dataset_loader, sequence_item:str, tokenizer, split:str):
        super().__init__()
        if all(sequence_item not in i for i in dataset.frames.keys()):
            raise ValueError(f'{sequence_item} is not found in the dataset!')
        
        # find the full key of the dataset to use for the split (train, dev, test)
        for i in dataset.frames.keys():
            if sequence_item in i:
                sequence_item = ('_').join(i.split('_')[:-1])
                break

        # load only the specific split of the dataset with the split_size
        if getattr(dataset, f'{split}_size') != 'all':
            self.items = dataset.frames[f'{sequence_item}_{split}'].head(getattr(dataset, f'{split}_size'))
            
        else:
            self.items = dataset.frames[f'{sequence_item}_{split}']
        logging.info(f'Loaded {len(self.items)} items from {sequence_item}_{split} split')

        self.text = self.items['text']
        self.labels = self.items['labels']
        self.tokenizer = tokenizer
        self.max_len = 512

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> dict:
        text = self.text[idx]
        inputs = self.tokenizer.encode_plus(text, 
                                            padding='max_length', 
                                            max_length = self.max_len, 
                                            truncation = True, 
                                            return_tensors='pt')
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = torch.from_numpy(self.labels[idx])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
def create_data_loader(dataset, language, tokenizer, split, params):
    dataset = MyDataset(dataset, language, tokenizer, split)
    return DataLoader(dataset, **params)

class dataset_loader_stratified:   # different pre split files!!
    def __init__(self, config):
        self.path = config.dataset.path
        self.num_entries = config.dataset.num_entries
        self.specific_sets = config.train.languages
        self.frames = self.get_dataset_dict()
        self.num_labels = self.get_num_labels()

    def load_single_df(self, file: str) -> pd.DataFrame:
        """ Load a single .csv file, transform the labels from text into nummeric, one-hot encoded version."""

        converter = {'labels': eval}
        df = pd.read_csv(file, converters = converter)
        df.rename(columns={'TEXT': 'text', 'notes': 'text'}, inplace=True)
        df['labels'] = df['labels'].apply(np.array)
        return df

    def load_multiple_dfs(self) -> dict:
        """ Load all .csv files in a given directory, using the load_and_label_df function for each file by default, /n
        return a dictionary with the dataframe, and a dictionary mapping the nummeric labels to the text labels /n
        specific_sets: str, default None, could be 'mimic' or 'codiesp' to load only one set of dataset."""
        
        # get all files in the directory
        all_files = glob.glob(os.path.join(self.path, "*.csv"))

        # filter the files if specific_sets is given
        for i in self.specific_sets:
            if all(i not in file for file in all_files):
                raise ValueError(f'{i} is not found in the data directory!')

        if self.specific_sets:
            all_files = [i for j in self.specific_sets for i in all_files if j in i]

        file_names = [i.split('/')[-1].split('.')[0] for i in all_files]  # get the file name without the path to use as dictionary key
        print(file_names)
        # load all files into the dictionaries
        frames = {}
        for idx, i in enumerate(file_names):
            j = i.split('_')
            i = f'{j[0]}_{j[1]}_{j[-1]}'
            if self.num_entries != 'all':
                frames[i] = self.load_single_df(all_files[idx]).head(self.num_entries)
            else:
                frames[i] = self.load_single_df(all_files[idx])

        return frames


    def get_dataset_dict(self) -> pd.DataFrame:
        """ Create a dictionary of datasets from a dictionary of dataframes."""

        return self.load_multiple_dfs()
    
    def get_num_labels(self) -> int:
        """ len(labels) to retrieve the amount of labels."""
        keys = list(self.frames.keys())
        return len(self.frames[keys[0]]['labels'][0])




if __name__ == '__main__':
    config = Config('/pvc/ContinualClinicalNLU/pipeline_test/config_remote_languages.yml')
    dataset = dataset_loader(config)
    print(dataset.num_labels)
    tokenizer =  AutoTokenizer.from_pretrained(config.model.tokenizer)
    train_dataset = MyDataset(dataset, 'mimic', tokenizer, 'train')
    params = {'batch_size' : config.train.batch_size,
              'shuffle': True,
              'num_workers': 3}
    train = DataLoader(train_dataset, **params)

    print(type(train))
    print(len(train_dataset))