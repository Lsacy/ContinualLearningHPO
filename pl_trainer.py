import os
import torch
import pandas as pd
import logging
import optuna
from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pl_dataloader import dataset_loader, create_data_loader
from pathlib import Path

from typing import List
from ContinualClinicalNLU.pipeline_test.hpo_optuna import objective, study_results
from utils import sequence_keys

import shutil
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class BaseTrainer:
    def __init__(self, config, loggers, num_labels, *args, **kwargs):
        self.config = config
        self.loggers = loggers
        self.logger = loggers.tensorboard
        self.custom_logger = loggers.custom_logger
        self.n_trial = config.train.n_trials
        self.hpo_optuna_study = config.train.hpo_optuna_study
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        self.model_name_or_path = config.model.name_or_path
        self.n_jobs = config.train.n_jobs
        self.base_model = config.model.name_or_path


    def create_study(self, study_name):
        return optuna.create_study(
            study_name=study_name, direction='maximize', storage=self.hpo_optuna_study,
            load_if_exists=True, pruner=optuna.pruners.HyperbandPruner()
        )

    def move_best_model(self, best_model_folder, experiment_name):
        optuna_all_results = Path(self.config.paths.optuna_all_results)
        old_best_model_path = optuna_all_results / best_model_folder

        optuna_best_results = Path(self.config.paths.optuna_best_results, experiment_name)
        optuna_best_results.mkdir(parents=True, exist_ok=True)

        best_model_path = optuna_best_results / best_model_folder / 'best-model.ckpt'

        if not old_best_model_path.exists():
            raise FileNotFoundError(f"Source directory {old_best_model_path} does not exist.")

        shutil.move(str(old_best_model_path), str(optuna_best_results))  # move the folder

        return str(best_model_path)

    def save_best_model(self, sequence, best_trial, config):
        """ best models from all trials are moved to a new folder."""
        sequence = sequence_keys(sequence, config)
        best_model_folder = self.best_model_folder(sequence, best_trial) # create a name of the best model folder
        return self.move_best_model(best_model_folder, self.experiment_name) # move the best model folder to a new folder
    
    def best_model_folder(self, sequence, best_trial):
        # name of the best model folder, must be exactly the same as the folder name in optuna_all_results
        # created by the optuna study object
        sequence = ('_').join(sequence)
        return f'{self.i}_{sequence}_trial-{best_trial.number}_lr-{best_trial.params["lr"]}'
        
    def find_model(self, i, sequence: list, path: str = None):
        """ find model in path with the name: position_sequence. /n
            returns a list of matched models"""
        paths = []
        sequence = sequence_keys(sequence, self.config)
        model_name = ('_').join(sequence) # name of the model
        model_name = f'{i}_{model_name}'
        if not os.path.exists(path):
            # create a new folder
            os.makedirs(path, exist_ok=True)
        for file in os.listdir(path):
            # if file contains string trained
            if model_name in file:
                logging.info(f'model found: {file}')
                # get pwd of the matched model
                paths.append(f'{path}{file}/best-model.ckpt')
        return paths


class ContinualTrainer(BaseTrainer):
    '''needs to be initialized with config, logger, num_labels '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trained_models = []
        self.results = []
        self.sequence = []
        self.model_name_or_path = kwargs.get('model_name_or_path', self.model_name_or_path)
        self.experiment_name = kwargs.get('experiment_name', self.config.experiment_name)
    def run(self, sequence: List, dataset: dataset_loader) -> List[str]:
        ''' sequence: list of sequence to train on
            dataset: dataset_loader object'''
        logging.info(f"Start training a sequence of {len(sequence)}: {sequence}")
        test_results_df = pd.DataFrame(columns=sequence, index=sequence)

        # transform the sequence via self.config.train.sequence_mapping
        for i, sequence_item in enumerate(sequence):
            self.sequence.append(sequence_item)
            logging.info(f"Training {i}-th item: {sequence_item} out of {sequence}")


            # load best model if it exists else HPO with optuna
          
            if paths := self.find_model(i, self.sequence, f'{self.config.paths.optuna_best_results}{self.experiment_name}/'): # can the path be specified in the config?
                self.model_name_or_path = paths[0]
                self.custom_logger.found_existing_model(paths)
            else:
                logging.info(f'Model not found -> start training {sequence_item} on the sequence {i}_{sequence}')
                self.model_name_or_path = Optuna_SingleTrainer(i=i, config=self.config,
                                        loggers = self.loggers, 
                                        num_labels=self.num_labels, 
                                        sequence_item = sequence_item,
                                        dataset = dataset, 
                                        model_name_or_path = self.model_name_or_path,
                                        experiment_name=self.experiment_name,
                                        sequence = self.sequence,
                                        ).train(sequence) 
                self.trained_models.append((i, sequence, self.model_name_or_path))
                logging.info(f"finished hpo for {sequence_item}, checkpoint path: {self.model_name_or_path}")

            # load model for evaluation, check for which pl base model to use
            pl_model = self.config.call_model(self.base_model)
            model = pl_model.load_from_checkpoint(config =self.config, 
                                                checkpoint_path=self.model_name_or_path,
                                                num_labels=self.num_labels)
            # evaluate
            
            evaluator = Evaluator(self.logger, model, self.tokenizer, self.config)
            test_results_df, results = evaluator.backward_test(i, sequence, dataset, test_results_df)

            # delete model and free up memory
            del model
            torch.cuda.empty_cache()
            self.results.append(results)
            print(test_results_df)

        return self.trained_models, self.results, test_results_df
    



class Optuna_SingleTrainer(BaseTrainer):
    def __init__(self, config, loggers, num_labels, *args, **kwargs):
        super().__init__(config, loggers, num_labels)
        
        # Extract additional keyword arguments specific to SingleTrainer
        self.i = kwargs.get('i', None)
        self.sequence_item = kwargs.get('sequence_item', None)
        self.dataset = kwargs.get('dataset', None)
        self.model_name_or_path = kwargs.get('model_name_or_path', None)
        self.experiment_name = kwargs.get('experiment_name', None)
        self.sequence = kwargs.get('sequence', None)

        # Initialize additional properties
        self.params_train = {
            'batch_size': int(self.config.train.batch_size),
            'shuffle': self.config.train.shuffle,
            'num_workers': int(self.config.train.num_workers),
            'pin_memory': self.config.train.pin_memory
        }
        self.params_val_test = {
            'batch_size': int(self.config.train.batch_size),
            'shuffle': self.config.dev.shuffle,
            'num_workers': int(self.config.train.num_workers),
            'pin_memory': self.config.train.pin_memory
        }


    def train(self, sequence):

        train = create_data_loader(self.dataset, self.sequence_item, self.tokenizer, "train", self.params_train)
        dev = create_data_loader(self.dataset, self.sequence_item, self.tokenizer, "dev", self.params_val_test)

        study_name = f"{self.i}-{self.sequence}-study"  # Unique identifier of the study. where is it located when saved?

        study = self.create_study(study_name)

        study.optimize(lambda trial: objective(trial=trial,
                                                config=self.config, 
                                                sequence_item=self.sequence_item, 
                                                train=train,
                                                dev=dev, 
                                                logger=self.logger, 
                                                model_name_or_path=self.model_name_or_path,
                                                num_labels=self.num_labels,
                                                sequence = sequence,
                                                position= self.i,
                                                experiment_name=self.experiment_name,
                                                ),  
                                                n_trials=self.n_trial,
                                                )

        best_trial = study_results(study)
        return self.save_best_model(self.sequence, best_trial, self.config)


class Evaluator():
    def __init__(self, logger, model, tokenizer, config):
        self.logger = logger
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.params_val_test = {'batch_size' : int(self.config.train.batch_size),
              'shuffle': config.dev.shuffle,
              'num_workers':int(config.train.num_workers),
              'pin_memory': config.train.pin_memory}

    def run(self, sequence_item, dataset):
        test = create_data_loader(dataset, sequence_item, self.tokenizer, "test", self.params_val_test)
        trainer = Trainer(logger=self.logger, num_sanity_val_steps=0)
        with torch.cuda.amp.autocast():
            results = trainer.test(self.model, test)
            logging.info(f'evaluated on {sequence_item}. Results: {results}.')
        return results
    
    def backward_test(self, i, sequence: List[str], dataset: dataset_loader, frame) -> pd.DataFrame():
        '''
        evaluate on all trained sequence so far to detect catastrophic forgetting
        '''
        results = []
        logging.info(f'backward test on the sequence: {sequence}')
        logging.info(f'using model {self.model.model_id} for evaluation')
        for idx, sequence_item in enumerate(sequence):
            logging.info(f'test on item: {sequence_item}')
            result = self.run(sequence_item, dataset)
            frame.iloc[i, idx] = result
            results.append(result)
        return frame, results