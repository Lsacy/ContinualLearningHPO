import os
from pytorch_lightning.loggers import TensorBoardLogger
from config import Config
from pl_dataloader import dataset_loader
from utils import current_time, set_seeds, CustomLogger, Loggers, save_csv
from pl_trainer import ContinualTrainer
from ContinualClinicalNLU.pipeline_test.pl_model import XLMClassification_Optuna, PubMedBERT_Optuna
import fire
import logging

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


config_path = '/pvc/ContinualClinicalNLU/pipeline_test/configs/config_remote_domains.yml'

# /opt/conda/bin/python /pvc/ContinualClinicalNLU/pipeline_test/main.py --experiment_name "domains_test_most_similiar" --sequence "a,b,c" --config_path "/pvc/ContinualClinicalNLU/pipeline_test/configs/config_remote_domains.yml"

def main(experiment_name=None, sequence='e,f,d', config_path=config_path):

    pl_models = {
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': PubMedBERT_Optuna,
        'xlm-roberta-base': XLMClassification_Optuna,
    }

    if config_path is not None:
        config = Config(config_path, pl_models)

    # check for additional arguments and add your own here
    if experiment_name is not None:
        config.experiment_name = experiment_name   
    print(type(sequence))
    if sequence is not None:
        # convert string to list, remove spaces
        if isinstance(sequence, tuple):
            sequence = ','.join(sequence)
        sequence = [item.strip() for item in sequence.split(',')]
        sequence = [config.train.sequence_mapping[item] for item in sequence]
    else:
        sequence = [config.train.sequence_mapping[item] for item in config.train.sequence]

    config.train.sequence = sequence
    set_seeds(seed=config.train.seed)

    ## Set up logging
    # Tensorboard logger, setup your naming convention here
    tensorboard_logger = TensorBoardLogger(save_dir=config.paths.tensorboard_path, name=f'{config.experiment_name}_{config.model.name_or_path}')
    # Custom logger, uses the initialized_logger from utils.py
    custom_logger = CustomLogger(config)
    # Loggers class, passed to the trainer for logging purposes
    loggers = Loggers(tensorboard_logger, custom_logger)

    logging.info(f'Training Sequence: {config.train.sequence}')
    logging.info(f'Loading dataset: {config.dataset.path}')
    # load dataset and get number of labels
    dataset = dataset_loader(config, custom_logger)
    num_labels = dataset.num_labels


    # Run HPO
    cl_trainer = ContinualTrainer(config, loggers, num_labels)
    _, _, results_df = cl_trainer.run(config.train.sequence, dataset)

    # Save results
    csv_path = save_csv(results_df, config)

    # print the loggers
    loggers.custom_logger.final_print(csv_path)


if __name__ == "__main__":
    # pass your arguments here
    # python main.py --experiment_name "test" --sequence "1,2,3,4,5,6,7,8,9,10" --config_path "/pvc/ContinualClinicalNLU/pipeline_test/configs/config_remote_languages.yml"
    # additional arguments and models can be added in the main function

    fire.Fire(main)