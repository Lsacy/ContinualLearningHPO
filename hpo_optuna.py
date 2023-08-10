import optuna
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
os.environ['WANDB_API_KEY'] = '57b9934412a21d921b33b1e1f30edaea2b1ddb03'

import torch
import logging
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import List
import gc

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from utils import set_seeds, log_model, sequence_keys
import math
import optuna
import uuid

#import ray 

## study results are saved in results.csv file
## f"/pvc/optuna_best_results/{study.study_name}_results.csv", "w"

def objective(trial, config, sequence_item, train, dev, logger, model_name_or_path, num_labels, sequence, position, experiment_name):
    
    pl_model = config.call_model(config.model.base_model)
    # define hyperparameters to optimize
    learning_rate = trial.suggest_float('lr', 5e-6, 8e-5)
    warmup_steps = trial.suggest_categorical('warmup_steps', [0, 100, 250, 500, 750])
    hidden_dropout = trial.suggest_categorical('hidden_dropout', [0.1])
    attention_dropout = trial.suggest_categorical('attention_dropout', [0.1])
    gradient_accumulation_steps =  trial.suggest_categorical('gradient_accumulation_steps', [4])

    # convert sequence from list to string
    # sequence = '_'.join(sequence)
    
    # load model
    past_sequence = sequence[:position+1]
    past_sequence = sequence_keys(past_sequence, config) # rename sequence full names to keys
    past_sequence = '_'.join(past_sequence)

    if model_name_or_path == config.model.name_or_path:
        model = pl_model(config, num_labels=num_labels,
                                         lr=learning_rate,
                                         warmup_steps=warmup_steps,
                                         hidden_dropout=hidden_dropout,
                                         attention_dropout=attention_dropout,
                                         gradient_accumulation_steps=gradient_accumulation_steps,
                                         model_id=f'{position}_{past_sequence}_trial-{trial.number}_lr-{learning_rate}_{uuid.uuid4().hex[:4]}'
                                        )

    else:
        model = pl_model.load_from_checkpoint(checkpoint_path=model_name_or_path,
                                          config=config, 
                                          num_labels=num_labels,
                                          lr=learning_rate,
                                          warmup_steps=warmup_steps,
                                          hidden_dropout=hidden_dropout,
                                          attention_dropout=attention_dropout,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          model_id=f'{position}_{past_sequence}_trial-{trial.number}_lr-{learning_rate}_{uuid.uuid4().hex[:4]}'
                                          )
    log_model(model, model_name_or_path, trial, sequence_item, sequence)

    model.training_steps= math.ceil(len((train)) // model.gradient_accumulation_steps * config.train.epochs_per_lang)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='torchmetrics.auroc',  # change to your validation metric name
        mode='max',  # or 'min' depending on the metric you are using
        save_top_k=1,  # saves only the best model
        verbose=True,
        dirpath=f'{config.paths.optuna_all_results}{position}_{past_sequence}_trial-{trial.number}_lr-{trial.params["lr"]}',  # directory path to save model checkpoints
        filename='best-model'  # filename for the checkpoint file
    )
    pruned_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor='torchmetrics.auroc')
    early_stop_callback = EarlyStopping(monitor='torchmetrics.auroc', 
                                                patience=4, 
                                                mode='max')

    # setup of trainer and training
    trainer = Trainer(max_epochs=config.train.epochs_per_lang, logger=logger,
                     accumulate_grad_batches=model.gradient_accumulation_steps,
                     callbacks=[checkpoint_callback, pruned_callback, early_stop_callback])

    trainer.fit(model, train, dev)
    best_model_score = checkpoint_callback.best_model_score.item()
    trial.report(best_model_score, step=trainer.current_epoch)
    #trial.report(trainer.callback_metrics['torchmetrics.auroc'], step=trainer.current_epoch)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return best_model_score


def study_results(study):
    ''' 
    - prints out the results of the study, 
    - writes the results to results.csv file,
    - returns study.best_trial
    '''
    trials = study.trials
    best_trial = study.best_trial
    pruned_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))
    complete_trials = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
    print("Study statistics: ")
    print(' study for: ', study.study_name)
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", pruned_trials)
    print("  Number of complete trials: ", complete_trials)
    print("Best trial:")
    print("  Number: ", best_trial.number)
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f'    {key}: {value}')
    logging.info(f'Best trial for {study.study_name} out of {complete_trials}/{len(trials)}: Trial number: {best_trial.number}, auroc: {best_trial.value}, params: {best_trial.params}')
    return best_trial

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()


    # Add stream handler of stdout to show the messages
    logging.basicConfig(level=logging.INFO)
    set_seeds(42)

    study_name = "study-optuna-updated"  # Unique identifier of the study.
    storage_name = f"sqlite:////pvc/{study_name}.db"
    study = optuna.create_study(study_name=study_name, direction ='maximize', storage=storage_name, 
                                load_if_exists=True, pruner=optuna.pruners.HyperbandPruner())


    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    print(study.best_params) 

    