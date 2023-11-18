from pathlib import Path

import yaml
import optuna
import torch
import pandas as pd

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI

from sp_data import SpDataModule
from sp_model import SpModel


class SpLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")

def cli_main(config_name):
    get_config = lambda config_name: f'src/configs/config_{config_name}.yaml'

    with open(get_config(config_name), 'r') as file:
        config = yaml.safe_load(file)

    for i in range(config['data']['num_folds']):
        config['data']['fold'] = i
        cli = SpLightningCLI(
            SpModel,
            SpDataModule,
            run=False,
            args=config,
            save_config_kwargs={"overwrite": True}
        )

        trainer = cli.trainer
        model = cli.model
        datamodule = cli.datamodule
        trainer.fit(model, datamodule=datamodule)

    print("Finished Training.")


def objective(trial: optuna.Trial, model_name: str, epochs: int):
    get_config = lambda config_name: f'src/configs/config_{config_name}.yaml'
    opt_yaml = get_config('opt')
    model_yaml = get_config(model_name)
    monitor_metric = 'Val Macro Accuracy'

    with open(opt_yaml, 'r') as file:
        config = yaml.safe_load(file)

    with open(model_yaml, 'r') as file:
        model_config = yaml.safe_load(file)
        config['model']['model_params'] = model_config['model']['model_params']

    config['trainer']['max_epochs'] = epochs
    config['optimizer']['init_args']['lr'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    config['optimizer']['init_args']['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-2)
    config['lr_scheduler']['init_args']['gamma'] = trial.suggest_float('gamma', 0.6, 0.9)
    config['optimizer']['class_path'] = trial.suggest_categorical('opt', ['torch.optim.AdamW', 'torch.optim.RAdam'])

    cli = SpLightningCLI(SpModel, SpDataModule, run=False, args=config, save_config_callback=None)

    trainer = cli.trainer
    model = cli.model
    datamodule = cli.datamodule

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor=monitor_metric)
    trainer.callbacks.append(pruning_callback)

    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics[monitor_metric].item()

def opt_main(model_name: str,  n_trials: int=10, epochs: int=12):
    results_location = Path(f'studies/{model_name}.csv')

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name, epochs), n_trials=n_trials)
    df = study.trials_dataframe()

    with open(results_location, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')

    opt_main('swin')

    # cli_main()
