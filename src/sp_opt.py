from pathlib import Path

import optuna
import pandas as pd
import yaml
from optuna.integration import PyTorchLightningPruningCallback

from sp_cli import SpLightningCLI
from sp_data import SpDataModule
from sp_model import SpModel


class SpOpt:
    def __init__(self, model_name: str, n_trials: int = 10, epochs: int = 12) -> None:
        self.model_name = model_name
        self.n_trials = n_trials
        self.epochs = epochs

        self.config_path = Path("src/configs/")
        self.opt_config_path = self.config_path / "config_opt.yaml"
        self.model_config_path = self.config_path / f"config_{model_name}.yaml"

    def objective(self, trial: optuna.Trial):
        monitor_metric = "Val Macro Accuracy"

        with open(self.opt_yaml, "r") as file:
            config = yaml.safe_load(file)

        with open(self.model_yaml, "r") as file:
            model_config = yaml.safe_load(file)
            config["model"]["model_params"] = model_config["model"]["model_params"]

        config["trainer"]["max_epochs"] = self.epochs
        config["optimizer"]["init_args"]["lr"] = trial.suggest_float(
            "learning_rate", 1e-4, 1e-1, log=True
        )
        config["optimizer"]["init_args"]["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-5, 1e-2
        )
        config["lr_scheduler"]["init_args"]["gamma"] = trial.suggest_float(
            "gamma", 0.6, 0.9
        )
        config["optimizer"]["class_path"] = trial.suggest_categorical(
            "opt", ["torch.optim.AdamW", "torch.optim.RAdam"]
        )

        cli = SpLightningCLI(
            SpModel, SpDataModule, run=False, args=config, save_config_callback=None
        )

        trainer = cli.trainer
        model = cli.model
        datamodule = cli.datamodule

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=monitor_metric
        )
        trainer.callbacks.append(pruning_callback)

        trainer.fit(model, datamodule=datamodule)
        return trainer.callback_metrics[monitor_metric].item()

    def sp_optimize(self):
        results_location = Path(f"studies/{self.model_name}.csv")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        df = study.trials_dataframe()

        with open(results_location, "a") as f:
            df.to_csv(f, mode="a", header=f.tell() == 0)
