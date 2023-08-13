from pathlib import Path

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI

from sp_data import SpDataModule
from sp_model import SpModel


class SpLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")

    def finish(self):
        logger_args = self.config["fit"]["trainer"]["logger"]["init_args"]
        config_file = "config.yaml"
        ckpt_folder = Path(logger_args["save_dir"])
        project_name = logger_args["project_name"]

        project_ckpts_path = ckpt_folder / project_name
        last_folder = self.get_last_experiment_folder(project_ckpts_path)

        if last_folder:
            config_path = ckpt_folder / config_file
            new_config_path = last_folder / config_file
            config_path.rename(new_config_path)

    @staticmethod
    def get_last_experiment_folder(project_ckpts_path):
        subfolders = list(project_ckpts_path.glob("*"))
        sorted_subfolders = sorted(subfolders, key=lambda p: p.stat().st_ctime)
        return sorted_subfolders[-1] if sorted_subfolders else None


def cli_main():
    cli = SpLightningCLI(
        SpModel,
        SpDataModule,
        save_config_kwargs={"overwrite": True},
    )
    cli.finish()


if __name__ == "__main__":
    cli_main()
