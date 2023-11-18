import torch
import yaml

from sp_cli import SpLightningCLI
from sp_data import SpDataModule
from sp_model import SpModel


def cli_main(config_name):
    get_config = lambda config_name: f"src/configs/config_{config_name}.yaml"

    with open(get_config(config_name), "r") as file:
        config = yaml.safe_load(file)

    for i in range(config["data"]["num_folds"]):
        config["data"]["fold"] = i
        cli = SpLightningCLI(
            SpModel,
            SpDataModule,
            run=False,
            args=config,
            save_config_kwargs={"overwrite": True},
        )

        trainer = cli.trainer
        model = cli.model
        datamodule = cli.datamodule
        trainer.fit(model, datamodule=datamodule)

    print("Finished Training.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    model_name = "convnext"
    cli_main(model_name)
