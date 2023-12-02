import torch
import yaml

from sp_cli import SpLightningCLI
from sp_data import SpDataModule
from sp_model import SpModel


def get_config(config_name: str) -> dict:
    conf = f"src/configs/config_{config_name}.yaml"
    with open(conf, "r") as file:
        config = yaml.safe_load(file)
    return config


def cli_main(config_name: str) -> None:
    config = get_config(config_name)
    num_folds = config.get("data").get("num_folds", 1)

    for i in range(num_folds):
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


def cli_test(ckpt):
    from pytorch_lightning import Trainer

    model = SpModel.load_from_checkpoint(ckpt)
    datamodule = SpDataModule("Databases/DB_TOP25")

    trainer = Trainer(accelerator="gpu", devices=1, num_nodes=1, logger=False)
    trainer.test(model, datamodule=datamodule)

    print("Finished Testing.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
